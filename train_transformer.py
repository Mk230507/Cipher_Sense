"""Optional, real DistilBERT fine-tuning workflow.

This command downloads a pretrained model only when explicitly requested:
python train_transformer.py --output-dir artifacts/distilbert
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ciphersense.classifier import LABELS, load_dataset


class MessageDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = [labels.index(label) for label in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/distilbert"))
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    start = time.perf_counter()

    texts, labels = load_dataset()
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.25, random_state=args.seed, stratify=labels
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={label: index for index, label in enumerate(LABELS)},
    )
    train = DataLoader(MessageDataset(train_texts, train_labels, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True)
    test = DataLoader(MessageDataset(test_texts, test_labels, tokenizer, args.max_length), batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for _ in range(args.epochs):
        for batch in train:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in test:
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions = model(**batch).logits.argmax(dim=-1)
            correct += int((predictions == batch["labels"]).sum())
            total += len(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps({"accuracy": correct / total, "device": str(device), "seconds": time.perf_counter() - start, "seed": args.seed}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved transformer artifact and measured metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
