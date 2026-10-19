"""Optional DistilBERT inference adapter.

Training is intentionally a separate, opt-in workflow because model downloads
and fine-tuning require substantial storage and compute. No results are
invented when an artifact is absent.
"""

from __future__ import annotations

from pathlib import Path


class TransformerUnavailable(RuntimeError):
    """Raised when an optional trained transformer artifact is unavailable."""


class TransformerClassifier:
    def __init__(self, model_directory: Path) -> None:
        self.model_directory = Path(model_directory)
        self.model = None
        self.tokenizer = None
        self.labels: list[str] = []

    def load(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise TransformerUnavailable("Install transformers and torch first.") from exc
        if not self.model_directory.exists():
            raise TransformerUnavailable(
                "No fine-tuned transformer artifact exists. Run the optional training workflow."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_directory)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_directory)
        self.labels = [
            self.model.config.id2label[index]
            for index in range(self.model.config.num_labels)
        ]

    def predict(self, text: str) -> tuple[str, float]:
        if self.model is None or self.tokenizer is None:
            self.load()
        import torch

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        index = int(torch.argmax(probabilities))
        return self.labels[index], float(probabilities[index])

    def token_attributions(self, text: str, limit: int = 10) -> list[tuple[str, float]]:
        """Return gradient magnitudes for tokens when an artifact is loaded."""
        if self.model is None or self.tokenizer is None:
            self.load()
        import torch

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        embeddings = self.model.get_input_embeddings()(inputs["input_ids"]).detach()
        embeddings.requires_grad_(True)
        outputs = self.model(inputs_embeds=embeddings, attention_mask=inputs.get("attention_mask"))
        target = int(outputs.logits.argmax(dim=-1)[0])
        outputs.logits[0, target].backward()
        scores = embeddings.grad.abs().sum(dim=-1)[0].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        ranked = sorted(zip(tokens, scores), key=lambda item: item[1], reverse=True)
        return [(token, float(score)) for token, score in ranked[:limit]]
