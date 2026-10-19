"""Measured evaluation, robustness, and comparison artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(model, texts: list[str], labels: list[str], name: str) -> dict[str, object]:
    start = time.perf_counter()
    predictions = model.predict(texts)
    latency_ms = (time.perf_counter() - start) * 1000 / max(len(texts), 1)
    return {
        "model": name,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "latency_ms_per_message": latency_ms,
        "classification_report": classification_report(labels, predictions, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "labels": sorted(set(labels)),
    }


def save_confusion_plot(result: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = result["labels"]
    matrix = result["confusion_matrix"]
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{result['model']} confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def robustness_check(model, texts: list[str], labels: list[str]) -> dict[str, float]:
    perturbed = [text.replace(" ", "  ").replace("e", "3") for text in texts]
    original = model.predict(texts)
    changed = model.predict(perturbed)
    return {
        "original_accuracy": float(accuracy_score(labels, original)),
        "perturbed_accuracy": float(accuracy_score(labels, changed)),
        "prediction_change_rate": float(sum(a != b for a, b in zip(original, changed)) / max(len(labels), 1)),
    }


def save_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
