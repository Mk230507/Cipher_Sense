"""Train and evaluate the reproducible local demonstration classifiers."""

from pathlib import Path

from sklearn.model_selection import train_test_split

from ciphersense.classifier import load_dataset, save_model, train_models
from ciphersense.dataset import provenance
from ciphersense.evaluation import evaluate, robustness_check, save_confusion_plot, save_json


def main() -> None:
    texts, labels = load_dataset()
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    models = train_models(train_texts, train_labels)
    output = Path("artifacts")
    metrics = {
        name: evaluate(model, test_texts, test_labels, name)
        for name, model in models.items()
    }
    for name, model in models.items():
        save_model(model, output / f"{name}.joblib")
        save_confusion_plot(metrics[name], output / f"{name}_confusion.png")
    save_json(
        {"dataset": "bundled transparent smoke-test corpus", "metrics": metrics,
         "robustness": {name: robustness_check(model, test_texts, test_labels)
                        for name, model in models.items()}},
        output / "classical_metrics.json",
    )
    save_json(provenance(), output / "dataset_provenance.json")
    print("Saved models and held-out metrics to artifacts/.")
    for name, result in metrics.items():
        print(f"{name}: macro-F1={result['macro_f1']:.3f}, latency={result['latency_ms_per_message']:.3f}ms")


if __name__ == "__main__":
    main()
