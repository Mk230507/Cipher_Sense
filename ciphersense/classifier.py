"""Reproducible classical text classification and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

LABELS = ("legitimate", "spam", "phishing")

# A small transparent smoke-test corpus. Replace with a licensed public corpus
# through load_dataset() for research evaluation; this is not a benchmark claim.
DEMO_DATASET = [
    ("Team meeting moved to 3 pm in room 4.", "legitimate"),
    ("Can you send me the project report today?", "legitimate"),
    ("Your appointment is confirmed for tomorrow.", "legitimate"),
    ("Lunch is ready when you are.", "legitimate"),
    ("You won a free vacation! Claim your prize now.", "spam"),
    ("Congratulations, receive your cash reward by replying YES.", "spam"),
    ("Exclusive offer: buy one and get five free.", "spam"),
    ("You have been selected for a gift card.", "spam"),
    ("Urgent: verify your bank account at http://secure-login.example.", "phishing"),
    ("Your password expires today. Sign in to avoid suspension.", "phishing"),
    ("Confirm your payment details using this link immediately.", "phishing"),
    ("We detected unusual activity. Validate your account now.", "phishing"),
]


@dataclass
class Prediction:
    label: str
    score: float
    score_kind: str
    features: list[tuple[str, float]]
    warning: str = "Decision support only: this classifier can make mistakes."


def load_dataset(rows: Iterable[tuple[str, str]] = DEMO_DATASET) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    for text, label in rows:
        normalized = str(label).strip().lower()
        if str(text).strip() and normalized in LABELS:
            texts.append(str(text).strip())
            labels.append(normalized)
    if len(set(labels)) < 2:
        raise ValueError("The dataset must contain at least two valid classes.")
    return texts, labels


def build_models() -> dict[str, Pipeline]:
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), sublinear_tf=True, min_df=1
    )
    return {
        "logistic_regression": Pipeline(
            [("tfidf", vectorizer), ("classifier", LogisticRegression(max_iter=1000, random_state=42))]
        ),
        "linear_svm": Pipeline(
            [("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)), ("classifier", LinearSVC())]
        ),
        "majority_baseline": Pipeline(
            [("tfidf", TfidfVectorizer()), ("classifier", DummyClassifier(strategy="most_frequent"))]
        ),
    }


def train_models(texts: list[str], labels: list[str]) -> dict[str, Pipeline]:
    models = build_models()
    for model in models.values():
        model.fit(texts, labels)
    return models


def evaluate_model(model: Pipeline, texts: list[str], labels: list[str]) -> dict[str, object]:
    predictions = model.predict(texts)
    return {
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "report": classification_report(labels, predictions, output_dict=True, zero_division=0),
    }


def predict(model: Pipeline, text: str) -> Prediction:
    label = str(model.predict([text])[0])
    classifier = model.named_steps["classifier"]
    features: list[tuple[str, float]] = []
    vectorizer = model.named_steps["tfidf"]
    row = model.named_steps["tfidf"].transform([text])
    if hasattr(classifier, "decision_function"):
        raw = classifier.decision_function(row)
        classes = list(classifier.classes_)
        if len(classes) == 2:
            raw_value = float(raw.ravel()[0])
            score = raw_value if label == classes[1] else -raw_value
        else:
            score = float(raw.ravel()[classes.index(label)])
        score_kind = "SVM decision score (not a probability)"
        if hasattr(classifier, "coef_"):
            class_index = classes.index(label)
            weights = classifier.coef_[0] if len(classes) == 2 else classifier.coef_[class_index]
            ranked = sorted(zip(vectorizer.get_feature_names_out(), weights), key=lambda item: abs(item[1]), reverse=True)
            features = [(term, float(weight)) for term, weight in ranked if term in text.lower()][:5]
    else:
        score = float(max(classifier.predict_proba(row)[0]))
        score_kind = "model probability estimate"
    return Prediction(label, score, score_kind, features)


def save_model(model: Pipeline, path: Path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_metrics(metrics: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
