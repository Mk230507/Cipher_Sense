"""Model explanations that are explicitly presented as evidence, not proof."""

from __future__ import annotations


def classical_features(model, text: str, limit: int = 8) -> list[tuple[str, float]]:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    row = vectorizer.transform([text])
    if not hasattr(classifier, "coef_"):
        return []
    terms = vectorizer.get_feature_names_out()
    active = row.nonzero()[1]
    classes = list(classifier.classes_)
    label = str(model.predict([text])[0])
    index = classes.index(label)
    weights = classifier.coef_[0] if len(classes) == 2 else classifier.coef_[index]
    return sorted(
        ((terms[i], float(weights[i] * row[0, i])) for i in active),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:limit]
