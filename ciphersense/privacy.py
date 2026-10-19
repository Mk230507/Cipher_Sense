"""Local NLP privacy analysis with spaCy NER and conservative regex rules."""

from __future__ import annotations

import re
from dataclasses import dataclass

try:
    import spacy
except (ImportError, TypeError):
    spacy = None

PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "PHONE": re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)"),
    "URL": re.compile(r"\bhttps?://[^\s<>\"]+", re.I),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}
SUPPORTED_NER_LABELS = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
}


@dataclass(frozen=True)
class SensitiveEntity:
    text: str
    category: str
    start: int
    end: int
    source: str


class PrivacyAnalyzer:
    """Analyze text locally; no message content is sent to an external service."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        if spacy is None:
            self.nlp = None
            self.model_status = (
                "spaCy could not be loaded; regex detection is active and NER is disabled."
            )
            return
        try:
            self.nlp = spacy.load(model_name)
            self.model_status = f"spaCy model loaded: {model_name}"
        except OSError:
            self.nlp = spacy.blank("en")
            self.model_status = (
                "spaCy model unavailable; regex detection is active and NER is disabled."
            )

    def analyze(self, text: str) -> list[SensitiveEntity]:
        entities = [
            SensitiveEntity(match.group(), category, match.start(), match.end(), "regex")
            for category, pattern in PATTERNS.items()
            for match in pattern.finditer(text)
        ]
        if self.nlp is None:
            return sorted(
                {(item.start, item.end, item.category): item for item in entities}.values(),
                key=lambda item: (item.start, item.end),
            )
        for entity in self.nlp(text).ents:
            category = SUPPORTED_NER_LABELS.get(entity.label_)
            if category:
                entities.append(
                    SensitiveEntity(
                        entity.text, category, entity.start_char, entity.end_char, "spacy"
                    )
                )
        unique = {(item.start, item.end, item.category): item for item in entities}
        return sorted(unique.values(), key=lambda item: (item.start, item.end))

    @staticmethod
    def redact(text: str, entities: list[SensitiveEntity]) -> str:
        for entity in sorted(entities, key=lambda item: item.start, reverse=True):
            text = text[: entity.start] + f"[REDACTED {entity.category}]" + text[entity.end :]
        return text
