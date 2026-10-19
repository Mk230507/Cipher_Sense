"""Dataset acquisition and provenance helpers."""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

from .classifier import LABELS

SMS_SPAM_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SMS_SPAM_LICENSE = "UCI Machine Learning Repository terms; binary ham/spam labels."


def download_sms_spam(destination: Path, timeout: int = 30) -> Path:
    """Download the UCI SMS Spam Collection without silently substituting data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(SMS_SPAM_URL, timeout=timeout) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        member = source.namelist()[0]
        destination.write_bytes(source.read(member))
    return destination


def load_tsv(path: Path) -> tuple[list[str], list[str]]:
    """Load a two-column label/text corpus and normalize only known labels."""
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) >= 2:
                label = {"ham": "legitimate", "spam": "spam"}.get(row[0].strip().lower())
                if label in LABELS:
                    rows.append((row[1].strip(), label))
    unique = list(dict.fromkeys(rows))
    if not unique:
        raise ValueError("The dataset contains no valid label/text rows.")
    return [text for text, _ in unique], [label for _, label in unique]


def provenance() -> dict[str, str]:
    return {
        "name": "UCI SMS Spam Collection",
        "url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "download_url": SMS_SPAM_URL,
        "license_note": SMS_SPAM_LICENSE,
        "label_mapping": "ham -> legitimate; spam -> spam; no phishing labels are inferred",
    }
