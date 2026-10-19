"""Dependency-free Vercel serverless text-risk decision support."""

import json
import re


def _response(status, body):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def handler(request):
    if request.method != "POST":
        return _response(405, {"error": "Use POST."})
    try:
        payload = json.loads(request.body or "{}")
        text = payload.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return _response(400, {"error": "A non-empty text value is required."})
        lower = text.lower()
        phishing_terms = ("verify", "password", "account", "suspended", "login", "bank")
        spam_terms = ("free", "winner", "prize", "offer", "cash", "claim")
        phishing_score = sum(term in lower for term in phishing_terms)
        spam_score = sum(term in lower for term in spam_terms)
        if phishing_score > spam_score and phishing_score:
            prediction, score = "phishing", min(0.55 + phishing_score * 0.07, 0.9)
        elif spam_score:
            prediction, score = "spam", min(0.55 + spam_score * 0.07, 0.9)
        else:
            prediction, score = "legitimate", 0.5
        return _response(
            200,
            {
                "label": prediction,
                "score": score,
                "score_kind": "heuristic risk score (not a calibrated probability)",
                "warning": "Vercel lightweight decision support only; use the research ML pipeline for measured model inference.",
            },
        )
    except Exception:
        return _response(500, {"error": "Classifier unavailable in this deployment."})
