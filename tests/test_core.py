import json
import unittest

from ciphersense.classifier import load_dataset, predict, train_models
from ciphersense.crypto import CryptoError, decrypt_message, encrypt_message, parse_payload
from ciphersense.evaluation import evaluate, robustness_check
from ciphersense.explainability import classical_features
from ciphersense.transformer import TransformerClassifier, TransformerUnavailable
from ciphersense.privacy import PrivacyAnalyzer


class CipherSenseTests(unittest.TestCase):
    def test_crypto_round_trip_supports_unicode_and_empty_text(self):
        for text in ("", "Hello, мир 🌍\nsecond line"):
            payload = encrypt_message(text, "correct horse battery staple")
            self.assertEqual(
                decrypt_message(payload, "correct horse battery staple"), text
            )

    def test_crypto_rejects_tampering_and_wrong_password(self):
        payload = json.loads(encrypt_message("secret", "pass"))
        payload["ciphertext"] = payload["ciphertext"][:-2] + "aa"
        with self.assertRaises(CryptoError):
            decrypt_message(json.dumps(payload), "pass")
        with self.assertRaises(CryptoError):
            decrypt_message(encrypt_message("secret", "pass"), "wrong")
        with self.assertRaises(CryptoError):
            parse_payload("{not-json}")

    def test_payload_has_random_nonce_and_salt(self):
        first = parse_payload(encrypt_message("same", "same password"))
        second = parse_payload(encrypt_message("same", "same password"))
        self.assertNotEqual(first.salt, second.salt)
        self.assertNotEqual(first.nonce, second.nonce)

    def test_privacy_analyzer_reports_spans_and_redacts(self):
        analyzer = PrivacyAnalyzer()
        text = "Contact Ada at ada@example.com or https://example.com."
        entities = analyzer.analyze(text)
        self.assertTrue(any(entity.category == "EMAIL" for entity in entities))
        redacted = analyzer.redact(text, entities)
        self.assertNotIn("ada@example.com", redacted)

    def test_classical_models_produce_decision_support_prediction(self):
        texts, labels = load_dataset()
        models = train_models(texts, labels)
        result = predict(models["logistic_regression"], "Urgent verify your account at a link")
        self.assertIn(result.label, {"legitimate", "spam", "phishing"})
        self.assertTrue(result.warning)

    def test_evaluation_and_explanation_are_measured(self):
        texts, labels = load_dataset()
        models = train_models(texts, labels)
        metrics = evaluate(models["logistic_regression"], texts, labels, "logistic")
        self.assertIn("confusion_matrix", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("perturbed_accuracy", robustness_check(models["logistic_regression"], texts, labels))
        self.assertIsInstance(classical_features(models["logistic_regression"], texts[0]), list)

    def test_missing_transformer_artifact_is_explicit(self):
        with self.assertRaises(TransformerUnavailable):
            TransformerClassifier("artifacts/does-not-exist").load()
