"""Versioned password-based AES-GCM encryption."""

from __future__ import annotations

import base64
import binascii
import json
import secrets
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

PAYLOAD_VERSION = 1
SALT_BYTES = 16
NONCE_BYTES = 12
KEY_BYTES = 32


class CryptoError(ValueError):
    """Raised when an encrypted payload cannot be safely processed."""


@dataclass(frozen=True)
class EncryptedPayload:
    """Validated, serializable encryption envelope."""

    version: int
    salt: bytes
    nonce: bytes
    ciphertext: bytes

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": self.version,
                "algorithm": "AES-256-GCM",
                "kdf": "scrypt",
                "salt": _encode(self.salt),
                "nonce": _encode(self.nonce),
                "ciphertext": _encode(self.ciphertext),
            },
            separators=(",", ":"),
        )


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii")


def _decode(value: object, field: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise CryptoError(f"Invalid encrypted payload field: {field}.")
    try:
        return base64.b64decode(value.encode("ascii"), altchars=b"-_", validate=True)
    except (ValueError, binascii.Error, UnicodeEncodeError) as exc:
        raise CryptoError(f"Invalid encrypted payload field: {field}.") from exc


def _derive_key(password: str, salt: bytes) -> bytes:
    if not password:
        raise CryptoError("A non-empty password is required.")
    return Scrypt(
        salt=salt,
        length=KEY_BYTES,
        n=2**14,
        r=8,
        p=1,
    ).derive(password.encode("utf-8"))


def encrypt_message(message: str, password: str) -> str:
    """Encrypt UTF-8 text and return a versioned JSON envelope."""
    salt = secrets.token_bytes(SALT_BYTES)
    nonce = secrets.token_bytes(NONCE_BYTES)
    key = _derive_key(password, salt)
    ciphertext = AESGCM(key).encrypt(nonce, message.encode("utf-8"), None)
    return EncryptedPayload(PAYLOAD_VERSION, salt, nonce, ciphertext).to_json()


def parse_payload(payload: str) -> EncryptedPayload:
    """Parse and validate the public envelope without decrypting it."""
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise CryptoError("The encrypted payload is not valid JSON.") from exc
    if not isinstance(data, dict):
        raise CryptoError("The encrypted payload must be a JSON object.")
    if data.get("version") != PAYLOAD_VERSION:
        raise CryptoError("This encrypted payload version is unsupported.")
    if data.get("algorithm") != "AES-256-GCM" or data.get("kdf") != "scrypt":
        raise CryptoError("This encrypted payload uses an unsupported format.")
    salt = _decode(data.get("salt"), "salt")
    nonce = _decode(data.get("nonce"), "nonce")
    ciphertext = _decode(data.get("ciphertext"), "ciphertext")
    if len(salt) != SALT_BYTES or len(nonce) != NONCE_BYTES or len(ciphertext) < 16:
        raise CryptoError("The encrypted payload has invalid binary field lengths.")
    return EncryptedPayload(PAYLOAD_VERSION, salt, nonce, ciphertext)


def decrypt_message(payload: str, password: str) -> str:
    """Authenticate and decrypt a JSON envelope."""
    envelope = parse_payload(payload)
    try:
        plaintext = AESGCM(_derive_key(password, envelope.salt)).decrypt(
            envelope.nonce, envelope.ciphertext, None
        )
        return plaintext.decode("utf-8")
    except (InvalidTag, UnicodeDecodeError, CryptoError) as exc:
        raise CryptoError("The password is incorrect or the payload was modified.") from exc
