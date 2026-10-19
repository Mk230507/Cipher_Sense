"""The original repeating-key transformation, retained for comparison only."""

import base64


def legacy_encode(key: str, message: str) -> str:
    if not key:
        raise ValueError("A key is required.")
    values = [
        (ord(character) + ord(key[index % len(key)])) % 256
        for index, character in enumerate(message)
    ]
    return base64.urlsafe_b64encode(bytes(values)).decode("ascii")


def legacy_decode(key: str, message: str) -> str:
    if not key:
        raise ValueError("A key is required.")
    try:
        values = base64.urlsafe_b64decode(message.encode("ascii"))
    except Exception as exc:
        raise ValueError("Invalid legacy payload.") from exc
    return "".join(
        chr((value - ord(key[index % len(key)])) % 256)
        for index, value in enumerate(values)
    )
