# Data flow

1. A user enters text locally.
2. The privacy analyzer optionally identifies supported entities and creates a reviewable redaction.
3. The crypto engine derives a key from the entered password, generates a fresh salt and nonce, and emits a versioned JSON payload.
4. The classifier receives text only when the user explicitly requests analysis.
5. Training scripts produce measured model artifacts and machine-readable reports under `artifacts/`.
