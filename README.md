# CipherSense AI

CipherSense AI is a local, modular demonstration of secure messaging, NLP-based privacy review, and text-risk classification. It is designed for reproducible university demonstration and portfolio use; it does not claim that a small demo corpus is a production security detector.

## Run

```powershell
python -m pip install -r requirements.txt
python encode-decode.py
```

The application has three tabs:

* **Secure messaging**: AES-256-GCM authenticated encryption with a fresh random salt and nonce for every message. Passwords are converted into keys with Scrypt and are never stored.
* **Privacy analyzer**: local spaCy NER plus regular expressions for email addresses, phone numbers, URLs, and IP addresses. It shows character spans and can redact before encryption.
* **Threat detection**: TF-IDF plus Logistic Regression, with Linear SVM and majority-class baseline available in the training module. Scores are decision-support signals, not proof or calibrated probabilities.

The Streamlit interface is also available:

```powershell
streamlit run streamlit_app.py
```

The original repeating-key algorithm is retained only in `ciphersense/legacy.py` for comparison. It is not used for secure messaging.

## Reproducible classical experiment

```powershell
python train_classical.py
python -m unittest discover -s tests -v
```

This creates `artifacts/` with models and JSON metrics generated from execution. The bundled 12-message corpus is a smoke-test/demo corpus, not a research benchmark. For a publishable experiment, use a public corpus with documented provenance and licensing, map its labels explicitly, deduplicate it, and keep a held-out test set.

`ciphersense.dataset` provides an explicit downloader for the UCI SMS Spam Collection. It maps only `ham` to `legitimate` and `spam` to `spam`; it does not manufacture phishing labels. Dataset acquisition is therefore a deliberate, documented step rather than an unverified claim.

## Optional transformer workflow

`ciphersense/transformer.py` loads a locally saved Hugging Face sequence-classification artifact and performs CPU-compatible inference. Fine-tuning is intentionally separate because downloading DistilBERT and training it can require substantial disk space, memory, and time. No transformer accuracy or benchmark is reported unless training is actually run and its metrics are saved.

To run the optional fine-tuning workflow:

```powershell
python train_transformer.py --output-dir artifacts/distilbert --epochs 2
```

It records the actual device, elapsed training time, seed, and held-out accuracy. The bundled corpus is deliberately too small for a meaningful research result; use a larger, licensed corpus before drawing conclusions.

## Deployment and academic package

`Dockerfile` and `docker-compose.yml` provide a CPU-compatible Streamlit deployment. `docs/architecture.md`, `docs/data-flow.md`, and `docs/project-report.md` contain the architecture, data flow, methodology, references, and limitations. Docker image execution was not run in this environment unless Docker is available.

In this environment the Docker CLI was installed, but the Docker daemon was not running, so image construction was attempted and reported as blocked by the unavailable daemon. Native execution and the Streamlit health endpoint were verified successfully.

## Vercel-only deployment

The `public/` and `api/` directories provide a Vercel-native deployment that does not need an always-on server:

```powershell
npx vercel
```

The browser performs AES-GCM encryption/decryption and privacy regex analysis locally. Passwords and plaintext are not sent to Vercel. `/api/health` is a serverless health endpoint, and `/api/classify` is a dependency-free heuristic triage endpoint so the deployment does not need Streamlit, Docker, PyTorch, spaCy, or a persistent backend. The full scikit-learn and transformer experiments remain available locally through `train_classical.py` and `train_transformer.py`; they are not falsely presented as running inside the lightweight Vercel deployment.

## Payload format

Encrypted payloads are compact JSON objects:

```json
{"version":1,"algorithm":"AES-256-GCM","kdf":"scrypt","salt":"...","nonce":"...","ciphertext":"..."}
```

Binary fields use URL-safe Base64. Any changed ciphertext, wrong password, malformed field, or unsupported version is rejected. Encoding is not encryption: Base64 only represents bytes, while AES-GCM provides confidentiality and integrity.

## NLP limitations

The privacy analyzer runs locally and supports the categories documented in the UI. General-purpose NER can miss entities, produce false positives, and is not a complete identity or privacy detector. Review redactions before encrypting. Plaintext is not transmitted to an external API.
