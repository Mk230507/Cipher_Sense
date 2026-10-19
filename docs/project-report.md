# CipherSense AI technical report

## Contribution

CipherSense combines authenticated local messaging with privacy-aware NLP and text-risk classification. The secure path uses AES-256-GCM and Scrypt rather than the retained legacy repeating-key demonstration.

## Experimental protocol

`train_classical.py` performs a stratified held-out split with a fixed seed and fits TF-IDF only inside each training pipeline. It records macro-F1, weighted-F1, precision, recall, class reports, confusion matrices, and latency when `ciphersense.evaluation` is used. The included 12-message corpus is a reproducible smoke test, not evidence of production accuracy.

## Limitations and future work

The NER model and regex rules have false positives and false negatives. The demo corpus is too small for general claims. A research submission should replace it with a licensed, documented corpus and report confidence intervals. Transformer fine-tuning is optional because its resource cost is materially larger.

## References

* NIST SP 800-38D, Recommendation for Block Cipher Modes of Operation: GCM and GMAC.
* UCI Machine Learning Repository, SMS Spam Collection, https://archive.ics.uci.edu/dataset/228/sms+spam+collection.
* spaCy documentation, https://spacy.io/usage.
* Hugging Face Transformers documentation, https://huggingface.co/docs/transformers.
