"""Tkinter interface for secure messaging and local threat analysis."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .classifier import load_dataset, predict, train_models
from .crypto import CryptoError, decrypt_message, encrypt_message
from .privacy import PrivacyAnalyzer


class CipherSenseApp:
    def __init__(self, window: tk.Tk) -> None:
        self.window = window
        self.window.title("CipherSense AI - Secure Messaging")
        self.window.geometry("920x700")
        self.window.minsize(760, 580)
        self.analyzer = PrivacyAnalyzer()
        texts, labels = load_dataset()
        self.models = train_models(texts, labels)
        self._build_ui()

    def _build_ui(self) -> None:
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=12)
        self._build_secure_tab()
        self._build_privacy_tab()
        self._build_detection_tab()

    def _build_secure_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="Secure messaging")
        ttk.Label(
            tab,
            text="AES-256-GCM encryption (not Base64 encoding)",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")
        ttk.Label(tab, text="Messages stay local. Passwords are never stored.").pack(anchor="w", pady=(2, 10))
        self.secure_text = tk.Text(tab, height=14, wrap="word")
        self.secure_text.pack(fill="both", expand=True)
        row = ttk.Frame(tab)
        row.pack(fill="x", pady=8)
        ttk.Label(row, text="Password:").pack(side="left")
        self.password = ttk.Entry(row, show="*", width=34)
        self.password.pack(side="left", padx=8)
        ttk.Button(row, text="Encrypt", command=self.encrypt).pack(side="left", padx=3)
        ttk.Button(row, text="Decrypt", command=self.decrypt).pack(side="left", padx=3)
        ttk.Button(row, text="Copy result", command=self.copy_result).pack(side="left", padx=3)
        ttk.Button(row, text="Import", command=self.import_payload).pack(side="left", padx=3)
        ttk.Button(row, text="Export", command=self.export_payload).pack(side="left", padx=3)
        self.secure_status = ttk.Label(tab, text="Ready.")
        self.secure_status.pack(anchor="w")

    def _build_privacy_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="Privacy analyzer")
        ttk.Label(tab, text="Local NLP privacy review", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(tab, text=self.analyzer.model_status).pack(anchor="w", pady=(2, 8))
        self.privacy_text = tk.Text(tab, height=14, wrap="word")
        self.privacy_text.pack(fill="both", expand=True)
        row = ttk.Frame(tab)
        row.pack(fill="x", pady=8)
        ttk.Button(row, text="Analyze", command=self.analyze_privacy).pack(side="left", padx=3)
        ttk.Button(row, text="Redact and use for encryption", command=self.redact_privacy).pack(side="left", padx=3)
        self.privacy_status = ttk.Label(tab, text="Supported regex categories: email, phone, URL, IP address; spaCy: person, organization, location.")
        self.privacy_status.pack(anchor="w")

    def _build_detection_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(tab, text="Threat detection")
        ttk.Label(tab, text="Classical text-risk decision support", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(tab, text="The demo corpus is transparent and small; train on a licensed public dataset for research results.").pack(anchor="w", pady=(2, 8))
        self.detection_text = tk.Text(tab, height=14, wrap="word")
        self.detection_text.pack(fill="both", expand=True)
        ttk.Button(tab, text="Analyze message", command=self.detect).pack(anchor="w", pady=8)
        self.detection_status = ttk.Label(tab, text="Scores are not proof of malicious intent.")
        self.detection_status.pack(anchor="w")

    def _get_secure_text(self) -> str:
        return self.secure_text.get("1.0", "end-1c")

    def encrypt(self) -> None:
        try:
            result = encrypt_message(self._get_secure_text(), self.password.get())
            self.secure_text.delete("1.0", "end")
            self.secure_text.insert("1.0", result)
            self.secure_status.config(text="Encrypted with a fresh salt and nonce.")
        except CryptoError as exc:
            self._error(str(exc))

    def decrypt(self) -> None:
        try:
            result = decrypt_message(self._get_secure_text(), self.password.get())
            self.secure_text.delete("1.0", "end")
            self.secure_text.insert("1.0", result)
            self.secure_status.config(text="Authenticated and decrypted successfully.")
        except CryptoError:
            self._error("Unable to decrypt: the password may be wrong or the payload may be damaged.")

    def copy_result(self) -> None:
        self.window.clipboard_clear()
        self.window.clipboard_append(self._get_secure_text())
        self.secure_status.config(text="Result copied to the clipboard.")

    def import_payload(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("JSON payload", "*.json"), ("All files", "*.*")])
        if path:
            self.secure_text.delete("1.0", "end")
            self.secure_text.insert("1.0", Path(path).read_text(encoding="utf-8"))

    def export_payload(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON payload", "*.json")])
        if path:
            Path(path).write_text(self._get_secure_text(), encoding="utf-8")
            self.secure_status.config(text="Encrypted payload exported.")

    def analyze_privacy(self) -> None:
        entities = self.analyzer.analyze(self.privacy_text.get("1.0", "end-1c"))
        if not entities:
            self.privacy_status.config(text="No supported sensitive entities were detected.")
            return
        details = ", ".join(f"{item.category}: {item.text!r} [{item.start}:{item.end}]" for item in entities)
        self.privacy_status.config(text=f"Detected {len(entities)} item(s): {details}")

    def redact_privacy(self) -> None:
        text = self.privacy_text.get("1.0", "end-1c")
        redacted = self.analyzer.redact(text, self.analyzer.analyze(text))
        self.secure_text.delete("1.0", "end")
        self.secure_text.insert("1.0", redacted)
        self.notebook.select(0)

    def detect(self) -> None:
        text = self.detection_text.get("1.0", "end-1c")
        result = predict(self.models["logistic_regression"], text)
        self.detection_status.config(
            text=f"Prediction: {result.label} | {result.score_kind}: {result.score:.3f} | {result.warning}"
        )

    def _error(self, message: str) -> None:
        messagebox.showerror("CipherSense", message)


def main() -> None:
    window = tk.Tk()
    CipherSenseApp(window)
    window.mainloop()


if __name__ == "__main__":
    main()
