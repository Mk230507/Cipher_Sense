function classify(text) {
  const lower = text.toLowerCase();
  const phishingTerms = ["verify", "password", "account", "suspended", "login", "bank"];
  const spamTerms = ["free", "winner", "prize", "offer", "cash", "claim"];
  const phishingScore = phishingTerms.filter((term) => lower.includes(term)).length;
  const spamScore = spamTerms.filter((term) => lower.includes(term)).length;

  if (phishingScore > spamScore && phishingScore) {
    return {
      label: "phishing",
      score: Math.min(0.55 + phishingScore * 0.07, 0.9),
    };
  }
  if (spamScore) {
    return {
      label: "spam",
      score: Math.min(0.55 + spamScore * 0.07, 0.9),
    };
  }
  return { label: "legitimate", score: 0.5 };
}

module.exports = function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Use POST." });
  }

  const text = typeof req.body?.text === "string" ? req.body.text : "";
  if (!text.trim()) {
    return res.status(400).json({ error: "A non-empty text value is required." });
  }

  const result = classify(text);
  return res.status(200).json({
    ...result,
    score_kind: "heuristic risk score (not a calibrated probability)",
    warning:
      "Vercel lightweight decision support only; use the research ML pipeline for measured model inference.",
  });
};
