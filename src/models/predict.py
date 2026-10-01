"""
One predictor class used by the dashboard (and anyone else).

- If a fine-tuned DistilBERT exists in models/distilbert/ AND PyTorch is installed,
  it is used for the prediction.
- Otherwise the TF-IDF baseline is used.
- Word-level explanations always come from the baseline, because a linear
  model's weights can be read directly: each word's contribution is
  (its TF-IDF value) x (its weight for the predicted class).
"""
import json

import joblib
import numpy as np
import sklearn

from src import config


def _major_minor(v: str) -> str:
    return ".".join(v.split(".")[:2])


def baseline_version_problem() -> str | None:
    """A pickled scikit-learn model is only reliable on the version that saved it."""
    if not config.BASELINE_META.exists():
        return None
    saved = json.loads(config.BASELINE_META.read_text()).get("library_versions", {})
    saved_v = saved.get("scikit-learn")
    if saved_v and _major_minor(saved_v) != _major_minor(sklearn.__version__):
        return (f"Baseline model saved with scikit-learn {saved_v}, you have "
                f"{sklearn.__version__}. Run `python main.py --step baseline` to retrain it.")
    return None


class SentimentPredictor:
    def __init__(self, prefer: str = "auto"):
        self.baseline = joblib.load(config.BASELINE_PATH) if config.BASELINE_PATH.exists() else None
        self.transformer = self.tokenizer = None
        self.name = "TF-IDF + Logistic Regression"

        if prefer != "baseline" and (config.TRANSFORMER_DIR / "config.json").exists():
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._torch = torch
                self.tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_DIR)
                self.transformer = AutoModelForSequenceClassification.from_pretrained(
                    config.TRANSFORMER_DIR).eval()
                self.name = "DistilBERT (fine-tuned)"
            except Exception:  # torch/transformers missing or broken -> stay on baseline
                self.transformer = self.tokenizer = None

        if self.baseline is None and self.transformer is None:
            raise FileNotFoundError("No trained model found. Run `python main.py` first.")

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self.transformer is not None:
            enc = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=config.MAX_LENGTH, return_tensors="pt")
            with self._torch.no_grad():
                logits = self.transformer(**enc).logits
            return self._torch.softmax(logits, dim=-1).numpy()
        return self.baseline.predict_proba(texts)

    def predict(self, text: str) -> dict:
        proba = self.predict_proba([text])[0]
        label = int(proba.argmax())
        return {"label": config.LABELS[label],
                "probabilities": {l: float(p) for l, p in zip(config.LABELS, proba)}}

    def explain(self, text: str, target_label: str, top_n: int = 8) -> dict[str, list]:
        """Words pushing the review towards / away from target_label (baseline weights)."""
        if self.baseline is None:
            return {"supporting": [], "opposing": []}
        tfidf = self.baseline.named_steps["tfidf"]
        coefs = self.baseline.named_steps["clf"].coef_
        c = config.LABELS.index(target_label)
        relative = coefs[c] - coefs.mean(axis=0)          # vs the average class
        x = tfidf.transform([text])
        idx = x.indices
        contrib = x.data * relative[idx]
        vocab = tfidf.get_feature_names_out()
        order = np.argsort(contrib)
        supporting = [(vocab[idx[i]], float(contrib[i])) for i in order[::-1][:top_n] if contrib[i] > 0]
        opposing = [(vocab[idx[i]], float(contrib[i])) for i in order[:top_n] if contrib[i] < 0]
        return {"supporting": supporting, "opposing": opposing}
