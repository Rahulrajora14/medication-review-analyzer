"""
Shared evaluation code used by BOTH models, so they are judged identically.

Why macro-F1 is the headline metric:
  66% of reviews are positive. A model that always says "positive" gets
  66% accuracy but is useless. Macro-F1 computes F1 for each class
  separately and averages them, so the small neutral class (9%) counts
  as much as the big positive class.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src import config


def compute_metrics(y_true, y_pred) -> dict:
    report = classification_report(y_true, y_pred, labels=[0, 1, 2],
                                   target_names=config.LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted")), 4),
        **{f"f1_{label}": round(float(report[label]["f1-score"]), 4) for label in config.LABELS},
    }


def save_results(model_key: str, display_name: str, test_df: pd.DataFrame,
                 y_pred: np.ndarray, proba: np.ndarray, extra: dict | None = None) -> dict:
    """Save metrics JSON, a predictions CSV and a confusion-matrix chart for one model."""
    for d in (config.METRICS_DIR, config.PREDICTIONS_DIR, config.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    metrics = {"model": display_name, **compute_metrics(test_df["label"], y_pred), **(extra or {})}
    (config.METRICS_DIR / f"{model_key}.json").write_text(json.dumps(metrics, indent=2))

    preds = test_df[["review_id", "drug", "condition", "rating", "label", "review"]].copy()
    preds["pred"] = y_pred
    for i, label in enumerate(config.LABELS):
        preds[f"p_{label}"] = proba[:, i].round(4)
    preds.to_csv(config.PREDICTIONS_DIR / f"{model_key}_test_predictions.csv", index=False)

    plot_confusion_matrix(test_df["label"], y_pred, display_name, f"confusion_matrix_{model_key}.png")
    return metrics


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    cm_pct = cm / cm.sum(axis=1, keepdims=True)   # row % = recall per true class
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_pct[i, j]:.0%}\n({cm[i, j]:,})", ha="center", va="center",
                    color="white" if cm_pct[i, j] > 0.5 else "black", fontsize=10)
    ax.set_xticks(range(3), [f"pred {l}" for l in config.LABELS])
    ax.set_yticks(range(3), [f"actual {l}" for l in config.LABELS])
    ax.set_title(f"{title}: confusion matrix (row %)", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / filename, dpi=130)
    plt.close(fig)
