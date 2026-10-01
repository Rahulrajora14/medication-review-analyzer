"""
Step 7 - Error analysis: WHERE and WHY does the model get it wrong?

Uses the DistilBERT predictions if they exist, otherwise the baseline's.
Writes a readable report to reports/error_analysis.md.
"""
import re

import pandas as pd

from src import config

CONTRAST = re.compile(r"\b(?:but|however|although|though|yet|except)\b", re.IGNORECASE)


def _pick_predictions() -> tuple[str, pd.DataFrame]:
    for key, name in [("transformer", "DistilBERT (fine-tuned)"),
                      ("baseline", "TF-IDF + Logistic Regression")]:
        path = config.PREDICTIONS_DIR / f"{key}_test_predictions.csv"
        if path.exists():
            return name, pd.read_csv(path)
    raise FileNotFoundError("No predictions found - train a model first.")


def _short(text: str, n: int = 260) -> str:
    text = str(text).replace("|", "/")
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + " ..."


def run_error_analysis() -> dict:
    model_name, df = _pick_predictions()
    df["correct"] = df["label"] == df["pred"]
    df["n_words"] = df["review"].str.split().str.len()
    df["has_contrast"] = df["review"].str.contains(CONTRAST)
    errors = df[~df["correct"]]
    L = config.LABELS

    # 1. Which mistakes are most common?
    pairs = (errors.groupby(["label", "pred"]).size().sort_values(ascending=False)
             .rename("count").reset_index())
    pairs["mistake"] = [f"true {L[a]} -> predicted {L[b]}" for a, b in zip(pairs["label"], pairs["pred"])]
    pairs["share_of_errors"] = (100 * pairs["count"] / len(errors)).round(1)

    # 2. Are errors concentrated on "borderline" ratings?
    borderline = errors["rating"].between(4, 7).mean()
    borderline_all = df["rating"].between(4, 7).mean()

    # 3. Reviews with contrast words ("worked great BUT ...")
    acc_contrast = df.loc[df["has_contrast"], "correct"].mean()
    acc_plain = df.loc[~df["has_contrast"], "correct"].mean()

    # 4. Accuracy by review length
    df["length_bucket"] = pd.cut(df["n_words"], [0, 25, 75, 150, 10_000],
                                 labels=["1-25", "26-75", "76-150", "150+"])
    by_len = df.groupby("length_bucket", observed=True)["correct"].mean().mul(100).round(1)

    # 5. Confident mistakes (the most interesting ones to read)
    errors = errors.copy()
    errors["confidence"] = errors[[f"p_{l}" for l in L]].max(axis=1)
    confident = errors.sort_values("confidence", ascending=False).head(6)
    examples_neutral = errors[errors["label"] == 1].sample(
        min(4, (errors["label"] == 1).sum()), random_state=config.RANDOM_STATE)

    lines = [
        f"# 🔬 Error analysis: {model_name}",
        "",
        f"Test reviews: **{len(df):,}** | errors: **{len(errors):,}** "
        f"({100 * len(errors) / len(df):.1f}%)",
        "",
        "## 1. Most common mistakes",
        "",
        "| Mistake | Count | Share of errors |",
        "|---|---:|---:|",
        *[f"| {r.mistake} | {r.count:,} | {r.share_of_errors}% |" for r in pairs.itertuples()],
        "",
        "## 2. Borderline ratings",
        "",
        f"**{100 * borderline:.0f}%** of errors come from ratings 4–7, although only "
        f"{100 * borderline_all:.0f}% of test reviews have those ratings. "
        "A 6/10 and a 7/10 often read almost the same, so part of this error is in the labels themselves.",
        "",
        "## 3. Mixed reviews (\"worked great, BUT ...\")",
        "",
        f"Accuracy on reviews with contrast words (but / however / although ...): "
        f"**{100 * acc_contrast:.1f}%** vs **{100 * acc_plain:.1f}%** on reviews without them.",
        "",
        "## 4. Accuracy by review length (words)",
        "",
        "| Length | Accuracy |",
        "|---|---:|",
        *[f"| {k} | {v}% |" for k, v in by_len.items()],
        "",
        "## 5. Most confident mistakes",
        "",
        "| Rating | True | Predicted | Confidence | Review |",
        "|---:|---|---|---:|---|",
        *[f"| {int(r.rating)} | {L[r.label]} | {L[r.pred]} | {r.confidence:.0%} | {_short(r.review)} |"
          for r in confident.itertuples()],
        "",
        "## 6. Sample neutral reviews the model missed",
        "",
        "| Rating | Predicted | Review |",
        "|---:|---|---|",
        *[f"| {int(r.rating)} | {L[r.pred]} | {_short(r.review)} |" for r in examples_neutral.itertuples()],
        "",
    ]
    config.ERROR_ANALYSIS_FILE.write_text("\n".join(lines), encoding="utf-8")

    summary = {"model": model_name, "error_rate": round(len(errors) / len(df), 4),
               "borderline_error_share": round(float(borderline), 3),
               "acc_contrast": round(float(acc_contrast), 4), "acc_plain": round(float(acc_plain), 4)}
    print(f"  Model analysed: {model_name}")
    print(f"  Errors from ratings 4-7: {100 * borderline:.0f}% (these ratings are "
          f"{100 * borderline_all:.0f}% of the test set)")
    print(f"  Accuracy with contrast words {100 * acc_contrast:.1f}% vs without {100 * acc_plain:.1f}%")
    print(f"  Report -> {config.ERROR_ANALYSIS_FILE}")
    return summary


if __name__ == "__main__":
    run_error_analysis()
