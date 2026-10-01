"""
Step 6 - Compare all trained models and update the README automatically.

Reads every reports/metrics/*.json, writes reports/model_comparison.csv,
draws a comparison chart, and rewrites the results table in README.md
between the RESULTS markers. So after you fine-tune DistilBERT on a GPU,
`python main.py --step report` puts the real numbers in your README.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config

START, END = "<!-- RESULTS_TABLE_START -->", "<!-- RESULTS_TABLE_END -->"
ORDER = ["baseline", "transformer"]


def load_metrics() -> pd.DataFrame:
    rows = []
    for key in ORDER:
        path = config.METRICS_DIR / f"{key}.json"
        if path.exists():
            m = json.loads(path.read_text())
            rows.append({k: v for k, v in m.items() if k != "history"})
    if not rows:
        raise FileNotFoundError("No metrics found - train a model first.")
    return pd.DataFrame(rows)


def plot_comparison(df: pd.DataFrame) -> None:
    cols = ["macro_f1", "f1_negative", "f1_neutral", "f1_positive"]
    labels = ["Macro-F1", "F1 negative", "F1 neutral", "F1 positive"]
    x = np.arange(len(cols))
    width = 0.8 / len(df)
    colors = ["#7f8c8d", "#2a7f8f"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, (_, row) in enumerate(df.iterrows()):
        bars = ax.bar(x + i * width - 0.4 + width / 2, row[cols].astype(float), width,
                      label=row["model"], color=colors[i % 2])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_title("Test-set F1 by class", loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "model_comparison.png", dpi=130)
    plt.close(fig)


def results_table(df: pd.DataFrame) -> str:
    lines = ["| Model | Accuracy | Macro-F1 | F1 negative | F1 neutral | F1 positive |",
             "|---|:---:|:---:|:---:|:---:|:---:|"]
    best = df["macro_f1"].idxmax()
    for i, r in df.iterrows():
        name = f"**{r['model']}** ⭐" if i == best and len(df) > 1 else r["model"]
        lines.append(f"| {name} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | "
                     f"{r['f1_negative']:.3f} | {r['f1_neutral']:.3f} | {r['f1_positive']:.3f} |")
    if "transformer" not in [k for k in ORDER if (config.METRICS_DIR / f"{k}.json").exists()]:
        lines.append("| DistilBERT (fine-tuned) | *run on GPU* | *see below* | | | |")
    return "\n".join(lines)


def update_readme(table: str) -> bool:
    if not config.README_FILE.exists():
        return False
    text = config.README_FILE.read_text(encoding="utf-8")
    if START not in text or END not in text:
        return False
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    config.README_FILE.write_text(f"{before}{START}\n{table}\n{END}{after}", encoding="utf-8")
    return True


def run_report() -> pd.DataFrame:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_metrics()
    df.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    plot_comparison(df)
    table = results_table(df)
    print(table)
    print("  README results table updated." if update_readme(table)
          else "  (README has no RESULTS markers - table printed above.)")
    return df


if __name__ == "__main__":
    run_report()
