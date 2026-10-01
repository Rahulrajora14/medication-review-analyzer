"""
Streamlit dashboard - Medication Review Analyzer.

Run from the project root:
    streamlit run app/streamlit_app.py

Tabs:
  1. Analyze a review  - sentiment + the words behind it + side effects mentioned
  2. Drug insights     - what patients say about a specific drug
  3. Model performance - baseline vs DistilBERT, confusion matrices, error analysis
  4. About
"""
import html
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.analysis.side_effects import detect_side_effects  # noqa: E402
from src.models.predict import SentimentPredictor, baseline_version_problem  # noqa: E402

st.set_page_config(page_title="Medication Review Analyzer", page_icon="💊", layout="wide")

EMOJI = {"negative": "😟", "neutral": "😐", "positive": "😊"}
COLOR = {"negative": "#c0504d", "neutral": "#8a8a8a", "positive": "#1f6f7a"}
EXAMPLES = {
    "Positive": "I have been on this for six months and it has honestly changed my life. "
                "My anxiety is so much better and I have had no side effects at all.",
    "Mixed": "It does help with the pain, but the drowsiness is really hard to deal with "
             "and I have gained weight. Not sure if I will stay on it.",
    "Negative": "Worst experience ever. Constant nausea and terrible headaches for two weeks "
                "and it did nothing for my symptoms. I stopped taking it.",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_predictor():
    return SentimentPredictor()


@st.cache_data
def load_insights():
    drugs_path = config.INSIGHTS_DIR / "drug_insights.csv"
    effects_path = config.INSIGHTS_DIR / "side_effects_by_drug.csv"
    if not drugs_path.exists():
        return None, None
    return pd.read_csv(drugs_path, index_col=0), pd.read_csv(effects_path)


if not config.BASELINE_PATH.exists() and not (config.TRANSFORMER_DIR / "config.json").exists():
    st.error("No trained model found. Run `python main.py` from the project root, then reload.")
    st.stop()

problem = baseline_version_problem()
if problem and not (config.TRANSFORMER_DIR / "config.json").exists():
    st.error(problem)
    st.stop()

predictor = load_predictor()
drugs, effects = load_insights()


def highlight(text: str, words: list[tuple[str, float]], color: str) -> str:
    """Return HTML with the given words/phrases highlighted (text is escaped first)."""
    out = html.escape(text)
    for phrase, _ in sorted(words, key=lambda w: -len(w[0])):
        pattern = re.compile(rf"\b({re.escape(html.escape(phrase))})\b", re.IGNORECASE)
        out = pattern.sub(rf'<mark style="background:{color}33;border-radius:3px;'
                          rf'padding:0 2px">\1</mark>', out)
    return out


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("💊 Medication review analyzer")
st.caption("Reads what patients write about their medicines, classifies the sentiment, and "
           f"pulls out side effects. Active model: **{predictor.name}**.")

tab_review, tab_drug, tab_perf, tab_about = st.tabs(
    ["Analyze a review", "Drug insights", "Model performance", "About"])

# ---------------------------------------------------------------------------
# Tab 1 - single review
# ---------------------------------------------------------------------------
with tab_review:
    st.session_state.setdefault("review_text", EXAMPLES["Mixed"])

    def use_example(name):
        st.session_state["review_text"] = EXAMPLES[name]

    cols = st.columns([1, 1, 1, 4])
    for col, name in zip(cols, EXAMPLES):
        col.button(f"{name} example", on_click=use_example, args=(name,))

    text = st.text_area("Patient review", key="review_text", height=130)
    condition = st.text_input("Condition being treated (optional)",
                              placeholder="e.g. Depression - helps separate symptoms from side effects")

    if text.strip():
        result = predictor.predict(text)
        label = result["label"]
        left, right = st.columns([1, 1.6])

        with left:
            st.markdown(f"### {EMOJI[label]} {label.capitalize()}")
            for name in config.LABELS:
                p = result["probabilities"][name]
                st.markdown(f"<div style='font-size:0.9rem'>{name} <b>{p:.0%}</b></div>",
                            unsafe_allow_html=True)
                st.progress(p)
            found = detect_side_effects(text, condition)
            st.markdown("**Side effects mentioned**")
            st.markdown(" ".join(f"`{e}`" for e in found) if found else "_none detected_")

        with right:
            exp = predictor.explain(text, label, top_n=8)
            st.markdown(f"**Words that point to *{label}***")
            st.markdown(f"<div style='line-height:1.9'>{highlight(text, exp['supporting'], COLOR[label])}</div>",
                        unsafe_allow_html=True)
            if exp["opposing"]:
                st.markdown("**Words pulling the other way:** " +
                            ", ".join(f"`{w}`" for w, _ in exp["opposing"][:6]))
            st.caption("Word highlights come from the TF-IDF baseline's weights, which can be read "
                       "directly. They show which words drive the sentiment.")

# ---------------------------------------------------------------------------
# Tab 2 - drug insights
# ---------------------------------------------------------------------------
with tab_drug:
    if drugs is None:
        st.info("Run `python main.py --step eda` to build drug insights.")
    else:
        popular = drugs[drugs["reviews"] >= 50]
        default = popular.index.get_loc("Sertraline") if "Sertraline" in popular.index else 0
        drug = st.selectbox(f"Choose a drug ({len(popular):,} drugs with 50+ reviews)",
                            popular.index, index=default)
        row = popular.loc[drug]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Reviews", f"{int(row['reviews']):,}")
        m2.metric("Average rating", f"{row['avg_rating']:.1f} / 10")
        m3.metric("Positive", f"{row['pct_positive']:.0f}%")
        m4.metric("Negative", f"{row['pct_negative']:.0f}%")
        st.caption(f"Most reviewed for: **{row['top_condition']}**")

        drug_effects = effects[effects["drug"] == drug].nlargest(8, "mentions")
        if len(drug_effects):
            st.markdown("**What unhappy patients mention most** (share of this drug's negative reviews)")
            d = drug_effects.sort_values("pct_of_negative_reviews")
            chart_col, _ = st.columns([3, 2])
            fig, ax = plt.subplots(figsize=(6, 0.32 * len(d) + 0.7))
            bars = ax.barh(d["side_effect"], d["pct_of_negative_reviews"], color="#c0504d")
            ax.bar_label(bars, fmt="%.0f%%", fontsize=9, padding=3)
            ax.set_xlabel("% of negative reviews mentioning it")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            chart_col.pyplot(fig)
            plt.close(fig)
        else:
            st.markdown("_No side effects from the lexicon found in this drug's negative reviews._")

        c1, c2 = st.columns(2)
        for col, name, cap in [(c1, "insight_side_effects_overall.png", "Across all drugs"),
                               (c2, "insight_drug_ratings.png", "Lowest vs highest rated drugs")]:
            if (config.FIGURES_DIR / name).exists():
                col.image(str(config.FIGURES_DIR / name), caption=cap, width="stretch")

# ---------------------------------------------------------------------------
# Tab 3 - performance
# ---------------------------------------------------------------------------
with tab_perf:
    comp_path = config.REPORTS_DIR / "model_comparison.csv"
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        show = [c for c in ["model", "accuracy", "macro_f1", "f1_negative", "f1_neutral",
                            "f1_positive"] if c in comp.columns]
        st.subheader("Test-set results")
        st.dataframe(comp[show], hide_index=True, width="stretch")
        st.caption("Macro-F1 is the headline metric: it weighs the small neutral class as much as "
                   "the large positive class.")
    figs = [p for p in ["model_comparison.png", "confusion_matrix_transformer.png",
                        "confusion_matrix_baseline.png"] if (config.FIGURES_DIR / p).exists()]
    cols = st.columns(2)
    for i, name in enumerate(figs):
        cols[i % 2].image(str(config.FIGURES_DIR / name), width="stretch")
    if config.ERROR_ANALYSIS_FILE.exists():
        with st.expander("Error analysis report"):
            st.markdown(config.ERROR_ANALYSIS_FILE.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Tab 4 - about
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(
        """
**What this is.** An NLP project on ~215k real patient reviews from Drugs.com
(UCI Machine Learning Repository). It classifies each review as negative, neutral or
positive and summarises side effects per drug.

**Why it matters.** Pharmacy and benefits teams receive huge volumes of free-text
feedback. Reading it automatically surfaces drugs with poor patient experience and the
side effects driving it, which can feed adherence programmes and formulary decisions.

**How it works.** Leakage-safe preparation (the raw test file contains ~32k copies of
training reviews, all removed), a TF-IDF + Logistic Regression baseline, a fine-tuned
DistilBERT small language model trained with a PyTorch loop, error analysis, and a
condition-aware side-effect lexicon.

**Limits.** Ratings are self-reported and noisy (a 6 and a 7 often read the same).
Side-effect detection is keyword-based: it finds mentions, it does not prove the drug
caused them. Educational project, not medical advice.
        """
    )
