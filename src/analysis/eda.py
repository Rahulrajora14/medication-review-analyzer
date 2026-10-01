"""
Step 3 - Exploratory data analysis + drug-level insights.

Produces charts in reports/figures/ and CSV tables in reports/insights/
that the dashboard's "Drug insights" tab reads.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.analysis.side_effects import SIDE_EFFECTS, detect_side_effects
from src.data.prepare import load_split, rating_to_label

COLORS = {"negative": "#c0504d", "neutral": "#9a9a9a", "positive": "#2a7f8f"}
MIN_REVIEWS_PER_DRUG = 300   # only rank drugs with enough reviews to be meaningful


def load_all() -> pd.DataFrame:
    return pd.concat([load_split(s) for s in ("train", "val", "test")], ignore_index=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=130)
    plt.close(fig)


def plot_overview(df: pd.DataFrame) -> None:
    # 1. rating distribution coloured by the sentiment class it maps to
    counts = df["rating"].value_counts().sort_index()
    colors = [COLORS[config.LABELS[rating_to_label(r)]] for r in counts.index]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(counts.index.astype(int).astype(str), counts.values, color=colors)
    ax.set(title="Ratings are heavily skewed towards 10/10", xlabel="Patient rating",
           ylabel="Reviews")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "eda_rating_distribution.png")

    # 2. review length
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(df["n_words"].clip(upper=400), bins=60, color="#4f6d8f")
    ax.axvline(df["n_words"].median(), color="black", ls="--", lw=1,
               label=f"median = {int(df['n_words'].median())} words")
    ax.set(title="Review length (capped at 400 words)", xlabel="Words per review",
           ylabel="Reviews")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "eda_review_length.png")

    # 3. top conditions
    top = df.loc[df["condition"] != "Unknown", "condition"].value_counts().head(12)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top.values, color="#2a7f8f")
    ax.set(title="Most reviewed conditions", xlabel="Reviews")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "eda_top_conditions.png")


def build_drug_insights(df: pd.DataFrame) -> pd.DataFrame:
    """One row per drug: review count, average rating, sentiment shares, top condition."""
    grouped = df.groupby("drug")
    insights = pd.DataFrame({
        "reviews": grouped.size(),
        "avg_rating": grouped["rating"].mean().round(2),
        "pct_negative": (grouped["label"].apply(lambda s: (s == 0).mean()) * 100).round(1),
        "pct_neutral": (grouped["label"].apply(lambda s: (s == 1).mean()) * 100).round(1),
        "pct_positive": (grouped["label"].apply(lambda s: (s == 2).mean()) * 100).round(1),
        "top_condition": grouped["condition"].agg(lambda s: s.value_counts().index[0]),
    }).sort_values("reviews", ascending=False)
    return insights


def build_side_effect_table(df: pd.DataFrame) -> pd.DataFrame:
    """Long table: drug, side_effect, mentions in negative reviews, share of negative reviews."""
    neg = df[df["label"] == 0][["drug", "review", "condition"]].copy()
    neg["effects"] = [detect_side_effects(r, c) for r, c in zip(neg["review"], neg["condition"])]
    exploded = neg.explode("effects").dropna(subset=["effects"])
    table = (exploded.groupby(["drug", "effects"]).size().rename("mentions").reset_index()
             .rename(columns={"effects": "side_effect"}))
    neg_counts = neg.groupby("drug").size().rename("negative_reviews")
    table = table.join(neg_counts, on="drug")
    table["pct_of_negative_reviews"] = (100 * table["mentions"] / table["negative_reviews"]).round(1)
    return table.sort_values(["drug", "mentions"], ascending=[True, False])


def plot_insights(df, insights, side_effects) -> None:
    # overall side effects in negative reviews
    overall = side_effects.groupby("side_effect")["mentions"].sum().sort_values()
    n_neg = int((df["label"] == 0).sum())
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(overall.index, 100 * overall.values / n_neg, color="#c0504d")
    ax.set(title="Side effects mentioned in negative reviews",
           xlabel="% of all negative reviews mentioning it")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "insight_side_effects_overall.png")

    # best and worst rated drugs with enough reviews
    big = insights[insights["reviews"] >= MIN_REVIEWS_PER_DRUG]
    ranked = pd.concat([big.nsmallest(8, "avg_rating"), big.nlargest(8, "avg_rating")])
    ranked = ranked.sort_values("avg_rating")
    colors = ["#c0504d" if r < big["avg_rating"].median() else "#2a7f8f" for r in ranked["avg_rating"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(ranked.index, ranked["avg_rating"], color=colors)
    ax.set(title=f"Lowest and highest rated drugs ({MIN_REVIEWS_PER_DRUG}+ reviews)",
           xlabel="Average rating (1-10)", xlim=(0, 10))
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "insight_drug_ratings.png")


def run_eda() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all()
    print(f"  {len(df):,} unique reviews | {df['drug'].nunique():,} drugs | "
          f"{df['condition'].nunique():,} conditions")
    print(f"  Median review length: {int(df['n_words'].median())} words")

    plot_overview(df)

    insights = build_drug_insights(df)
    insights.to_csv(config.INSIGHTS_DIR / "drug_insights.csv")
    side_effects = build_side_effect_table(df)
    side_effects.to_csv(config.INSIGHTS_DIR / "side_effects_by_drug.csv", index=False)
    plot_insights(df, insights, side_effects)

    top = side_effects.groupby("side_effect")["mentions"].sum().nlargest(5)
    print("  Most mentioned side effects in negative reviews: " + ", ".join(top.index))
    print(f"  Lexicon covers {len(SIDE_EFFECTS)} side effects")
    print(f"  Charts -> {config.FIGURES_DIR}   Tables -> {config.INSIGHTS_DIR}")


if __name__ == "__main__":
    run_eda()
