"""
Step 2 - Clean, label, de-duplicate and split the data.

Rules (each explained in docs/HOW_IT_WORKS.md):
1. Clean text: decode HTML entities (&#039; -> '), strip wrapping quotes, squash spaces.
2. Fix broken condition values (some rows contain scraped HTML like "3</span> users...").
3. Label: rating 1-4 -> negative, 5-6 -> neutral, 7-10 -> positive.
4. De-duplicate: the same review is often listed under both the brand and the
   generic drug name. Keep one copy of each review text.
5. LEAKAGE FIX: drop test reviews whose text also appears in train.
   In the raw files ~60% of test reviews are exact copies of training reviews.
6. Split train into train / validation (stratified by label).
"""
import html
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

_SPACES = re.compile(r"\s+")


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = text.strip().strip('"').strip()
    return _SPACES.sub(" ", text)


def rating_to_label(rating: float) -> int:
    """1-4 -> 0 (negative), 5-6 -> 1 (neutral), 7-10 -> 2 (positive)."""
    if rating <= config.NEGATIVE_MAX_RATING:
        return 0
    if rating <= config.NEUTRAL_MAX_RATING:
        return 1
    return 2


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"Unnamed: 0": "review_id", "drugName": "drug",
                            "usefulCount": "useful_count"}).copy()
    df["review"] = df["review"].map(clean_text)
    df["condition"] = df["condition"].where(
        ~df["condition"].fillna("").str.contains("</span>"), np.nan
    ).fillna("Unknown")
    df["date"] = pd.to_datetime(df["date"], format="%B %d, %Y", errors="coerce")
    df["label"] = df["rating"].map(rating_to_label).astype(int)
    df["sentiment"] = df["label"].map(dict(enumerate(config.LABELS)))
    df["n_words"] = df["review"].str.split().str.len()
    return df[df["review"].str.len() > 0].reset_index(drop=True)


def prepare_data(verbose: bool = True) -> dict[str, pd.DataFrame]:
    train_raw = pd.read_csv(config.RAW_TRAIN, sep="\t")
    test_raw = pd.read_csv(config.RAW_TEST, sep="\t")
    log = [("Raw train reviews", len(train_raw)), ("Raw test reviews", len(test_raw))]

    train = clean_frame(train_raw)
    test = clean_frame(test_raw)

    # 4. one copy per review text (keep the most "useful"-voted copy)
    train = (train.sort_values("useful_count", ascending=False)
                  .drop_duplicates("review").reset_index(drop=True))
    test = (test.sort_values("useful_count", ascending=False)
                .drop_duplicates("review").reset_index(drop=True))
    log.append(("Train after de-duplication", len(train)))
    log.append(("Test after de-duplication", len(test)))

    # 5. leakage fix: no test review may also be a training review
    overlap = test["review"].isin(set(train["review"]))
    test = test[~overlap].reset_index(drop=True)
    log.append(("Test copies of train reviews removed", int(overlap.sum())))
    log.append(("Final test reviews", len(test)))

    # 6. validation split, stratified so class shares stay the same
    train, val = train_test_split(
        train, test_size=config.VAL_SIZE, stratify=train["label"],
        random_state=config.RANDOM_STATE,
    )
    log.append(("Final train reviews", len(train)))
    log.append(("Final validation reviews", len(val)))

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    splits = {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True),
              "test": test}
    for name, path in [("train", config.TRAIN_FILE), ("val", config.VAL_FILE),
                       ("test", config.TEST_FILE)]:
        splits[name].to_parquet(path, index=False)

    if verbose:
        for step, n in log:
            print(f"  {step:<40} {n:>9,}")
        shares = splits["train"]["sentiment"].value_counts(normalize=True)
        print("  Class shares (train): " + ", ".join(f"{k} {v:.1%}" for k, v in shares.items()))
    return splits


def load_split(name: str) -> pd.DataFrame:
    path = {"train": config.TRAIN_FILE, "val": config.VAL_FILE, "test": config.TEST_FILE}[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python main.py --step prepare` first.")
    return pd.read_parquet(path)


if __name__ == "__main__":
    prepare_data()
