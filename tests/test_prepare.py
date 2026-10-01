"""Tests for text cleaning, labelling and the de-duplication / leakage rules."""
import pandas as pd
import pytest

from src import config
from src.data import prepare


def test_clean_text_decodes_html_and_quotes():
    raw = '"I&#039;ve tried it &amp; it   works"'
    assert prepare.clean_text(raw) == "I've tried it & it works"
    assert prepare.clean_text(None) == ""


@pytest.mark.parametrize("rating, label", [(1, 0), (4, 0), (5, 1), (6, 1), (7, 2), (10, 2)])
def test_rating_to_label(rating, label):
    assert prepare.rating_to_label(rating) == label


def _raw(rows):
    return pd.DataFrame(rows, columns=["Unnamed: 0", "drugName", "condition", "review",
                                       "rating", "date", "usefulCount"])


def test_prepare_removes_duplicates_and_test_leakage(tmp_path, monkeypatch):
    base = "This drug worked well for me number {}"
    train_rows = [[i, "DrugA", "Pain", f'"{base.format(i)}"', 9 if i % 3 else 2,
                   "May 20, 2012", i] for i in range(60)]
    train_rows.append([999, "BrandA", "Pain", f'"{base.format(1)}"', 9, "May 20, 2012", 0])  # duplicate
    test_rows = [
        [500, "DrugA", "Pain", f'"{base.format(5)}"', 9, "May 20, 2012", 1],   # copy of a train review
        [501, "DrugA", "3</span> users found this comment helpful.", '"A brand new review"', 3,
         "June 1, 2013", 1],
    ]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _raw(train_rows).to_csv(raw_dir / "train.tsv", sep="\t", index=False)
    _raw(test_rows).to_csv(raw_dir / "test.tsv", sep="\t", index=False)
    monkeypatch.setattr(config, "RAW_TRAIN", raw_dir / "train.tsv")
    monkeypatch.setattr(config, "RAW_TEST", raw_dir / "test.tsv")
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "TRAIN_FILE", tmp_path / "processed" / "train.parquet")
    monkeypatch.setattr(config, "VAL_FILE", tmp_path / "processed" / "val.parquet")
    monkeypatch.setattr(config, "TEST_FILE", tmp_path / "processed" / "test.parquet")

    splits = prepare.prepare_data(verbose=False)
    train_val = pd.concat([splits["train"], splits["val"]])

    assert len(train_val) == 60                                  # duplicate removed
    assert splits["test"]["review"].tolist() == ["A brand new review"]  # leaked copy removed
    assert splits["test"]["condition"].iloc[0] == "Unknown"     # broken HTML condition fixed
    assert not set(splits["test"]["review"]) & set(train_val["review"])
