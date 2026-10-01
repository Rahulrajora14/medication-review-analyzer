"""
Model tests.
- Baseline: trains on a tiny dataset and checks the interpretable word weights.
- Transformer: builds a TINY DistilBERT-shaped model locally (no download) and runs the
  real training loop end to end. Skipped automatically if PyTorch isn't installed.
"""
import collections
import re

import pandas as pd
import pytest

from src.models.baseline import make_pipeline, top_words_per_class

POS = ["great drug works amazing", "love it no side effects", "best medicine ever amazing"]
NEU = ["it helps but side effects", "works okay however tired", "some relief but not sure"]
NEG = ["terrible awful nausea stopped", "worst drug never again", "horrible headaches awful"]


def _toy_df(n=10):
    rows = [(t, 2) for t in POS] + [(t, 1) for t in NEU] + [(t, 0) for t in NEG]
    df = pd.DataFrame(rows * n, columns=["review", "label"])
    df["review_id"] = range(len(df))
    for col, val in [("drug", "X"), ("condition", "Pain"), ("rating", 5)]:
        df[col] = val
    return df


def test_baseline_learns_obvious_words():
    df = _toy_df()
    model = make_pipeline(C=1.0)
    model.named_steps["tfidf"].set_params(min_df=1)
    model.fit(df["review"], df["label"])
    assert model.predict(["amazing great drug"])[0] == 2
    assert model.predict(["awful terrible nausea"])[0] == 0
    words = top_words_per_class(model, n=5)
    assert set(words) == {"negative", "neutral", "positive"}


def _require_torch():
    """Skip (not fail) if PyTorch is missing OR installed but unusable."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"PyTorch / transformers not usable here: {err.__class__.__name__}")


def _make_tiny_model(texts, out_dir):
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    words = collections.Counter(w for t in texts for w in re.findall(r"[a-z']+", t.lower()))
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocab = {w: i for i, w in enumerate(specials + sorted(words))}
    tk = tokenizers.Tokenizer(tokenizers.models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    tk.normalizer = tokenizers.normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
    tk.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]")
    model = transformers.DistilBertForSequenceClassification(transformers.DistilBertConfig(
        vocab_size=len(vocab), dim=32, n_layers=1, n_heads=2, hidden_dim=64, num_labels=3))
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)


def test_transformer_training_loop_runs(tmp_path):
    _require_torch()
    from src.models.transformer import train_transformer

    df = _toy_df(n=6)
    _make_tiny_model(df["review"], tmp_path / "tiny")
    metrics = train_transformer(
        model_name=str(tmp_path / "tiny"), output_dir=tmp_path / "out", epochs=1,
        batch_size=8, max_length=16, train_subset=None, save_outputs=False,
        data={"train": df, "val": df.head(18), "test": df.head(18)},
    )
    assert (tmp_path / "out" / "config.json").exists()           # best checkpoint saved
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["best_epoch"] == 1
