"""
Step 4 - Baseline model: TF-IDF + Logistic Regression.

TF-IDF turns each review into a long vector of word (and word-pair) scores:
  - TF  (term frequency): how often a word appears in THIS review
  - IDF (inverse document frequency): how rare the word is across ALL reviews
  Common words like "the" get tiny weights; telling words like "miracle" or
  "nightmare" get large ones.

Logistic Regression then learns one weight per word per class. It is fast,
strong on text, and fully interpretable: we can list the words that push a
review towards each sentiment. Every fancier model must beat this.
"""
import json

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src import config
from src.data.prepare import load_split
from src.models.evaluate import save_results


def make_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=config.TFIDF_NGRAMS,     # single words + word pairs ("not good")
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=3,                            # ignore words seen in < 3 reviews
            sublinear_tf=True,                   # 1 + log(count): dampens repeated words
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=C,                                 # inverse regularisation strength
            class_weight="balanced",             # rare neutral class gets more weight
            max_iter=2000,
        )),
    ])


def top_words_per_class(pipeline: Pipeline, n: int = 15) -> dict[str, list[str]]:
    vocab = np.array(pipeline.named_steps["tfidf"].get_feature_names_out())
    coefs = pipeline.named_steps["clf"].coef_
    return {label: vocab[np.argsort(coefs[i])[::-1][:n]].tolist()
            for i, label in enumerate(config.LABELS)}


def train_baseline() -> dict:
    train, val, test = load_split("train"), load_split("val"), load_split("test")

    # Pick C on the VALIDATION set (the test set is only used once, at the end)
    best_C, best_f1 = None, -1.0
    for C in config.LOGREG_C_GRID:
        model = make_pipeline(C).fit(train["review"], train["label"])
        f1 = f1_score(val["label"], model.predict(val["review"]), average="macro")
        print(f"  C={C:<4} validation macro-F1 = {f1:.4f}")
        if f1 > best_f1:
            best_C, best_f1 = C, f1
    print(f"  Best C = {best_C}")

    model = make_pipeline(best_C).fit(train["review"], train["label"])
    proba = model.predict_proba(test["review"])
    y_pred = proba.argmax(axis=1)
    metrics = save_results("baseline", "TF-IDF + Logistic Regression", test, y_pred, proba,
                           extra={"val_macro_f1": round(best_f1, 4), "C": best_C})

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.BASELINE_PATH, compress=3)
    words = top_words_per_class(model)
    config.BASELINE_META.write_text(json.dumps({
        "C": best_C, "top_words": words,
        "library_versions": {"scikit-learn": sklearn.__version__},
    }, indent=2))

    print(f"  Test: accuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f} | "
          f"F1 neg {metrics['f1_negative']:.3f} / neu {metrics['f1_neutral']:.3f} / "
          f"pos {metrics['f1_positive']:.3f}")
    for label, w in words.items():
        print(f"  Top {label} words: {', '.join(w[:10])}")
    return metrics


if __name__ == "__main__":
    train_baseline()
