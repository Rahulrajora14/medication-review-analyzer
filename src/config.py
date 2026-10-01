"""
Central configuration: every path and setting in one place.
Change a value here and the whole pipeline picks it up.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_TRAIN = RAW_DIR / "drugsComTrain_raw.tsv"
RAW_TEST = RAW_DIR / "drugsComTest_raw.tsv"

TRAIN_FILE = PROCESSED_DIR / "train.parquet"
VAL_FILE = PROCESSED_DIR / "val.parquet"
TEST_FILE = PROCESSED_DIR / "test.parquet"

MODELS_DIR = ROOT / "models"
BASELINE_PATH = MODELS_DIR / "baseline_tfidf_logreg.joblib"
BASELINE_META = MODELS_DIR / "baseline_metadata.json"
TRANSFORMER_DIR = MODELS_DIR / "distilbert"          # fine-tuned model is saved here

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
PREDICTIONS_DIR = REPORTS_DIR / "predictions"
INSIGHTS_DIR = REPORTS_DIR / "insights"
ERROR_ANALYSIS_FILE = REPORTS_DIR / "error_analysis.md"

README_FILE = ROOT / "README.md"

# ---------------------------------------------------------------------------
# Data source (UCI ML Repository, dataset 462)
# ---------------------------------------------------------------------------
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/462/drug+review+dataset+drugs+com.zip"
MIRROR_BASE = "https://raw.githubusercontent.com/OdeliaAhdout/Drug-dataset/main/data/"

# ---------------------------------------------------------------------------
# Labels: patient's 1-10 rating -> sentiment class
# ---------------------------------------------------------------------------
LABELS = ["negative", "neutral", "positive"]   # index = class id
NEGATIVE_MAX_RATING = 4     # 1-4  -> negative
NEUTRAL_MAX_RATING = 6      # 5-6  -> neutral, 7-10 -> positive

# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
VAL_SIZE = 0.10             # share of the training file used for validation

# ---------------------------------------------------------------------------
# Baseline: TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAMS = (1, 2)
LOGREG_C_GRID = [0.25, 0.5, 1.0, 2.0]   # tried on the validation set

# ---------------------------------------------------------------------------
# Transformer: DistilBERT fine-tuning (run on a GPU - see notebooks/)
# ---------------------------------------------------------------------------
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256            # tokens per review (longer reviews are cut)
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
TRAIN_SUBSET = None         # e.g. 40_000 for a quicker run; None = use all
