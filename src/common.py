import re
import sys
from pathlib import Path

import pandas as pd

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS / "runs"
FINAL_MODEL_PATH = ARTIFACTS / "final_model" / "best_model.joblib"

TFIDF_KWARGS = {
    "ngram_range": (1, 2),
    "min_df": 2,
}

RUNS_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_dataset_csv(data_dir: Path) -> tuple[pd.DataFrame, Path]:
    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        print("[ERROR] No CSV found in data/. Put Symptom2Disease.csv into data/ and rerun.")
        sys.exit(1)

    for f in csvs:
        tmp = pd.read_csv(f)
        cols = {c.lower() for c in tmp.columns}
        if {"text", "label"}.issubset(cols):
            df = tmp.rename(columns={c: c.lower() for c in tmp.columns})
            return df, f

    print("[ERROR] Found CSV files, but none had columns 'text' and 'label'.")
    print("Files found:", [f.name for f in csvs])
    sys.exit(1)