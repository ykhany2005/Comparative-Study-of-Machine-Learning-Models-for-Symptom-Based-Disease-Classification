import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from common import DATA_DIR, find_dataset_csv


def test_dataset_loader_finds_valid_dataset():
    df, path = find_dataset_csv(DATA_DIR)

    assert path.exists()
    assert "text" in df.columns
    assert "label" in df.columns
    assert len(df) > 0


def test_dataset_has_expected_basic_shape():
    df, _ = find_dataset_csv(DATA_DIR)

    assert len(df) == 1200
    assert df["label"].nunique() == 24


def test_dataset_text_and_label_not_empty():
    df, _ = find_dataset_csv(DATA_DIR)

    assert df["text"].notna().all()
    assert df["label"].notna().all()