import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import joblib

from common import FINAL_MODEL_PATH


def test_final_model_file_exists():
    assert FINAL_MODEL_PATH.exists()


def test_final_model_can_be_loaded():
    model = joblib.load(FINAL_MODEL_PATH)
    assert model is not None


def test_final_model_has_predict_method():
    model = joblib.load(FINAL_MODEL_PATH)
    assert hasattr(model, "predict")