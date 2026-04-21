import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from common import clean_text


def test_clean_text_lowercases_and_removes_punctuation():
    text = "Fever, Sore-Throat!!"
    cleaned = clean_text(text)
    assert cleaned == "fever sore throat"


def test_clean_text_normalises_whitespace():
    text = "  cough   chest   pain  "
    cleaned = clean_text(text)
    assert cleaned == "cough chest pain"


def test_clean_text_handles_numbers_and_symbols():
    text = "Pain level 8/10!!!"
    cleaned = clean_text(text)
    assert cleaned == "pain level 8 10"