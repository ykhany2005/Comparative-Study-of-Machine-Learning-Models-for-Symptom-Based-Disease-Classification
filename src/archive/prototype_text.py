# prototype_text.py — Symptom2Disease TF-IDF + baseline models
# Outputs:
#  - artifacts/eda/class_dist_text.png
#  - artifacts/eda/text_length_hist.png
#  - artifacts/results/prototype_text_baselines.csv

import os, sys, json, re
from pathlib import Path
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

# --- speed up matplotlib on Windows by using a local cache dir ---
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mpl-cache").absolute()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

SEED = 42
DATA_DIR = Path("data")
ART_EDA = Path("artifacts/eda")
ART_RES = Path("artifacts/results")
ART_EDA.mkdir(parents=True, exist_ok=True)
ART_RES.mkdir(parents=True, exist_ok=True)

# ---- find a CSV in data/ that contains 'text' and 'label' columns ----
csvs = list(DATA_DIR.glob("*.csv"))
if not csvs:
    print("[INFO] No CSV found in data/. Put the Symptom2Disease CSV into data/ and rerun.")
    sys.exit(0)

df = None
chosen = None
for f in csvs:
    tmp = pd.read_csv(f)
    cols_lower = {c.lower() for c in tmp.columns}
    if {"text", "label"}.issubset(cols_lower):
        df = tmp
        chosen = f
        break

if df is None:
    print("[INFO] Found CSV files, but none had columns 'text' and 'label'.")
    print("Files found:", [f.name for f in csvs])
    print("Tip: open the CSV and check the column names. Tell me what they are and I’ll adapt the script.")
    sys.exit(0)

# ---- normalise column names just in case they differ in capitalisation ----
rename_map = {c: c.lower() for c in df.columns}
df = df.rename(columns=rename_map)

# ---- basic NLP cleanup (lightweight & safe) ----
def clean_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)   # remove punctuation/symbols
    s = re.sub(r"\s+", " ", s).strip()   # normalise whitespace
    return s

df["text"] = df["text"].astype(str).map(clean_text)
X_text = df["text"]
y = df["label"].astype(str)

# ---- EDA outputs ----
import matplotlib.pyplot as plt

# class distribution plot
class_counts = y.value_counts().sort_values(ascending=False)
plt.figure()
class_counts.plot(kind="bar")
plt.title("Class distribution (Symptom2Disease)")
plt.tight_layout()
plt.savefig(ART_EDA / "class_dist_text.png", dpi=150)
plt.close()

# text length histogram
lengths = X_text.str.split().map(len)
plt.figure()
plt.hist(lengths, bins=30)
plt.title("Text length (words)")
plt.xlabel("words")
plt.ylabel("count")
plt.tight_layout()
plt.savefig(ART_EDA / "text_length_hist.png", dpi=150)
plt.close()

# ---- CV + scoring ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {"acc": "accuracy", "f1w": "f1_weighted"}

def evaluate(model, name):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("clf", model),
    ])
    out = cross_validate(pipe, X_text, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    return {
        "model": name,
        "acc_mean": float(out["test_acc"].mean()),
        "acc_std":  float(out["test_acc"].std()),
        "f1w_mean": float(out["test_f1w"].mean()),
        "f1w_std":  float(out["test_f1w"].std()),
    }

rows = []
rows.append(evaluate(LogisticRegression(max_iter=2000, solver="saga"), "LogisticRegression"))
rows.append(evaluate(LinearSVC(), "LinearSVC"))
rows.append(evaluate(MultinomialNB(), "MultinomialNB"))

res = pd.DataFrame(rows).sort_values("f1w_mean", ascending=False)

print("\nDataset used:", chosen)
print("Rows:", len(df), "| Classes:", y.nunique())
print(res)

# save results
res.to_csv(ART_RES / "prototype_text_baselines.csv", index=False)
with open(ART_RES / "run_params_text.json", "w") as f:
    json.dump({"seed": SEED, "cv": 5, "dataset": str(chosen)}, f, indent=2)

print("\nSaved:")
print(" - artifacts/eda/class_dist_text.png")
print(" - artifacts/eda/text_length_hist.png")
print(" - artifacts/results/prototype_text_baselines.csv")
