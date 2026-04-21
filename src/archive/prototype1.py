# prototype1.py — safe to run before dataset is downloaded.
# It sets a local matplotlib cache (faster on Windows) and exits cleanly
# if data/Training.csv is not present yet.

import os, sys, json
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB

# --- speed up matplotlib on Windows/OneDrive by using a local cache dir ---
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mpl-cache").absolute()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

# --- paths & setup ---
SEED = 42
DATA = Path("data/Training.csv")
ART_EDA = Path("artifacts/eda"); ART_RES = Path("artifacts/results")
ART_EDA.mkdir(parents=True, exist_ok=True); ART_RES.mkdir(parents=True, exist_ok=True)

# --- check dataset presence early ---
if not DATA.exists():
    print(f"[INFO] Missing {DATA}.")
    print("→ Download the Kaggle dataset and place Training.csv into the data/ folder.")
    print("  Kaggle slug: kaushil268/disease-prediction-using-machine-learning")
    sys.exit(0)

# --- load data ---
df = pd.read_csv(DATA)
target = "prognosis"
X = df.drop(columns=[target])
y = df[target]

# import matplotlib only when needed
import matplotlib.pyplot as plt

# --- quick EDA: class distribution ---
cls_counts = y.value_counts().sort_values(ascending=False)
plt.figure()
cls_counts.plot(kind="bar")
plt.title("Class distribution")
plt.tight_layout()
plt.savefig(ART_EDA / "class_dist.png", dpi=150)
plt.close()

# --- preprocessing (leakage-safe) ---
numeric_cols = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

numeric = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("scale", StandardScaler(with_mean=False)),
])
categorical = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

pre = ColumnTransformer([
    ("num", numeric, numeric_cols),
    ("cat", categorical, categorical_cols),
], remainder="drop")

# --- shared CV + metrics ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {"acc": "accuracy", "f1w": "f1_weighted"}

def evaluate(model, name):
    pipe = Pipeline([("pre", pre), ("clf", model)])
    out = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
    return {
        "model": name,
        "acc_mean": float(out["test_acc"].mean()),
        "acc_std":  float(out["test_acc"].std()),
        "f1w_mean": float(out["test_f1w"].mean()),
        "f1w_std":  float(out["test_f1w"].std()),
    }

rows = []
rows.append(evaluate(LogisticRegression(max_iter=1000), "LogisticRegression"))
rows.append(evaluate(BernoulliNB(), "BernoulliNB"))
rows.append(evaluate(DecisionTreeClassifier(max_depth=6, random_state=SEED), "DecisionTree"))

res = pd.DataFrame(rows).sort_values("f1w_mean", ascending=False)
print(res)
res.to_csv(ART_RES / "prototype1_baselines.csv", index=False)

with open(ART_RES / "run_params.json", "w") as f:
    json.dump({"seed": SEED, "cv": 5, "dataset": str(DATA)}, f, indent=2)

print("\nSaved:")
print(f" - {ART_EDA / 'class_dist.png'}")
print(f" - {ART_RES / 'prototype1_baselines.csv'}")
