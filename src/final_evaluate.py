import os, sys, json, re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import joblib

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS = PROJECT_ROOT / "artifacts"
FINAL_MODEL_PATH = ARTIFACTS / "final_model" / "best_model.joblib"

# ---------- Matplotlib setup (avoid Tkinter issues on Windows) ----------
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".mpl-cache").absolute()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42

TFIDF_KWARGS = dict(ngram_range=(1, 2), min_df=2)
FINAL_MODEL_NAME = "LinearSVC"
FINAL_MODEL_KWARGS = dict(C=10)


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


def make_final_eval_folder() -> Path:
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_final_eval"
    out = ARTIFACTS / "final_eval" / run_id
    for sub in ["configs", "results", "plots"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("clf", LinearSVC(**FINAL_MODEL_KWARGS)),
    ])


def top_k_scores_from_saved_model(model, text: str, k: int = 3):
    pred = model.predict([text])[0]
    if hasattr(model, "decision_function"):
        scores = np.array(model.decision_function([text])).ravel()
        classes = model.classes_
        idx = np.argsort(scores)[::-1][:k]
        top = [(classes[i], float(scores[i])) for i in idx]
        return pred, top
    return pred, None


def main():
    out_dir = make_final_eval_folder()
    print("[INFO] Final eval folder:", out_dir)

    df, dataset_path = find_dataset_csv(DATA_DIR)
    df["text"] = df["text"].astype(str).map(clean_text)

    X = df["text"]
    y = df["label"].astype(str)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "acc": "accuracy",
        "prec_w": "precision_weighted",
        "rec_w": "recall_weighted",
        "f1w": "f1_weighted",
    }

    pipe = build_pipeline()

    # 1) Overall metrics
    scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

    metrics_summary = pd.DataFrame([{
        "model": f"{FINAL_MODEL_NAME}(C={FINAL_MODEL_KWARGS['C']})",
        "acc_mean": float(scores["test_acc"].mean()),
        "acc_std": float(scores["test_acc"].std()),
        "prec_w_mean": float(scores["test_prec_w"].mean()),
        "prec_w_std": float(scores["test_prec_w"].std()),
        "rec_w_mean": float(scores["test_rec_w"].mean()),
        "rec_w_std": float(scores["test_rec_w"].std()),
        "f1w_mean": float(scores["test_f1w"].mean()),
        "f1w_std": float(scores["test_f1w"].std()),
    }])

    metrics_csv = out_dir / "results" / "metrics_summary.csv"
    metrics_summary.to_csv(metrics_csv, index=False)

    # 2) Per-class report + confusion matrix (out-of-fold)
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)

    report_df = pd.DataFrame(classification_report(y, y_pred, output_dict=True, zero_division=0)).transpose()
    per_class_csv = out_dir / "results" / "per_class_report.csv"
    report_df.to_csv(per_class_csv)

    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(
        ax=ax, xticks_rotation=90, values_format="d", colorbar=False
    )
    plt.title(f"Confusion Matrix (Out-of-Fold) — {FINAL_MODEL_NAME}")
    plt.tight_layout()
    cm_path = out_dir / "plots" / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    # 3) Learning curve
    train_sizes = np.linspace(0.1, 1.0, 6)
    sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, cv=cv, train_sizes=train_sizes, scoring="f1_weighted", n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure()
    plt.plot(sizes, train_mean, label="train")
    plt.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(sizes, val_mean, label="validation")
    plt.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.title(f"Learning Curve (f1_weighted) — {FINAL_MODEL_NAME}")
    plt.xlabel("Training examples")
    plt.ylabel("F1 weighted")
    plt.legend()
    plt.tight_layout()
    lc_path = out_dir / "plots" / "learning_curve.png"
    plt.savefig(lc_path, dpi=200)
    plt.close()

    # 4) Example predictions (using saved model file)
    example_txt = out_dir / "results" / "example_predictions.txt"
    examples = [
        "itching skin rash redness",
        "stomach pain nausea vomiting",
        "fever sore throat headache",
    ]

    with open(example_txt, "w", encoding="utf-8") as f:
        f.write("Example predictions using saved final model\n")
        f.write("Note: decision scores are not probabilities; higher means more confident.\n\n")

        if FINAL_MODEL_PATH.exists():
            saved_model = joblib.load(FINAL_MODEL_PATH)
            for ex in examples:
                pred, top = top_k_scores_from_saved_model(saved_model, ex, k=3)
                f.write(f"Input: {ex}\n")
                f.write(f"Top prediction: {pred}\n")
                if top:
                    f.write("Top 3:\n")
                    for i, (label, score) in enumerate(top, start=1):
                        f.write(f"  {i}) {label} (score={score:.3f})\n")
                f.write("\n")
        else:
            f.write(f"[WARN] Missing saved model file: {FINAL_MODEL_PATH}\n")

    # 5) Save config
    cfg_path = out_dir / "configs" / "final_eval_config.json"
    cfg = {
        "dataset_file": str(dataset_path),
        "seed": SEED,
        "cv": {"type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": SEED},
        "tfidf": {"ngram_range": [1, 2], "min_df": TFIDF_KWARGS["min_df"]},
        "final_model": {"name": FINAL_MODEL_NAME, "params": FINAL_MODEL_KWARGS},
        "outputs": {
            "metrics_summary_csv": str(metrics_csv),
            "per_class_report_csv": str(per_class_csv),
            "confusion_matrix_png": str(cm_path),
            "learning_curve_png": str(lc_path),
            "example_predictions_txt": str(example_txt),
        }
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("\n[OK] Final evaluation pack created:")
    print(" -", metrics_csv)
    print(" -", per_class_csv)
    print(" -", cm_path)
    print(" -", lc_path)
    print(" -", example_txt)
    print(" -", cfg_path)


if __name__ == "__main__":
    main()