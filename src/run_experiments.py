import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from common import SEED, DATA_DIR, RUNS_DIR, TFIDF_KWARGS, clean_text, find_dataset_csv


os.environ.setdefault("MPLCONFIGDIR", str(Path(".mpl-cache").absolute()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

scoring = {
    "acc": "accuracy",
    "prec_w": "precision_weighted",
    "rec_w": "recall_weighted",
    "f1w": "f1_weighted",
}


def build_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("clf", model),
    ])


def evaluate_model(X_text, y, model, name):
    pipe = build_pipeline(model)

    out = cross_validate(
        pipe,
        X_text,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    return {
        "model": name,
        "acc_mean": float(out["test_acc"].mean()),
        "acc_std": float(out["test_acc"].std()),
        "prec_w_mean": float(out["test_prec_w"].mean()),
        "prec_w_std": float(out["test_prec_w"].std()),
        "rec_w_mean": float(out["test_rec_w"].mean()),
        "rec_w_std": float(out["test_rec_w"].std()),
        "f1w_mean": float(out["test_f1w"].mean()),
        "f1w_std": float(out["test_f1w"].std()),
    }


def make_run_folder(tag: str = "baseline") -> Path:
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S") + f"_{tag}"
    run_dir = RUNS_DIR / run_id
    for sub in ["configs", "data", "results", "plots", "models"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def per_class_and_confusion(X_text, y, model, name, run_dir):
    pipe = build_pipeline(model)

    y_pred = cross_val_predict(pipe, X_text, y, cv=cv, n_jobs=-1)

    report_dict = classification_report(y, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    out_csv = run_dir / "results" / f"per_class_{name}.csv"
    report_df.to_csv(out_csv)

    return y_pred, out_csv


def save_learning_curve(run_dir, X_text, y, model, name):
    import matplotlib.pyplot as plt

    pipe = build_pipeline(model)
    train_sizes = np.linspace(0.1, 1.0, 6)

    sizes, train_scores, val_scores = learning_curve(
        pipe,
        X_text,
        y,
        cv=cv,
        train_sizes=train_sizes,
        scoring="f1_weighted",
        n_jobs=-1,
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
    plt.title(f"Learning Curve (f1_weighted) — {name}")
    plt.xlabel("Training examples")
    plt.ylabel("F1 weighted")
    plt.legend()
    plt.tight_layout()

    out_path = run_dir / "plots" / f"learning_curve_{name}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("\n[INFO] Learning curve saved:")
    print(" -", out_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of model names to run (e.g. --models LinearSVC LogisticRegression)",
    )
    return parser.parse_args()


def main():
    df, chosen_path = find_dataset_csv(DATA_DIR)

    run_dir = make_run_folder(tag="baseline")
    print(f"[INFO] Run folder: {run_dir}")

    raw_copy_path = run_dir / "data" / "raw_dataset.csv"
    shutil.copy2(chosen_path, raw_copy_path)

    df["text"] = df["text"].astype(str).map(clean_text)
    cleaned_path = run_dir / "data" / "cleaned_dataset.csv"
    df.to_csv(cleaned_path, index=False)

    X_text = df["text"]
    y = df["label"].astype(str)

    args = parse_args()

    model_registry = {
        "LogisticRegression": LogisticRegression(max_iter=2000, solver="saga", random_state=SEED),
        "LinearSVC": LinearSVC(random_state=SEED),
        "MultinomialNB": MultinomialNB(),
        "SGDClassifier_SVM": SGDClassifier(loss="hinge", random_state=SEED),
        "ComplementNB": ComplementNB(),
    }

    if args.models:
        unknown = [m for m in args.models if m not in model_registry]
        if unknown:
            print("[ERROR] Unknown model names:", unknown)
            print("Valid options:", list(model_registry.keys()))
            sys.exit(1)
        selected_models = {k: model_registry[k] for k in args.models}
    else:
        selected_models = model_registry

    print("[INFO] Models selected:", list(selected_models.keys()))

    rows = []
    for name, model in selected_models.items():
        rows.append(evaluate_model(X_text, y, model, name))

    res = pd.DataFrame(rows).sort_values("f1w_mean", ascending=False)

    baseline_csv = run_dir / "results" / "baseline_table.csv"
    res.to_csv(baseline_csv, index=False)

    print("\n[INFO] Baseline results saved:")
    print(" -", baseline_csv)
    print(res)

    best_name = res.iloc[0]["model"]
    best_model = selected_models[best_name]

    save_learning_curve(run_dir, X_text, y, best_model, best_name)

    per_class_paths = {}
    pred_store = {}

    for name, model in selected_models.items():
        y_pred, p_csv = per_class_and_confusion(X_text, y, model, name, run_dir)
        per_class_paths[name] = p_csv
        pred_store[name] = y_pred

    print("\n[INFO] Per-class reports saved:")
    for name, p in per_class_paths.items():
        print(f" - {name}: {p}")

    import matplotlib.pyplot as plt

    best_pred = pred_store[best_name]
    labels = sorted(y.unique())

    cm = confusion_matrix(y, best_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=90, values_format="d", colorbar=False)
    plt.title(f"Confusion Matrix (Out-of-Fold) — {best_name}")
    plt.tight_layout()

    cm_path = run_dir / "plots" / f"confusion_matrix_{best_name}.png"
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    print("\n[INFO] Confusion matrix saved:")
    print(" -", cm_path)

    run_config = {
        "dataset_file_used": str(chosen_path),
        "raw_copy_saved_as": str(raw_copy_path),
        "cleaned_copy_saved_as": str(cleaned_path),
        "seed": SEED,
        "models_selected": list(selected_models.keys()),
        "best_model": best_name,
        "cv": {
            "type": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": SEED,
        },
        "preprocessing": {
            "clean_text": "lowercase + remove punctuation/symbols + normalise whitespace"
        },
        "tfidf": {
            "ngram_range": list(TFIDF_KWARGS["ngram_range"]),
            "min_df": TFIDF_KWARGS["min_df"],
        },
        "notes": "Objective 3 baseline evaluation with archived data, CV metrics, per-class reports, confusion matrix, and learning curve",
    }

    config_path = run_dir / "configs" / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("\n[INFO] Dataset used:", chosen_path)
    print("[INFO] Rows:", len(df), "| Classes:", df["label"].nunique())
    print("[INFO] Saved:")
    print(" -", raw_copy_path)
    print(" -", cleaned_path)
    print(" -", config_path)

    print("\n[OK] Baseline evaluation complete.")
    print(f"[OK] Open this run folder: {run_dir}")


if __name__ == "__main__":
    main()
