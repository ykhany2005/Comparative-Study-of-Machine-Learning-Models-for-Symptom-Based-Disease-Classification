import os
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier

from common import SEED, DATA_DIR, RUNS_DIR, TFIDF_KWARGS, clean_text, find_dataset_csv


os.environ.setdefault("MPLCONFIGDIR", str(Path(".mpl-cache").absolute()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def make_run_folder(tag: str = "tuning") -> Path:
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S") + f"_{tag}"
    run_dir = RUNS_DIR / run_id
    for sub in ["configs", "data", "results", "plots", "models"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["LinearSVC", "LogisticRegression", "SGDClassifier_SVM"],
        help="Which model to tune",
    )
    return parser.parse_args()


def build_base_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("clf", LinearSVC(random_state=SEED)),  # placeholder; replaced below
    ])


def get_model_and_grid(model_name: str):
    if model_name == "LinearSVC":
        clf = LinearSVC(random_state=SEED)
        param_grid = {
            "clf__C": [0.1, 1, 10, 30],
        }

    elif model_name == "LogisticRegression":
        clf = LogisticRegression(max_iter=3000, solver="saga", random_state=SEED)
        param_grid = {
            "clf__C": [0.1, 1, 10, 30],
        }

    else:  # SGDClassifier_SVM
        clf = SGDClassifier(loss="hinge", random_state=SEED)
        param_grid = {
            "clf__alpha": [1e-5, 1e-4, 1e-3],
            "clf__penalty": ["l2", "l1"],
        }

    return clf, param_grid


def main():
    args = parse_args()

    df, chosen_path = find_dataset_csv(DATA_DIR)

    run_dir = make_run_folder(tag="tuning")
    print(f"[INFO] Run folder: {run_dir}")

    raw_copy_path = run_dir / "data" / "raw_dataset.csv"
    shutil.copy2(chosen_path, raw_copy_path)

    df["text"] = df["text"].astype(str).map(clean_text)
    cleaned_path = run_dir / "data" / "cleaned_dataset.csv"
    df.to_csv(cleaned_path, index=False)

    X_text = df["text"]
    y = df["label"].astype(str)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    pipe = build_base_pipeline()
    clf, param_grid = get_model_and_grid(args.model)
    pipe.set_params(clf=clf)

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    print(f"[INFO] Tuning model: {args.model}")
    search.fit(X_text, y)

    results_df = pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False)
    cv_path = run_dir / "results" / f"cv_results_{args.model}.csv"
    results_df.to_csv(cv_path, index=False)

    best_params_path = run_dir / "results" / f"best_params_{args.model}.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(search.best_params_, f, indent=2)

    best_model_path = run_dir / "models" / f"best_{args.model}.joblib"
    joblib.dump(search.best_estimator_, best_model_path)

    tune_config = {
        "dataset_file_used": str(chosen_path),
        "raw_copy_saved_as": str(raw_copy_path),
        "cleaned_copy_saved_as": str(cleaned_path),
        "seed": SEED,
        "cv": {
            "type": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": SEED,
        },
        "tfidf": {
            "ngram_range": list(TFIDF_KWARGS["ngram_range"]),
            "min_df": TFIDF_KWARGS["min_df"],
        },
        "model": args.model,
        "param_grid": param_grid,
        "best_params": search.best_params_,
        "best_score_f1_weighted": float(search.best_score_),
    }

    cfg_path = run_dir / "configs" / "tune_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(tune_config, f, indent=2)

    print("\n[INFO] Best score (f1_weighted):", search.best_score_)
    print("[INFO] Best params:", search.best_params_)
    print("\n[INFO] Saved:")
    print(" -", cv_path)
    print(" -", best_params_path)
    print(" -", best_model_path)
    print(" -", cfg_path)
    print(" -", raw_copy_path)
    print(" -", cleaned_path)


if __name__ == "__main__":
    main()