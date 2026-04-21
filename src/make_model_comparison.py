import json
from pathlib import Path

import pandas as pd

from common import PROJECT_ROOT


ARTIFACTS = PROJECT_ROOT / "artifacts"
TRANSFORMER_DIR = ARTIFACTS / "transformer"
RESULTS_DIR = ARTIFACTS / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = RESULTS_DIR / "model_comparison.csv"


def find_latest_distilbert_metrics():
    runs = sorted(
        TRANSFORMER_DIR.glob("*_distilbert"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise FileNotFoundError("No DistilBERT runs found in artifacts/transformer/")

    latest_run = runs[0]
    metrics_path = latest_run / "results" / "metrics_summary.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics_summary.json in {latest_run}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return latest_run, metrics


def find_best_linearsvc_tuning():
    tuning_runs = sorted(
        (ARTIFACTS / "runs").glob("*_tuning"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    best_row = None

    for run in tuning_runs:
        cv_path = run / "results" / "cv_results_LinearSVC.csv"
        params_path = run / "results" / "best_params_LinearSVC.json"

        if not cv_path.exists():
            continue

        df = pd.read_csv(cv_path)
        best_score = float(df["mean_test_score"].max())

        best_params = None
        if params_path.exists():
            with open(params_path, "r", encoding="utf-8") as f:
                best_params = json.load(f)

        row = {
            "run_folder": str(run),
            "best_score_f1_weighted": best_score,
            "best_params": json.dumps(best_params) if best_params is not None else "",
        }

        if best_row is None or row["best_score_f1_weighted"] > best_row["best_score_f1_weighted"]:
            best_row = row

    if best_row is None:
        raise FileNotFoundError("No LinearSVC tuning results found in artifacts/runs/")

    return best_row


def main():
    distilbert_run, distilbert_metrics = find_latest_distilbert_metrics()
    best_linearsvc = find_best_linearsvc_tuning()

    rows = [
        {
            "model": "LinearSVC",
            "variant": "tuned",
            "evaluation_setup": "Stratified 5-Fold CV",
            "accuracy": "",
            "precision_weighted": "",
            "recall_weighted": "",
            "f1_weighted": best_linearsvc["best_score_f1_weighted"],
            "best_params": best_linearsvc["best_params"],
            "source_run": best_linearsvc["run_folder"],
            "notes": "Best classical model",
        },
        {
            "model": "DistilBERT",
            "variant": "4 epochs",
            "evaluation_setup": "Stratified train/validation split",
            "accuracy": distilbert_metrics.get("accuracy", ""),
            "precision_weighted": distilbert_metrics.get("precision_weighted", ""),
            "recall_weighted": distilbert_metrics.get("recall_weighted", ""),
            "f1_weighted": distilbert_metrics.get("f1_weighted", ""),
            "best_params": json.dumps(
                {
                    "model_name": distilbert_metrics.get("model_name", "distilbert-base-uncased"),
                    "num_epochs": 4,
                }
            ),
            "source_run": str(distilbert_run),
            "notes": "Best transformer run",
        },
    ]

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(OUT_CSV, index=False)

    print("[OK] Saved comparison file:")
    print(" -", OUT_CSV)
    print("\nComparison:")
    print(comparison_df)


if __name__ == "__main__":
    main()