import json
from pathlib import Path

import pandas as pd


# Always locate the project root correctly (works in PyCharm + PowerShell)
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../fyp
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"

# Find latest baseline run automatically
baseline_runs = sorted(RUNS_DIR.glob("*_baseline"), key=lambda p: p.stat().st_mtime, reverse=True)
if not baseline_runs:
    raise FileNotFoundError(
        f"No baseline runs found in: {RUNS_DIR}\n"
        f"Run: python .\\src\\run_experiments.py to create one."
    )

latest_baseline = baseline_runs[0]
BASELINE_RUN = latest_baseline / "results" / "baseline_table.csv"
print("[INFO] Using baseline:", BASELINE_RUN)

# Find all tuning runs automatically
TUNING_RUNS = sorted(RUNS_DIR.glob("*_tuning"))

OUT_DIR = PROJECT_ROOT / "artifacts" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "tuned_summary.csv"


def read_best_score_from_cv_results(csv_path: Path) -> float:
    df = pd.read_csv(csv_path)
    return float(df["mean_test_score"].max())


def main():
    # Baseline table: only need model + baseline f1
    baseline = pd.read_csv(BASELINE_RUN)[["model", "f1w_mean"]].copy()
    baseline = baseline.rename(columns={"f1w_mean": "baseline_f1w_mean"})

    rows = []

    for run in TUNING_RUNS:
        results_dir = run / "results"
        if not results_dir.exists():
            continue

        for params_file in results_dir.glob("best_params_*.json"):
            model_name = params_file.stem.replace("best_params_", "")
            cv_file = results_dir / f"cv_results_{model_name}.csv"
            if not cv_file.exists():
                continue

            best_params = json.loads(params_file.read_text(encoding="utf-8"))
            tuned_best = read_best_score_from_cv_results(cv_file)

            rows.append({
                "model": model_name,
                "tuned_best_f1_weighted": tuned_best,
                "best_params": json.dumps(best_params),
                "tuning_run_folder": str(run),
            })

    tuned = pd.DataFrame(rows)
    # Keep only the best tuning result per model (removes duplicates)
    if not tuned.empty:
        tuned = tuned.sort_values("tuned_best_f1_weighted", ascending=False)
        tuned = tuned.drop_duplicates(subset=["model"], keep="first")

    merged = baseline.merge(tuned, on="model", how="left")
    merged["improvement"] = merged["tuned_best_f1_weighted"] - merged["baseline_f1w_mean"]

    # sort: best tuned first, but keep untuned at bottom
    merged = merged.sort_values(["tuned_best_f1_weighted", "baseline_f1w_mean"], ascending=False)

    merged.to_csv(OUT_CSV, index=False)
    print("[OK] Saved:", OUT_CSV)
    print(merged)


if __name__ == "__main__":
    main()