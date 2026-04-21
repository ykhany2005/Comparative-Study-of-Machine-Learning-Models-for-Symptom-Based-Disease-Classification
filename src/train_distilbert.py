import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from common import PROJECT_ROOT, DATA_DIR, clean_text, find_dataset_csv


os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".mpl-cache").absolute()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "transformer"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
SEED = 42
TEST_SIZE = 0.2
MAX_LENGTH = 128
NUM_EPOCHS = 4
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8


def make_run_folder() -> Path:
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S") + "_distilbert"
    out = ARTIFACTS_DIR / run_id
    for sub in ["configs", "results", "plots", "model"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision_weighted": prec,
        "recall_weighted": rec,
        "f1_weighted": f1,
    }


def main():
    out_dir = make_run_folder()
    print("[INFO] Output folder:", out_dir)

    df, dataset_path = find_dataset_csv(DATA_DIR)
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df["label"] = df["label"].astype(str)

    labels_sorted = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels_sorted)}
    id2label = {i: label for label, i in label2id.items()}

    df["label_id"] = df["label"].map(label2id)

    train_df, val_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["label_id"],
    )

    print("[INFO] Train size:", len(train_df))
    print("[INFO] Validation size:", len(val_df))
    print("[INFO] Num classes:", len(labels_sorted))
    print("[INFO] Device:", "cuda" if torch.cuda.is_available() else "cpu")

    train_ds = Dataset.from_pandas(train_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
    val_ds = Dataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "label"}))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    train_ds = train_ds.map(tokenize_batch, batched=True)
    val_ds = val_ds.map(tokenize_batch, batched=True)

    keep_cols = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=keep_cols)
    val_ds.set_format(type="torch", columns=keep_cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels_sorted),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(out_dir / "model" / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("[INFO] Starting DistilBERT training...")
    trainer.train()

    metrics = trainer.evaluate()
    print("[INFO] Validation metrics:", metrics)

    preds_output = trainer.predict(val_ds)
    y_pred = np.argmax(preds_output.predictions, axis=1)
    y_true = np.array(val_df["label_id"])

    metrics_summary = {
        "model_name": MODEL_NAME,
        "num_classes": len(labels_sorted),
        "train_size": int(len(train_df)),
        "validation_size": int(len(val_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics_summary["precision_weighted"] = float(prec)
    metrics_summary["recall_weighted"] = float(rec)
    metrics_summary["f1_weighted"] = float(f1)

    metrics_path = out_dir / "results" / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    report_df = pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            target_names=labels_sorted,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    report_path = out_dir / "results" / "per_class_report.csv"
    report_df.to_csv(report_path)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels_sorted))))
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels_sorted,
    ).plot(ax=ax, xticks_rotation=90, values_format="d", colorbar=False)
    plt.title("Confusion Matrix — DistilBERT")
    plt.tight_layout()
    cm_path = out_dir / "plots" / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    label_map_path = out_dir / "results" / "label_mapping.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
            },
            f,
            indent=2,
        )

    config_path = out_dir / "configs" / "train_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_file": str(dataset_path),
                "model_name": MODEL_NAME,
                "seed": SEED,
                "test_size": TEST_SIZE,
                "max_length": MAX_LENGTH,
                "num_epochs": NUM_EPOCHS,
                "train_batch_size": TRAIN_BATCH_SIZE,
                "eval_batch_size": EVAL_BATCH_SIZE,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            f,
            indent=2,
        )

    final_model_dir = out_dir / "model" / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print("\n[OK] DistilBERT run complete.")
    print(" -", metrics_path)
    print(" -", report_path)
    print(" -", cm_path)
    print(" -", label_map_path)
    print(" -", config_path)
    print(" -", final_model_dir)


if __name__ == "__main__":
    main()