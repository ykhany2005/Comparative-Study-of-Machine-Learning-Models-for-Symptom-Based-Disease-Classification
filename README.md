

# AI Symptom Checker (Final Year Project)

## Overview

This project is an AI-based symptom checker prototype that predicts a likely disease label from free-text symptom descriptions.

The main aim is to compare machine learning approaches for symptom-text classification and evaluate which model performs best on the Symptom2Disease dataset.

This project is a **research and benchmarking prototype**, not a clinical diagnostic system.

---

## Problem

People often describe symptoms in natural language, but this kind of input is messy, unstructured, and difficult to handle with fixed rule-based systems.

This project explores whether machine learning models can:

* learn patterns from symptom descriptions
* classify likely disease labels consistently
* be compared fairly using standard evaluation methods

---

## Objectives

The project was developed in stages:

1. build a baseline NLP + machine learning pipeline
2. compare multiple classical models fairly
3. tune the strongest models
4. save artifacts and outputs for reproducibility
5. add a user-facing prediction interface
6. compare the best classical model with a transformer-based model

---

## Dataset

* **Dataset:** Symptom2Disease
* **Location:** `data/Symptom2Disease.csv`
* **Size:** ~1,200 samples
* **Classes:** 24 disease labels

**Columns:**

* `text` → free-text symptom description
* `label` → disease class

---

## Method

### Text preprocessing

The text pipeline uses:

* lowercasing
* punctuation/symbol removal
* whitespace normalisation

### Feature extraction (classical models)

Classical models use **TF-IDF** with:

* `ngram_range = (1, 2)`
* `min_df = 2`

This captures both individual words and short symptom phrases while filtering out very rare terms.

### Classical ML evaluation

Classical models were evaluated using:

* **Stratified 5-Fold Cross-Validation**

Metrics:

* accuracy
* weighted precision
* weighted recall
* weighted F1

### Transformer evaluation

A transformer-based model (**DistilBERT**) was implemented for comparison.

Due to computational cost, it was evaluated using a **single stratified train/validation split** rather than full cross-validation.

---

## Project Structure

```text
fyp/
├─ data/
│  └─ Symptom2Disease.csv
│
├─ src/
│  ├─ common.py
│  ├─ check_env.py
│  ├─ run_experiments.py
│  ├─ tune_models.py
│  ├─ make_tuning_summary.py
│  ├─ make_model_comparison.py
│  ├─ predict_cli.py
│  ├─ final_evaluate.py
│  ├─ train_distilbert.py
│  ├─ streamlit_app.py
│  └─ archive/
│
├─ tests/
│  ├─ test_text_cleaning.py
│  ├─ test_dataset_loading.py
│  └─ test_model_file.py
│
├─ artifacts/
│  ├─ runs/
│  ├─ results/
│  ├─ final_model/
│  ├─ final_eval/
│  ├─ transformer/
│  └─ eda/
│
├─ README.md
├─ requirements.txt
└─ .venv/
```



## How to Run

### 1. Set up environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

### 2. Run baseline experiments

```bash
python src/run_experiments.py
```

---

### 3. Tune models

```bash
python src/tune_models.py
```

---

### 4. Generate results summary

```bash
python src/make_tuning_summary.py
```

(Optional)

```bash
python src/make_model_comparison.py
```

---

### 5. Run final evaluation

```bash
python src/final_evaluate.py
```

---

### 6. Train transformer model

```bash
python src/train_distilbert.py
```

---

### 7. Run predictions (command line)

```bash
python src/predict_cli.py
```

---

### 8. Run web interface

```bash
streamlit run src/streamlit_app.py
```

---

### 9. Run tests (optional)

```bash
pytest
```

---

### Recommended order

1. run_experiments.py
2. tune_models.py
3. final_evaluate.py
4. predict_cli.py or streamlit_app.py

