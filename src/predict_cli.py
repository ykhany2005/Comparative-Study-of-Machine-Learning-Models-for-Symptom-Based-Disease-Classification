import sys

import joblib
import numpy as np

from common import FINAL_MODEL_PATH, clean_text


DISCLAIMER = (
    "\nDisclaimer: This tool is a student prototype for research/learning. "
    "It does not provide medical advice or a diagnosis. "
    "If you are worried or symptoms are severe, seek professional help."
)

EXAMPLES = [
    "fever sore throat headache",
    "itching skin rash redness",
    "stomach pain nausea vomiting",
    "cough chest pain shortness of breath",
]


def load_model():
    if not FINAL_MODEL_PATH.exists():
        print(f"[ERROR] Model file not found:\n  {FINAL_MODEL_PATH}")
        print("Tip: ensure you copied the tuned model to artifacts/final_model/best_model.joblib")
        sys.exit(1)
    return joblib.load(FINAL_MODEL_PATH)


def top_k_with_scores(model, text: str, k: int = 3):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        classes = model.classes_
        idx = np.argsort(probs)[::-1][:k]
        pred = classes[idx[0]]
        top = [(classes[i], float(probs[i])) for i in idx]
        return pred, top, "proba"

    if hasattr(model, "decision_function"):
        scores = np.array(model.decision_function([text])).ravel()
        classes = model.classes_
        idx = np.argsort(scores)[::-1][:k]
        pred = classes[idx[0]]
        top = [(classes[i], float(scores[i])) for i in idx]
        return pred, top, "score"

    pred = model.predict([text])[0]
    return pred, None, None


def main():
    print("=== Symptom Checker (Prototype 3 — CLI) ===")
    print(f"[INFO] Loading model from:\n  {FINAL_MODEL_PATH}")

    model = load_model()

    print("\nExample inputs:")
    for ex in EXAMPLES:
        print(" -", ex)

    print("\nType symptoms and press Enter. Type 'quit' to exit.")

    while True:
        raw_text = input("\nEnter symptoms: ").strip().lstrip("-•* ").strip()

        if raw_text.lower() in {"quit", "exit", "q"}:
            print("Exiting.")
            break

        cleaned_text = clean_text(raw_text)

        if len(cleaned_text.split()) < 2:
            print("[WARN] Please enter a longer symptom description (at least a few words).")
            continue

        pred, top, mode = top_k_with_scores(model, cleaned_text, k=3)

        print("\nTop prediction:", pred)

        if top is None:
            print("(Top-3 not available for this model type.)")
        elif mode == "proba":
            print("\nTop 3 (probabilities):")
            for i, (label, prob) in enumerate(top, start=1):
                print(f"  {i}) {label}  (prob={prob:.3f})")
        else:
            print("Note: these are decision scores, not probabilities. Higher = more confident.")
            print("\nTop 3 (decision scores):")
            for i, (label, score) in enumerate(top, start=1):
                print(f"  {i}) {label}  (score={score:.3f})")

        print(DISCLAIMER)


if __name__ == "__main__":
    main()