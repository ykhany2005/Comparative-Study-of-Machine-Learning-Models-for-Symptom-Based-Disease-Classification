import joblib
import numpy as np
import streamlit as st

from common import FINAL_MODEL_PATH, clean_text


DISCLAIMER = (
    "Disclaimer: This tool is a student prototype for research/learning. "
    "It does not provide medical advice or a diagnosis. "
    "If you are worried or symptoms are severe, seek professional help."
)

EXAMPLES = [
    "fever sore throat headache",
    "itching skin rash redness",
    "stomach pain nausea vomiting",
    "cough chest pain shortness of breath",
]

VAGUE_TERMS = {
    "pain", "cold", "sick", "ill", "bad", "headache", "fever", "cough"
}


@st.cache_resource
def load_model():
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {FINAL_MODEL_PATH}. "
            "Make sure artifacts/final_model/best_model.joblib exists."
        )
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


def is_vague_input(cleaned_text: str) -> bool:
    words = cleaned_text.split()
    if len(words) < 2:
        return True
    if len(words) <= 3 and all(w in VAGUE_TERMS for w in words):
        return True
    return False


st.set_page_config(page_title="AI Symptom Checker", page_icon="🩺")

if "symptom_text" not in st.session_state:
    st.session_state.symptom_text = ""

st.title("AI Symptom Checker")
st.write(
    "This final year project prototype predicts a likely disease label from free-text symptom descriptions "
    "using a trained machine learning model."
)

with st.expander("About this model"):
    st.write("**Deployed model:** tuned LinearSVC")
    st.write("**Task:** classify disease labels from free-text symptom descriptions")
    st.write("**Dataset:** Symptom2Disease")
    st.write("**Important:** this is a demonstration prototype, not a clinical system")

st.subheader("How to use")
st.write("Enter symptoms in natural language, then click **Predict**.")

st.write("**Quick examples:**")
cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if cols[i].button(f"Example {i+1}"):
        st.session_state.symptom_text = ex

try:
    model = load_model()
    st.success("Model loaded successfully.")
except Exception as e:
    st.error(str(e))
    st.stop()

user_text = st.text_area(
    "Enter symptoms",
    value=st.session_state.symptom_text,
    placeholder="Example: fever, cough, chest pain, shortness of breath",
    height=140,
)

predict_clicked = st.button("Predict")

if predict_clicked:
    cleaned_text = clean_text(user_text)

    st.subheader("Preprocessed input")
    st.code(cleaned_text if cleaned_text else "[empty after cleaning]")

    if len(cleaned_text.split()) < 2:
        st.warning("Please enter a longer symptom description with at least a few words.")
    else:
        if is_vague_input(cleaned_text):
            st.warning(
                "This input is quite short or vague, so the prediction may be less reliable. "
                "Try including more symptom detail."
            )

        pred, top, mode = top_k_with_scores(model, cleaned_text, k=3)

        st.subheader("Top prediction")
        st.success(pred)

        st.subheader("Top 3 results")
        if top is None:
            st.write("Top-3 results are not available for this model type.")
        elif mode == "proba":
            for i, (label, prob) in enumerate(top, start=1):
                st.write(f"{i}. {label} (prob={prob:.3f})")
        else:
            st.caption("These are decision scores, not probabilities. Higher means more confident.")
            for i, (label, score) in enumerate(top, start=1):
                st.write(f"{i}. {label} (score={score:.3f})")

        st.info(DISCLAIMER)

st.markdown("---")
st.caption(
    "Best deployed model: tuned LinearSVC. "
    "A DistilBERT transformer model was also tested separately as a modern comparison model."
)