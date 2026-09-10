"""
Streamlit demo app for panel presentation.
Loads the best-performing model (SVM + Word2Vec) and lets the user
classify a live SMS message as Smish or Legitimate.
"""
import streamlit as st
import numpy as np
import joblib
import pickle
import re
import sys
from pathlib import Path
from gensim.models import Word2Vec

sys.path.append(str(Path(__file__).resolve().parent / "src" / "preprocessing"))
from clean_text import clean_pipeline  # reuses our exact cleaning pipeline

MODELS_DIR = Path("saved_models")
FEATURES_DIR = Path("data/features")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "W2V_SVM.pkl")
    scaler = joblib.load(MODELS_DIR / "w2v_scaler.pkl")
    w2v_model = Word2Vec.load(str(FEATURES_DIR / "word2vec_model.model"))
    with open(FEATURES_DIR / "word2vec_word_weights.pkl", "rb") as f:
        word_weights = pickle.load(f)
    return model, scaler, w2v_model, word_weights

def message_to_vector(tokens, w2v_model, word_weights, vector_size=200):
    vectors, weights = [], []
    for tok in tokens:
        if tok in w2v_model.wv:
            vectors.append(w2v_model.wv[tok])
            weights.append(word_weights.get(tok, 1.0))
    if not vectors:
        return np.zeros(vector_size)
    return np.average(np.array(vectors), axis=0, weights=np.array(weights))

st.set_page_config(page_title="Smishing Detector", page_icon="🛡️")
st.title("🛡️ Smishing Detection System")

model, scaler, w2v_model, word_weights = load_artifacts()

message = st.text_area("Enter an SMS message to classify:", height=100,
                         placeholder="e.g. Congratulations! You've won a $500 gift card. Click here to claim now!")

if st.button("Analyze Message", type="primary"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        cleaned = clean_pipeline(message)
        tokens = cleaned.split()
        vec = message_to_vector(tokens, w2v_model, word_weights).reshape(1, -1)
        vec_scaled = scaler.transform(vec)

        pred = model.predict(vec_scaled)[0]
        prob = model.predict_proba(vec_scaled)[0][1]  # probability of class 1 (smish)

        st.divider()
        if pred == 1:
            st.error(f"🚨 **SMISHING DETECTED** — Confidence: {prob:.1%}")
        else:
            st.success(f"✅ **LEGITIMATE MESSAGE** — Confidence: {1-prob:.1%}")

        with st.expander("See cleaned text (what the model actually sees)"):
            st.code(cleaned)

st.divider()
st.caption("Model: SVM + Word2Vec | Test Accuracy: 99.91% | F1: 99.74%")