"""
Layer 1 — SMS text classification.

Wraps the trained TF-IDF + SVM (v3) artifacts. Nothing is retrained here;
this only loads the saved model and scores one message.

v3 adds transactional SMS to the legitimate class (see scripts/09_retrain_svm_v3.py).
The fusion layer is calibrated against v3 scores, so these must stay in step:
pointing this back at v2 without retraining fusion would mis-weight layer 1.

Two numbers come out and they mean different things:
  smish_probability -> P(class = smish). Directional, 0..1, this is the one
                       the fusion layer should consume.
  confidence        -> confidence in whichever class was predicted (>= 0.5),
                       matching what app.py already shows the user.
"""
import sys
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "saved_models"

sys.path.append(str(PROJECT_ROOT / "src" / "preprocessing"))
from clean_text import clean_pipeline

SMISH_LABEL = 1


@lru_cache(maxsize=1)
def load_artifacts():
    model = joblib.load(MODELS_DIR / "TFIDF_SVM_v3.pkl")
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer_v3.pkl")
    return model, vectorizer


def predict_sms(text: str) -> dict:
    """Score one SMS body with the layer 1 model."""
    model, vectorizer = load_artifacts()

    cleaned = clean_pipeline(text)

    # A message that was nothing but a URL / digits cleans down to nothing.
    # The text model has no signal to offer, so say so rather than guess —
    # layer 2 still has the URL to work with.
    if not cleaned.strip():
        return {
            "label": None,
            "prediction": "unknown",
            "smish_probability": None,
            "confidence": None,
            "decision_score": None,
            "cleaned_text": "",
            "note": "message was empty after cleaning; no text signal available",
        }

    vec = vectorizer.transform([cleaned])
    pred = int(model.predict(vec)[0])

    proba = model.predict_proba(vec)[0]
    smish_index = list(model.classes_).index(SMISH_LABEL)
    smish_probability = float(proba[smish_index])

    # Same confidence formula app.py uses, kept for continuity with the demo.
    decision_score = float(model.decision_function(vec)[0])
    confidence = float(1 / (1 + np.exp(-abs(decision_score))))

    return {
        "label": pred,
        "prediction": "smish" if pred == SMISH_LABEL else "legitimate",
        "smish_probability": smish_probability,
        "confidence": confidence,
        "decision_score": decision_score,
        "cleaned_text": cleaned,
        "note": None,
    }
