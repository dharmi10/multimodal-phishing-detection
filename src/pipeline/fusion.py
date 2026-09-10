"""
Fusion layer.

Combines the three layer scores into one verdict using the logistic
regression trained by scripts/11_train_fusion.py. Each layer contributes a score in
0..1 plus a flag saying whether it had anything to say at all, so a message
with no URL or an unseen sender is handled explicitly rather than being
scored as if those layers had returned "safe".

Output carries four things:
  score      - fused probability that the message is phishing
  verdict    - phishing / legitimate at the decision threshold
  confidence - how many of the three layers actually contributed
  reasons    - plain-English explanation, ranked by how much each layer
               moved the decision
"""
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FUSION_MODEL = PROJECT_ROOT / "saved_models" / "fusion_lr.pkl"

THRESHOLD = 0.5

CONFIDENCE_BY_COUNT = {3: "high", 2: "medium", 1: "low", 0: "none"}

LAYER_LABELS = {
    "l1": "SMS text model",
    "l2": "URL model",
    "l3": "sender ID lookup",
}


@lru_cache(maxsize=1)
def load_fusion_model():
    if not FUSION_MODEL.exists():
        raise FileNotFoundError(
            f"{FUSION_MODEL} not found - run `python scripts/11_train_fusion.py` first"
        )
    return joblib.load(FUSION_MODEL)


def _collect_scores(layer1: dict, layer2: dict, layer3: dict) -> dict:
    """Pull the one number each layer contributes; None means 'no signal'."""
    return {
        "l1": layer1.get("smish_probability"),
        "l2": layer2.get("url_phish_score"),
        "l3": layer3.get("sender_phish_score"),
    }


def _build_row(scores: dict, neutral: float, feature_names: list) -> np.ndarray:
    """
    Built from the feature list stored alongside the model, so the columns
    here always match the ones it was trained on.
    """
    l3 = scores["l3"]
    available = {
        "l1_score": neutral if scores["l1"] is None else float(scores["l1"]),
        "l1_present": 0.0 if scores["l1"] is None else 1.0,
        "l2_score": neutral if scores["l2"] is None else float(scores["l2"]),
        "l2_present": 0.0 if scores["l2"] is None else 1.0,
        "l3_known_legit": 1.0 if l3 == 0.0 else 0.0,
        "l3_known_phish": 1.0 if l3 == 1.0 else 0.0,
    }
    return np.array([[available[name] for name in feature_names]])


def _explain(scores: dict, model, neutral: float, feature_names: list,
             layer2: dict, layer3: dict) -> list:
    """
    Rank the layers by how far each pushed the log-odds away from neutral,
    then describe them in that order.
    """
    coefs = dict(zip(feature_names, model.coef_[0]))

    contributions = []
    for layer in ("l1", "l2", "l3"):
        value = scores[layer]
        if value is None:
            continue

        if layer == "l3":
            # Layer 3 is encoded as directional indicators, so its push is
            # whichever indicator fired - not a deviation from neutral.
            key = "l3_known_phish" if value == 1.0 else "l3_known_legit"
            push = coefs.get(key, 0.0)
        else:
            push = coefs.get(f"{layer}_score", 0.0) * (float(value) - neutral)

        contributions.append((abs(push), push, layer, float(value)))

    contributions.sort(reverse=True)

    reasons = []
    for _, push, layer, value in contributions:
        if push > 0.05:
            direction = "raises the risk"
        elif push < -0.05:
            direction = "lowers the risk"
        else:
            direction = "carries no learned weight (see training notes)"
        name = LAYER_LABELS[layer]

        if layer == "l3":
            status = layer3.get("status")
            detail = f"sender '{layer3.get('sender_id')}' is listed as {status} in the sender dataset"
        elif layer == "l2":
            worst = layer2.get("worst_url")
            detail = f"most suspicious link scored {value:.2f}" + (f" ({worst})" if worst else "")
        else:
            detail = f"message wording scored {value:.2f}"

        reasons.append(f"{name}: {detail} - {direction}")

    for layer in ("l1", "l2", "l3"):
        if scores[layer] is None:
            if layer == "l2":
                reasons.append(f"{LAYER_LABELS[layer]}: no link in the message, so it could not contribute")
            elif layer == "l3":
                note = layer3.get("note") or "no sender information"
                reasons.append(f"{LAYER_LABELS[layer]}: {note}, so it could not contribute")
            else:
                reasons.append(f"{LAYER_LABELS[layer]}: no usable text after cleaning, so it could not contribute")

    return reasons


def fuse(layer1: dict, layer2: dict, layer3: dict) -> dict:
    """Combine the three layers into the final verdict."""
    artifacts = load_fusion_model()
    model = artifacts["model"]
    neutral = artifacts["neutral"]

    feature_names = artifacts["feature_names"]

    scores = _collect_scores(layer1, layer2, layer3)
    row = _build_row(scores, neutral, feature_names)

    score = float(model.predict_proba(row)[0][1])
    n_present = sum(1 for v in scores.values() if v is not None)

    return {
        "score": score,
        "verdict": "phishing" if score >= THRESHOLD else "legitimate",
        "threshold": THRESHOLD,
        "confidence": CONFIDENCE_BY_COUNT[n_present],
        "layers_used": n_present,
        "inputs": scores,
        "reasons": _explain(scores, model, neutral, feature_names, layer2, layer3),
    }
