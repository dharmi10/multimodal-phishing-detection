"""
Fusion layer.

Combines the three layer scores into one verdict using the logistic
regression trained by scripts/11_train_fusion.py. Each layer contributes a score in
0..1 plus a flag saying whether it had anything to say at all, so a message
with no URL or an unseen sender is handled explicitly rather than being
scored as if those layers had returned "safe".

Output carries five things:
  score      - fused probability that the message is phishing
  verdict    - phishing / legitimate at the decision threshold
  label      - what to SHOW for that verdict, which is not always the verdict
  confidence - how many of the three layers actually contributed
  reasons    - plain-English explanation, ranked by how much each layer
               moved the decision

verdict vs label
----------------
`verdict` is the raw threshold comparison and is left exactly as it was - the
model, the threshold and the score are untouched. `label` is the display layer
on top of it, and exists because a bare "LEGITIMATE" was being printed in two
situations where it was not something the pipeline had actually established:

  - no layer contributed at all, so the 0.4502 being compared to the threshold
    is the model's prior, not a measurement of this message
  - one layer contributed and the score landed a hair under the threshold
    (0.4948 off a URL that scored 0.60), where the difference between the two
    sides of the line is far smaller than the uncertainty of a single signal

In both cases "LEGITIMATE" reads as a clearance the evidence does not support.
The score is unchanged and still shown; only the word next to it changes.
"""
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np

# Constant only. Imported rather than re-spelled so the tier name cannot drift
# between the layer that assigns it and the rule here that acts on it.
# layer3_sender does not import this module, so there is no cycle.
from .layer3_sender import TIER_HIGH

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FUSION_MODEL = PROJECT_ROOT / "saved_models" / "fusion_lr.pkl"

THRESHOLD = 0.5

CONFIDENCE_BY_COUNT = {3: "high", 2: "medium", 1: "low", 0: "none"}

# --- Display labels --------------------------------------------------------
# How close to the threshold still counts as "too close to call". Applied to the
# score only when the evidence behind it is thin; a 0.49 backed by all three
# layers is a real finding and keeps its label.
UNCERTAIN_MARGIN = 0.05

# Below this many contributing layers, a near-threshold score is not called.
MIN_LAYERS_FOR_NEAR_THRESHOLD_CALL = 2

LABEL_INSUFFICIENT = "INSUFFICIENT EVIDENCE"
LABEL_UNCERTAIN = "UNCERTAIN"

# --- Sender rule overrides -------------------------------------------------
# Layer 3 can establish two things the model cannot be asked about, because the
# training data contains no examples of either: a header that impersonates a
# registered brand, and a header of a shape no registered sender has. Both are
# RULES. They override the label and nothing else - the score, the verdict and
# the feature row are untouched, and are still reported.
#
# They are not fusion features and must not become them. The l3_known_phish
# column was all-zero across the whole training set and carries a coefficient of
# ~0, so a phishing score routed through the model would be multiplied by
# nothing and move no verdict. With exactly one real phishing sender in the data
# there is nothing to retrain on either. This is the same shape of decision as
# layer 2's trusted-domain short-circuit: a narrow, stated rule sitting beside
# the model rather than inside it.
LABEL_IMPERSONATION = "PHISHING - SENDER IMPERSONATION"
LABEL_INVALID_SENDER = "PHISHING - INVALID SENDER"


def verdict_label(score: float, verdict: str, layers_used: int) -> tuple:
    """
    The label to display, and one line explaining it.

    Returns (label, reason). The reason is "" when the label is just the
    verdict, so callers can print it unconditionally without a stray dash.

    Order matters: zero layers is checked first, because a message nothing
    could be measured on is not "uncertain" between two readings - there is
    no reading at all.
    """
    if layers_used == 0:
        return LABEL_INSUFFICIENT, (
            "no layer was able to contribute, so this score is the model's prior "
            "rather than a measurement of this message"
        )

    if (abs(score - THRESHOLD) <= UNCERTAIN_MARGIN
            and layers_used < MIN_LAYERS_FOR_NEAR_THRESHOLD_CALL):
        return LABEL_UNCERTAIN, (
            f"score is within {UNCERTAIN_MARGIN} of the {THRESHOLD} threshold and only "
            f"{layers_used} layer contributed - too close to call on one signal"
        )

    return verdict.upper(), ""


def sender_override(layer3: dict) -> tuple:
    """
    The label a layer 3 rule forces, or (None, "") when no rule fired.

    Kept separate from verdict_label so the model-derived label is computed
    first and can still be shown alongside the override — the rule replaces what
    is displayed, it does not erase what the model said.
    """
    # Only the HIGH tier overrides. A MEDIUM match - one ordinary letter
    # substituted, inserted or deleted - is how a brand extends its own header
    # range (ICICIO and ICICIT alongside ICICIH), and overriding on it produced
    # 4 false positives in 47 held-out legitimate headers. HIGH produced 0.
    if layer3.get("impersonation") and layer3.get("impersonation_tier") == TIER_HIGH:
        brand = layer3.get("resembles")
        entity = layer3.get("resembles_entity") or "an unnamed registered entity"
        header = layer3.get("header") or layer3.get("sender_id")
        edit = layer3.get("impersonation_edit") or "single-character edit"
        return LABEL_IMPERSONATION, (
            f"RULE, not a model score: sender header {header} is not on the register "
            f"and differs from {brand}, which belongs to {entity}, by one {edit}. "
            "Treated as impersonation regardless of the fused score below."
        )

    if layer3.get("invalid_header"):
        header = layer3.get("header") or layer3.get("sender_id")
        return LABEL_INVALID_SENDER, (
            f"RULE, not a model score: sender header {header} has a shape no "
            "registered sender header has. Treated as invalid regardless of the "
            "fused score below."
        )

    return None, ""


LAYER_LABELS = {
    "l1": "SMS text model",
    "l2": "URL model",
    "l3": "sender ID register",
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
            header = layer3.get("header")
            entity = layer3.get("entity_name")
            shown = f"'{layer3.get('sender_id')}'"
            if header and header != layer3.get("sender_id"):
                shown += f" (header {header})"
            detail = f"sender {shown} is on the sender register"
            if entity:
                detail += f", registered to {entity}"
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
                high = layer3.get("impersonation_tier") == TIER_HIGH
                if high or layer3.get("invalid_header"):
                    # It contributed nothing to the SCORE - the rule above is what
                    # acted - so "could not contribute" would be actively wrong here.
                    reasons.append(
                        f"{LAYER_LABELS[layer]}: {note}; this fed no score into the "
                        "model, it triggered the rule above"
                    )
                elif layer3.get("impersonation"):
                    # MEDIUM: reported, but no rule fired and no score was fed in.
                    reasons.append(
                        f"{LAYER_LABELS[layer]}: {note}; reported for the reader, but "
                        "it did not override the verdict and fed no score into the model"
                    )
                else:
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

    verdict = "phishing" if score >= THRESHOLD else "legitimate"
    label, label_reason = verdict_label(score, verdict, n_present)

    # A layer 3 rule replaces the label. The score, the verdict and the feature
    # row are deliberately left exactly as the model produced them, so what the
    # model said remains visible next to what the rule decided.
    model_label = label
    forced, forced_reason = sender_override(layer3)
    if forced is not None:
        label, label_reason = forced, forced_reason

    reasons = _explain(scores, model, neutral, feature_names, layer2, layer3)
    if forced is not None:
        reasons.insert(0, forced_reason)

    return {
        "score": score,
        "verdict": verdict,
        "label": label,
        "label_reason": label_reason,
        "model_label": model_label,
        "sender_override": forced is not None,
        "threshold": THRESHOLD,
        "confidence": CONFIDENCE_BY_COUNT[n_present],
        "layers_used": n_present,
        "inputs": scores,
        "reasons": reasons,
    }
