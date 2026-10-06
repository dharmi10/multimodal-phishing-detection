"""
Fusion layer.

Combines the three layer scores into one verdict using the logistic
regression trained by scripts/11_train_fusion.py. Each layer contributes a score in
0..1 plus a flag saying whether it had anything to say at all, so a message
with no URL or an unseen sender is handled explicitly rather than being
scored as if those layers had returned "safe".

Output carries six things:
  score       - fused probability that the message is phishing
  verdict     - phishing / legitimate at the decision threshold
  label       - what to SHOW for that verdict, which is not always the verdict
  confidence  - how many of the three layers actually contributed
  reasons     - one plain-English line per layer, ranked by how much it moved
                the decision
  explanation - the structured account of the whole verdict: what each layer
                measured, the model's weight on it, how far it moved the
                log-odds, what the score would have been without it, which
                rules were checked, and the arithmetic that adds up to the
                score. See "Structured explanation" at the bottom of this file.
                `reasons` is derived from it, so the two cannot drift apart.

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
import math
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

    explanation = build_explanation(
        layer1, layer2, layer3, scores, model, neutral, feature_names,
        score=score, verdict=verdict, label=label, label_reason=label_reason,
        model_label=model_label, forced=forced, forced_reason=forced_reason,
        n_present=n_present,
    )

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
        "reasons": explanation["reasons"],
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Structured explanation
# ---------------------------------------------------------------------------
# `reasons` says which layer mattered most. It does not say how the verdict was
# arrived at, and those are different questions - "the URL model raised the
# risk" leaves a reader with no way to check whether it raised it enough to
# matter, or what the other two layers did about it.
#
# What follows decomposes the score EXACTLY. For a logistic regression the
# log-odds are a plain sum:
#
#     logit = intercept + SUM over features of (coefficient * value)
#
# so if every layer's features are set to the values they take when that layer
# has nothing to say, the remaining logit is the model's no-evidence baseline,
# and each layer's contribution is what its own features add on top. Those
# contributions sum to the final logit with nothing left over - the breakdown is
# an identity, not an attribution heuristic, and it is worth saying so because
# most per-feature explanations of a model are the latter.
#
# The "silent" value for a score column is `neutral`, which is the mean the
# training script imputed for a missing score; for a presence flag or a
# directional indicator it is 0. That is precisely the row a message no layer
# could measure produces, and it scores 0.4502 - the number the module
# docstring above already calls the model's prior.

# Which feature columns belong to which layer. The partition must be complete:
# every column the model was trained on has to sit under exactly one layer, or
# the contributions would not add up to the final logit and the breakdown would
# be quietly wrong rather than loudly wrong.
LAYER_FEATURES = {
    "l1": ("l1_score", "l1_present"),
    "l2": ("l2_score", "l2_present"),
    "l3": ("l3_known_legit", "l3_known_phish"),
}

# A contribution smaller than this is reported as "no material effect" rather
# than given a direction. Below it the sign is real but the movement is not: the
# l1_present coefficient is -0.007, which is noise dressed as evidence.
NEGLIGIBLE_CONTRIBUTION = 0.05


def _sigmoid(value: float) -> float:
    """Logistic function, written out so the arithmetic below can be checked."""
    return 1.0 / (1.0 + math.exp(-value))


def _silent_values(neutral: float, feature_names: list) -> dict:
    """
    The feature row of a message no layer could measure.

    A score column takes `neutral` - the imputed mean, which is what the model
    was trained to receive for a missing score. Everything else is a flag or a
    directional indicator and takes 0: not fired.
    """
    return {
        name: (float(neutral) if name.endswith("_score") else 0.0)
        for name in feature_names
    }


def decompose(scores: dict, model, neutral: float, feature_names: list) -> dict:
    """
    Split the final log-odds into a baseline plus one term per layer.

    Returns:
    {
        "intercept":       float,
        "baseline_logit":  float,   # every layer silent
        "baseline_score":  float,   # sigmoid of the above = the model's prior
        "contributions":   {"l1": float, "l2": float, "l3": float},
        "logit":           float,   # baseline + sum(contributions)
        "score":           float,   # sigmoid(logit)
        "coefficients":    {feature: coefficient},
        "values":          {feature: value actually fed in},
    }

    baseline_logit + sum(contributions) is the model's own decision function on
    this row, exactly - see `check_identity` for the assertion that says so.
    """
    coefficients = dict(zip(feature_names, model.coef_[0]))
    intercept = float(model.intercept_[0])

    silent = _silent_values(neutral, feature_names)
    row = _build_row(scores, neutral, feature_names)[0]
    values = {name: float(value) for name, value in zip(feature_names, row)}

    baseline_logit = intercept + sum(
        coefficients[name] * silent[name] for name in feature_names
    )

    contributions = {}
    for layer, features in LAYER_FEATURES.items():
        contributions[layer] = sum(
            coefficients[name] * (values[name] - silent[name])
            for name in features if name in coefficients
        )

    logit = baseline_logit + sum(contributions.values())

    return {
        "intercept": intercept,
        "baseline_logit": baseline_logit,
        "baseline_score": _sigmoid(baseline_logit),
        "contributions": contributions,
        "logit": logit,
        "score": _sigmoid(logit),
        "coefficients": coefficients,
        "values": values,
    }


def check_identity(breakdown: dict, score: float, tolerance: float = 1e-9) -> bool:
    """
    True when the decomposition reproduces the model's own probability.

    Exposed so a test can assert the property the explanation rests on, rather
    than the explanation merely claiming it.
    """
    return abs(breakdown["score"] - score) <= tolerance


def _direction(contribution: float) -> str:
    """How to describe a contribution's sign, or its lack of magnitude."""
    if contribution > NEGLIGIBLE_CONTRIBUTION:
        return "raises the risk"
    if contribution < -NEGLIGIBLE_CONTRIBUTION:
        return "lowers the risk"
    return "no material effect"


def _layer1_finding(layer1: dict) -> tuple:
    """(headline, detail lines) for what layer 1 found."""
    probability = layer1.get("smish_probability")
    headline = f"message wording scored {probability:.2f}"
    if layer1.get("prediction"):
        headline += f", read as {layer1['prediction']}"

    details = []
    if layer1.get("confidence") is not None:
        details.append(f"model confidence {layer1['confidence']:.0%}")

    if layer1.get("ocr_used"):
        body = layer1.get("body_smish_probability")
        ocr = layer1.get("ocr_smish_probability")
        if body is None:
            details.append(
                "the message has no body text, so the score is from the text read "
                f"out of the image alone ({_fmt_probability(ocr)})"
            )
        elif ocr is None:
            details.append(
                "the image yielded no scoreable text, so the score is from the "
                f"message body alone ({_fmt_probability(body)})"
            )
        else:
            details.append(
                "scored twice and the higher reading kept: message body "
                f"{_fmt_probability(body)}, text read out of the image "
                f"{_fmt_probability(ocr)} -> {layer1.get('source') or 'body'}"
            )
    elif layer1.get("source") == "body" and layer1.get("ocr_used") is False:
        details.append("the image yielded no usable text, so only the body was scored")

    if layer1.get("note"):
        details.append(layer1["note"])
    return headline, details


def _layer2_finding(layer2: dict) -> tuple:
    """(headline, detail lines) for what layer 2 found."""
    score = layer2.get("url_phish_score")
    worst = layer2.get("worst_url")
    results = layer2.get("results") or []

    headline = f"worst of {len(results)} link(s) scored {score:.2f}"
    if worst:
        headline += f" - {worst}"

    details = []
    for result in results:
        if result.get("error"):
            details.append(f"{result['url']}: error - {result['error']}")
        elif result.get("abstained"):
            details.append(f"{result['url']}: abstained - {result.get('note')}")
        else:
            details.append(f"{result['url']}: {result['phish_score']:.3f}")
    if len(results) > 1:
        details.append("aggregated with max: a message is as risky as its worst link")
    return headline, details


def _layer3_finding(layer3: dict) -> tuple:
    """(headline, detail lines) for what layer 3 found."""
    header = layer3.get("header")
    shown = layer3.get("sender_id") or header
    headline = f"sender '{shown}'"
    if header and header != layer3.get("sender_id"):
        headline += f" (header {header})"
    headline += " is on the sender register"
    if layer3.get("entity_name"):
        headline += f", registered to {layer3['entity_name']}"

    details = []
    wrapping = []
    if layer3.get("prefix"):
        wrapping.append(f"operator prefix {layer3['prefix']}")
    if layer3.get("suffix"):
        wrapping.append(f"type suffix {layer3['suffix']}")
    if wrapping:
        details.append(f"normalised by stripping {' and '.join(wrapping)}")
    details.append(
        "a registered header is the only thing this layer scores; everything "
        "else it reports travels as a rule, not a number"
    )
    return headline, details


def _absence_reason(layer: str, layer3: dict) -> str:
    """Why a layer contributed nothing. One wording, used everywhere."""
    if layer == "l1":
        return "no usable text after cleaning, so it could not contribute"

    if layer == "l2":
        return "no link in the message, so it could not contribute"

    note = layer3.get("note") or "no sender information"
    if layer3.get("impersonation_tier") == TIER_HIGH or layer3.get("invalid_header"):
        # It contributed nothing to the SCORE - the rule is what acted - so
        # "could not contribute" would be actively wrong here.
        return f"{note}; this fed no score into the model, it triggered the rule above"
    if layer3.get("impersonation"):
        # MEDIUM: reported, but no rule fired and no score was fed in.
        return (f"{note}; reported for the reader, but it did not override the "
                "verdict and fed no score into the model")
    return f"{note}, so it could not contribute"


def _fmt_probability(value) -> str:
    """Format an optional probability, for explanation text."""
    return "n/a" if value is None else f"{value:.2f}"


def _rules(layer3: dict, forced: str, forced_reason: str, model_label: str) -> list:
    """
    The rules that sit beside the model, and what each one did on this message.

    Reported whether or not they fired. A rule that was checked and did not fire
    is part of the explanation - "nothing overrode this verdict" is a statement
    worth being able to make, and it cannot be made from a list that only ever
    contains what happened.
    """
    rules = []

    impersonation = bool(layer3.get("impersonation"))
    tier = layer3.get("impersonation_tier")
    if impersonation and tier == TIER_HIGH:
        effect = (f"FIRED - label forced to {LABEL_IMPERSONATION}; the model on its "
                  f"own said {model_label}")
    elif impersonation:
        effect = (f"checked, did not override - the near-miss of "
                  f"{layer3.get('resembles')} is a {tier} tier match, which is also "
                  "how a brand extends its own header range")
    else:
        effect = "checked, no match - the header does not resemble a protected brand"
    rules.append({
        "name": "sender impersonation",
        "fired": impersonation and tier == TIER_HIGH,
        "effect": effect,
    })

    invalid = bool(layer3.get("invalid_header"))
    rules.append({
        "name": "invalid sender shape",
        "fired": invalid,
        "effect": (f"FIRED - label forced to {LABEL_INVALID_SENDER}; the model on its "
                   f"own said {model_label}") if invalid
        else "checked, no match - the header has a shape registered senders do have",
    })

    return rules


def _summary(label: str, score: float, breakdown: dict, layers: list,
             n_present: int, forced: str) -> str:
    """One paragraph a person can read instead of the table."""
    moved = [entry for entry in layers if entry["contributed"]]

    if not moved:
        return (
            f"No layer could measure this message, so the score of {score:.3f} is "
            f"the model's no-evidence baseline of {breakdown['baseline_score']:.3f} "
            f"and not a reading of anything. That is why the label is {label} "
            "rather than a verdict."
        )

    ordered = sorted(moved, key=lambda e: abs(e["contribution"]), reverse=True)
    movements = ", then ".join(
        f"{entry['name']} ({entry['contribution']:+.2f})" for entry in ordered
    )

    agreement = ""
    if len(ordered) > 1:
        signs = {entry["contribution"] > 0 for entry in ordered}
        agreement = (" All of them pushed the same way."
                     if len(signs) == 1
                     else " They did not all push the same way, and the score is "
                          "where that disagreement settled.")

    comparison = "at or above" if score >= THRESHOLD else "below"
    sentence = (
        f"{n_present} of 3 layers contributed. From the model's no-evidence baseline "
        f"of {breakdown['baseline_score']:.3f} they moved the score to {score:.3f}. "
        f"Ranked by how far each shifted the log-odds: {movements}.{agreement} "
        f"That is {comparison} the {THRESHOLD} threshold."
    )

    if forced is not None:
        sentence += (f" The displayed label is {label}, which comes from a sender "
                     "rule rather than from this score.")
    else:
        sentence += f" Hence {label}."
    return sentence


def build_explanation(layer1: dict, layer2: dict, layer3: dict, scores: dict,
                      model, neutral: float, feature_names: list, score: float,
                      verdict: str, label: str, label_reason: str,
                      model_label: str, forced, forced_reason: str,
                      n_present: int) -> dict:
    """
    The full structured account of one verdict.

    Everything a reader needs to answer "why this label?" without reading the
    source: what each layer measured, what the model's weight on that
    measurement is, how far it moved the log-odds, what the score would have
    been without it, which rules were checked, and how the arithmetic adds up.

    `reasons` is derived from this rather than built separately, so the one-line
    summary and the detailed breakdown cannot drift apart.
    """
    breakdown = decompose(scores, model, neutral, feature_names)
    coefficients = breakdown["coefficients"]

    finders = {"l1": _layer1_finding, "l2": _layer2_finding, "l3": _layer3_finding}

    silent = _silent_values(neutral, feature_names)
    total_movement = sum(abs(value) for value in breakdown["contributions"].values())

    layers = []
    for layer in ("l1", "l2", "l3"):
        value = scores[layer]
        contribution = breakdown["contributions"][layer]
        contributed = value is not None

        if contributed:
            headline, details = finders[layer](
                {"l1": layer1, "l2": layer2, "l3": layer3}[layer]
            )
            absence = None
        else:
            headline, details, absence = None, [], _absence_reason(layer, layer3)

        # What the score would have been with this layer silent. Exact for a
        # logistic model, and the most direct answer to "did this layer actually
        # decide the verdict?" - if removing it moves the score across the
        # threshold, it did.
        without = _sigmoid(breakdown["logit"] - contribution)

        # The term-by-term arithmetic, so the breakdown can be shown the same way
        # for all three layers. Layer 3 has no score column - it is encoded as
        # two directional indicators - and without this its row in a
        # score x weight table would have to be left blank or made up.
        # A layer that contributed nothing gets no terms at all. Listing its
        # columns anyway would print a coefficient next to a layer that moved
        # the score by zero, which reads as a weight that was applied.
        terms = []
        for name in (LAYER_FEATURES[layer] if contributed else ()):
            if name not in coefficients:
                continue
            delta = breakdown["values"][name] - silent[name]
            terms.append({
                "feature": name,
                "value": breakdown["values"][name],
                "silent_value": silent[name],
                "coefficient": coefficients[name],
                "push": coefficients[name] * delta,
            })

        # The weight to quote in a one-line summary: the coefficient on whichever
        # of this layer's columns actually did the moving.
        dominant = max(terms, key=lambda t: abs(t["push"]), default=None)

        # The arithmetic written out in full, term by term. A compact
        # "measurement x weight" cell would be a lie wherever a layer owns more
        # than one column: layer 1's push is its score column PLUS its presence
        # flag, and the two do not multiply out to the number next to them.
        arithmetic_text = "; ".join(
            f"{term['feature']} {term['value']:.3f} "
            f"(silent {term['silent_value']:.3f}) x {term['coefficient']:+.3f} "
            f"= {term['push']:+.3f}"
            for term in terms
        )

        layers.append({
            "layer": layer,
            "name": LAYER_LABELS[layer],
            "contributed": contributed,
            "score": None if value is None else float(value),
            "weight": dominant["coefficient"] if dominant else None,
            "weighted_feature": dominant["feature"] if dominant else None,
            "weighted_value": dominant["value"] if dominant else None,
            "terms": terms,
            "arithmetic_text": arithmetic_text or None,
            "indicator": (
                ("l3_known_phish" if value == 1.0 else "l3_known_legit")
                if layer == "l3" and contributed else None
            ),
            "contribution": contribution,
            "direction": _direction(contribution) if contributed else "did not contribute",
            "share": (abs(contribution) / total_movement) if total_movement else 0.0,
            "score_without": without,
            "decisive": contributed and ((without >= THRESHOLD) != (score >= THRESHOLD)),
            "finding": headline,
            "details": details,
            "absence_reason": absence,
        })

    reasons = []
    for entry in sorted(
            (e for e in layers if e["contributed"]),
            key=lambda e: abs(e["contribution"]), reverse=True):
        reasons.append(
            f"{entry['name']}: {entry['finding']} - {entry['direction']} "
            f"({entry['contribution']:+.2f} log-odds)"
        )
    for entry in layers:
        if not entry["contributed"]:
            reasons.append(f"{entry['name']}: {entry['absence_reason']}")
    if forced is not None:
        reasons.insert(0, forced_reason)

    return {
        "label": label,
        "label_reason": label_reason,
        "verdict": verdict,
        "score": score,
        "threshold": THRESHOLD,
        "decision": (
            f"score {score:.3f} is "
            f"{'at or above' if score >= THRESHOLD else 'below'} the "
            f"{THRESHOLD} threshold, so the model's verdict is {verdict.upper()}"
        ),
        "baseline": {
            "score": breakdown["baseline_score"],
            "logit": breakdown["baseline_logit"],
            "what": "what the model returns when no layer has anything to say",
        },
        "layers": layers,
        "arithmetic": {
            "intercept": breakdown["intercept"],
            "baseline_logit": breakdown["baseline_logit"],
            "contributions": breakdown["contributions"],
            "total_contribution": sum(breakdown["contributions"].values()),
            "logit": breakdown["logit"],
            "score": breakdown["score"],
            "exact": check_identity(breakdown, score),
            "how": "logit = intercept + sum(coefficient * value); score = 1/(1+e^-logit)",
        },
        "rules": _rules(layer3, forced, forced_reason, model_label),
        "confidence": {
            "level": CONFIDENCE_BY_COUNT[n_present],
            "layers_used": n_present,
            "layers_total": 3,
            "basis": (
                "a missing layer is treated as no information, not as safe, so a "
                "verdict on fewer layers rests on less evidence"
                if n_present < 3
                else "all three layers measured something, which is the most evidence "
                     "this pipeline can have"
            ),
        },
        "reasons": reasons,
        "summary": _summary(label, score, breakdown, layers, n_present, forced),
    }
