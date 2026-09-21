"""
Layer 4 - image scoring.

Mirrors src/stage2_url/predict.py: features.py measures, this file turns the
measurements into one number and a handful of human-readable notes.

The number is a SHALLOW HAND-WEIGHTED RULE, not a model. Nothing is trained,
nothing is loaded from disk, and the weights below were chosen by judgement
rather than fitted, because this project has no labelled image corpus to fit
them on. They are honest about what they are.

What image_score is NOT built from
----------------------------------
The OCR text and the QR URLs are excluded on purpose, and this is the single
most important property of this module. Those two go to layer 1 and layer 2
respectively, which already know how to score them and whose outputs the fusion
layer already weights. Scoring them here as well would hand the fusion two
copies of one signal and let a single piece of evidence vote twice.

So image_score answers only: how suspicious is this thing AS AN IMAGE, setting
aside anything written in it or linked from it?
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .decode import decode_image
from .features import MIN_OCR_TOKENS, extract_image_features

# --- Hand-set weights, summed then clamped to 1.0 --------------------------
# PROVISIONAL, like everything else in this layer. The ordering is the part
# worth defending; the exact magnitudes are a starting point.
#
# A QR code in a message is the base signal: a legitimate SMS almost never needs
# one, because the recipient is already holding the device that would scan it.
W_QR_PRESENT = 0.35
# A payment QR is the strongest image-only signal available. A link asks you to
# visit somewhere; a upi:// collect request asks your phone to move money, and
# it is invisible to layer 2 because it never becomes an http URL.
W_PAYMENT_QR = 0.35
# Shaped like a phone screenshot: consistent with forwarding a picture of a
# message rather than sending the message.
W_SCREENSHOT_SHAPE = 0.15
# Densely packed with text: the image is carrying a message body, which is what
# a text-only classifier is being routed around.
W_TEXT_DENSE = 0.15


def _score_and_notes(features: Dict) -> Tuple[float, List[str]]:
    """Internal: accumulate the weighted signals into (score, notes)."""
    score = 0.0
    notes: List[str] = []

    if features["has_qr"]:
        score += W_QR_PRESENT
        count = features["qr_count"]
        notes.append(
            f"image contains {count} QR/barcode symbol{'s' if count != 1 else ''}"
        )

        if features["has_payment_qr"]:
            score += W_PAYMENT_QR
            if features["has_collect_request"]:
                notes.append(
                    "QR code is a payment COLLECT request - scanning it would "
                    "authorise money leaving the account, not open a page"
                )
            else:
                notes.append("QR code is a payment request, not a link")

    if features["is_screenshot_shaped"]:
        score += W_SCREENSHOT_SHAPE
        notes.append(
            f"portrait screenshot proportions ({features['width']}x{features['height']}, "
            f"ratio {features['aspect_ratio']:.2f})"
        )

    if features["is_text_dense"]:
        score += W_TEXT_DENSE
        notes.append(
            f"densely packed with text ({features['text_density']:.0f} tokens per megapixel) "
            "- the message body is inside the picture"
        )

    return min(1.0, score), notes


def score_features(features: Dict) -> Dict:
    """
    Score one already-extracted feature set.

    Split out from predict_image so the scoring rule can be exercised on
    synthetic features without needing a real image or the decoder's
    dependencies installed.
    """
    score, notes = _score_and_notes(features)

    if features["ocr_empty"]:
        # Neutral. Most images legitimately contain no text, and a QR code on a
        # white background is the commonest of them.
        notes.append("no text found in this image")
    elif features["ocr_degraded"]:
        if features["ocr_token_count"] == 0:
            # Fragments scored, no whole word survived: the text was there and
            # has been destroyed. "0 tokens, floor is 20" would understate that.
            notes.append(
                f"OCR recovered only unreadable fragments (no words of 3+ characters, "
                f"mean confidence {features['ocr_confidence']:.0f}) - this image's text "
                "appears destroyed, and is not being used"
            )
        else:
            notes.append(
                f"OCR text is degraded ({features['ocr_token_count']} tokens, floor is "
                f"{MIN_OCR_TOKENS}) - text from this image is not being used"
            )

    return {"image_score": float(score), "notes": notes}


def predict_image(image_bytes: bytes) -> Dict:
    """
    Decode and score a single image.

    image_bytes : the image as received (MMS attachment, uploaded file, or a
                  response body fetched by layer 2)

    Returns:
    {
        "qr_urls":          list[str],   # payloads that parse as web URLs -> layer 2
        "qr_payment_payloads": list[str],# upi:// and friends, NOT sent to layer 2
        "qr_other_payloads":list[str],   # WIFI:, vCard, plain text
        "ocr_text":         str,         # extracted text -> layer 1
        "ocr_confidence":   float,
        "ocr_token_count":  int,
        "ocr_empty":        bool,        # OCR ran and found no text at all
        "ocr_degraded":     bool,        # 1..MIN_OCR_TOKENS-1 tokens; too little to trust
                                         # (either flag means: do not use the text)
        "image_score":      float,       # 0-1, image-only signals
        "width":            int,
        "height":           int,
        "notes":            list[str],
        "error":            str | None,
    }

    Never raises - decode_image does not, and the scoring is pure arithmetic
    over its output.
    """
    decoded = decode_image(image_bytes)
    features = extract_image_features(decoded)
    scored = score_features(features)

    return {
        "qr_urls": features["qr_urls"],
        "qr_payment_payloads": features["qr_payment_payloads"],
        "qr_other_payloads": features["qr_other_payloads"],
        "ocr_text": decoded["ocr_text"],
        "ocr_confidence": decoded["ocr_confidence"],
        "ocr_token_count": decoded["ocr_token_count"],
        "ocr_empty": features["ocr_empty"],
        "ocr_degraded": features["ocr_degraded"],
        "image_score": scored["image_score"],
        "width": decoded["width"],
        "height": decoded["height"],
        "notes": scored["notes"],
        "error": decoded["error"],
    }


def aggregate_images(results: List[Dict]) -> Dict:
    """
    Combine per-image results into the one summary layer 4 hands the pipeline.

    Aggregation matches layer 2's rule - MAX across images, because a message is
    only as safe as its most dangerous attachment. QR URLs are unioned so layer 2
    sees every link; OCR text is concatenated so layer 1 sees the whole message
    however it was split across pictures.

    Text from a DEGRADED image is left out of the combined OCR text entirely.
    That is the point of the flag: partial text from a damaged image is exactly
    the input that produces a confident low score for the wrong reason. An EMPTY
    image contributes nothing either, for the obvious reason.

    The two flags aggregate differently because they are reported differently.
    `ocr_degraded` is true if ANY image was damaged - one damaged attachment is
    worth warning about even when another was fine. `ocr_empty` is true only
    when nothing was damaged and no image yielded any text at all, because "no
    text found" is a statement about the whole message and would be misleading
    if one of the pictures plainly did contain text.

    Returns image_score = None when there was no image at all, which is
    different from a score of 0.0 (an image was checked and looked ordinary).
    """
    if not results:
        return {
            "results": [],
            "image_score": None,
            "qr_urls": [],
            "ocr_text": "",
            "ocr_empty": False,
            "ocr_degraded": False,
            "notes": [],
            "images_scored": 0,
        }

    qr_urls: List[str] = []
    text_parts: List[str] = []
    notes: List[str] = []
    scores: List[float] = []
    any_degraded = False
    any_empty = False

    for index, result in enumerate(results, 1):
        for url in result["qr_urls"]:
            if url not in qr_urls:
                qr_urls.append(url)

        if result["ocr_degraded"]:
            any_degraded = True
        elif result.get("ocr_empty"):
            any_empty = True
        elif result["ocr_text"].strip():
            text_parts.append(result["ocr_text"].strip())

        # An image that never opened scores 0.0 by arithmetic, but reporting
        # that as the aggregate would turn "we could not look" into "we looked
        # and it was fine". Only images that actually decoded get a vote.
        if result["width"] and result["height"]:
            scores.append(float(result["image_score"]))

        prefix = f"image {index}: " if len(results) > 1 else ""
        notes.extend(prefix + note for note in result["notes"])
        if result["error"]:
            notes.append(f"{prefix}decoder reported: {result['error']}")

    return {
        "results": results,
        "image_score": max(scores) if scores else None,
        "qr_urls": qr_urls,
        "ocr_text": " ".join(text_parts),
        "ocr_empty": any_empty and not any_degraded and not text_parts,
        "ocr_degraded": any_degraded,
        "notes": notes,
        "images_scored": len(results),
    }
