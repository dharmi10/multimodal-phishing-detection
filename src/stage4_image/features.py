"""
Layer 4 - image-only feature extraction.

Everything here answers one question: what can be seen in the IMAGE that no
other layer can see? The decoded text and the decoded URLs are deliberately
absent from that list. They already have owners - layer 1 scores the OCR text,
layer 2 scores the QR URLs - and a signal that reaches the fusion layer twice
gets counted twice, which is the whole reason this split exists.

So the features below are about the image as an object, not its contents:
whether it carries a QR code at all, whether that code is a payment request
rather than a link, whether the picture is shaped like a phone screenshot, and
how densely packed with text it is. A legitimate business does not usually
deliver its message as a photograph of a message.

Every threshold and weight in this module is HAND-SET, not fitted. There is no
labelled image corpus in this project to fit them on, and inventing one would
be worse than admitting the numbers are provisional.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# The project's existing URL recogniser, reused rather than re-implemented so a
# QR payload and a message body are parsed by exactly the same rules - including
# the schemeless "bit.ly/xYz" case that stage 0 exists to catch.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.extract import extract_urls

# --- Degradation -----------------------------------------------------------
# PROVISIONAL. Calibrated on a SINGLE test image at three quality levels, where
# token count fell 34 -> 34 -> 13 as word recall fell 100% -> 100% -> 30%. 20
# sits in that gap. One image is not a calibration set: revisit this with a
# real sample of damaged images before trusting it in anything but a demo.
MIN_OCR_TOKENS = 20

# --- Screenshot shape ------------------------------------------------------
# Phone screens run roughly 16:9 (1.78) to 20:9 (2.22) tall. The band is opened
# slightly at both ends to tolerate cropping and status-bar trimming. A photo
# of a document or a wide banner falls outside it.
SCREENSHOT_MIN_ASPECT = 1.6
SCREENSHOT_MAX_ASPECT = 2.6

# --- Text density ----------------------------------------------------------
# Tokens per megapixel. A screenshot of an SMS thread is mostly text; a photo or
# a logo is not. Hand-set by eye, not measured.
DENSE_TEXT_TOKENS_PER_MEGAPIXEL = 40.0

# URI schemes that mean "move money", not "open a page". A QR code that asks a
# phone to open a payment collect request is a different and more aggressive
# act than one carrying a link, and it is invisible to every other layer
# because it never reaches layer 2 as a URL.
PAYMENT_SCHEMES = ("upi", "gpay", "phonepe", "paytmmp", "tez", "bharatpe")

# Payment intents that pull money FROM the scanner, as opposed to displaying a
# payee. Both are reported; only the presence of a payment scheme is scored.
COLLECT_HINTS = ("collect", "mandate")


def is_payment_payload(payload: str) -> bool:
    """
    True when a QR payload is a payment/collect request rather than a link.

    Checked by URI scheme rather than by keyword, so "upi://pay?pa=..." matches
    but an ordinary page that merely mentions UPI in its path does not.
    """
    if not isinstance(payload, str):
        return False
    scheme = payload.split(":", 1)[0].strip().lower()
    return scheme in PAYMENT_SCHEMES


def is_collect_request(payload: str) -> bool:
    """True when a payment payload pulls money rather than just naming a payee."""
    if not is_payment_payload(payload):
        return False
    return any(hint in payload.lower() for hint in COLLECT_HINTS)


def classify_payloads(payloads: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Sort raw QR payloads into (urls, payment_payloads, other_payloads).

    A payment payload is NEVER put in the URL list even though it is
    syntactically a URI. Layer 2's model was trained on http(s) web addresses
    and its HTML red-flag half would try to fetch one, which is meaningless for
    a upi:// intent. Keeping them apart is also what stops the payment signal
    being counted once here and again in layer 2.

    Anything that is neither - WIFI:, BEGIN:VCARD, plain text - is preserved in
    the third list rather than dropped, because "this QR code contained
    something we do not handle" is information the report should still carry.
    """
    urls: List[str] = []
    payments: List[str] = []
    other: List[str] = []

    for payload in payloads:
        if not isinstance(payload, str) or not payload.strip():
            continue

        if is_payment_payload(payload):
            payments.append(payload)
            continue

        found = extract_urls(payload)
        if found:
            for url in found:
                if url not in urls:
                    urls.append(url)
        else:
            other.append(payload)

    return urls, payments, other


def aspect_ratio(width: int, height: int) -> float:
    """Tall-over-wide ratio. 0.0 when the dimensions are unusable."""
    if not width or not height or width <= 0 or height <= 0:
        return 0.0
    return float(height) / float(width)


def is_screenshot_shaped(width: int, height: int) -> bool:
    """True when the image is shaped like a portrait phone screen."""
    ratio = aspect_ratio(width, height)
    return SCREENSHOT_MIN_ASPECT <= ratio <= SCREENSHOT_MAX_ASPECT


def text_density(token_count: int, width: int, height: int) -> float:
    """
    OCR tokens per megapixel.

    Density rather than raw count, so a big screenshot and a small one holding
    the same message land in the same place. 0.0 when the image has no usable
    dimensions.
    """
    if not width or not height or width <= 0 or height <= 0:
        return 0.0
    megapixels = (float(width) * float(height)) / 1_000_000.0
    if megapixels <= 0:
        return 0.0
    return float(token_count) / megapixels


def is_text_dense(token_count: int, width: int, height: int) -> bool:
    """True when the image is packed with text rather than merely containing some."""
    return text_density(token_count, width, height) >= DENSE_TEXT_TOKENS_PER_MEGAPIXEL


def is_ocr_empty(token_count: int, ocr_confidence: float, ocr_attempted: bool) -> bool:
    """
    True when OCR ran and the image genuinely has no text in it.

    Zero tokens is not enough on its own, and that gap was a real hole: an image
    whose text had been completely destroyed by noise also produces zero tokens,
    and calling that "no text found" hid the damage behind neutral wording.

    `ocr_confidence` separates them. It is the mean over words Tesseract scored
    above zero, so a picture with nothing written on it yields no words at all
    and the mean is exactly 0.0. A destroyed picture still yields fragments -
    "|", "e", stray marks read as characters - which score moderately (30-70 in
    testing) but are one or two characters long and so survive no token filter.
    Positive confidence with nothing to show for it means text was there.

    Separate from degradation because the two mean different things to whoever
    reads the result. A QR code on a plain background has no text in it, and
    saying so is a neutral fact about the picture. Announcing it as damage is
    both wrong and, repeated often enough, the thing that teaches people to
    ignore the warning that does matter.

    The two are treated IDENTICALLY downstream - neither contributes text or
    URLs, and layer 1 still abstains when there is no body text either. Only
    the wording differs.
    """
    if not ocr_attempted:
        return False
    return token_count == 0 and ocr_confidence <= 0.0


def is_ocr_degraded(token_count: int, ocr_confidence: float, ocr_attempted: bool) -> bool:
    """
    True when OCR found text but too little of it to be trusted.

    Two ways in:
      0 < token_count < MIN_OCR_TOKENS  - some words survived, not enough
      token_count == 0 with confidence > 0 - fragments survived, no whole words

    The second is the case this function exists to catch. See is_ocr_empty for
    why confidence is the discriminator.

    `ocr_attempted` distinguishes "OCR ran and found almost nothing" from "OCR
    never ran" (Tesseract missing, decode error). Only the first is degradation;
    the second is an absent layer, which the caller reports differently.
    """
    if not ocr_attempted:
        return False
    if token_count == 0:
        return ocr_confidence > 0.0
    return token_count < MIN_OCR_TOKENS


def extract_image_features(decoded: Dict) -> Dict:
    """
    Turn one decode_image() result into the image-only feature set.

    decoded : the dict returned by src/stage4_image/decode.py

    The returned dict carries the classified payloads and the raw measurements
    behind each boolean, so the caller can explain a score rather than just
    quote it.
    """
    payloads = decoded.get("qr_payloads") or []
    width = int(decoded.get("width") or 0)
    height = int(decoded.get("height") or 0)
    token_count = int(decoded.get("ocr_token_count") or 0)
    confidence = float(decoded.get("ocr_confidence") or 0.0)

    urls, payments, other = classify_payloads(payloads)

    # OCR is only "attempted" if the image opened AND the OCR half did not fail.
    # decode_image folds an OCR failure into its combined error string; these two
    # substrings mirror the wording it uses there.
    error = decoded.get("error") or ""
    ocr_failed = "OCR unavailable" in error or "OCR failed" in error
    ocr_attempted = bool(width and height) and not ocr_failed

    return {
        "qr_urls": urls,
        "qr_payment_payloads": payments,
        "qr_other_payloads": other,
        "has_qr": bool(payloads),
        "qr_count": len(payloads),
        "has_payment_qr": bool(payments),
        "has_collect_request": any(is_collect_request(p) for p in payments),
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio(width, height),
        "is_screenshot_shaped": is_screenshot_shaped(width, height),
        "ocr_token_count": token_count,
        "ocr_confidence": confidence,
        "text_density": text_density(token_count, width, height),
        "is_text_dense": is_text_dense(token_count, width, height),
        "ocr_attempted": ocr_attempted,
        "ocr_empty": is_ocr_empty(token_count, confidence, ocr_attempted),
        "ocr_degraded": is_ocr_degraded(token_count, confidence, ocr_attempted),
    }
