"""
Layer 4 - image decoding.

A DECODER, not a classifier. This module owns no model and makes no judgement
about whether an image is malicious: it pulls the two things out of an image
that the existing layers already know how to score, and hands them back raw.

    QR / barcode payloads  ->  layer 2 (they are URLs far more often than not)
    OCR text               ->  layer 1 (a smishing message rendered as a picture
                                        to slip past a text-only classifier)
    OCR text, line by line ->  layer 3 (the sender header sits alone on its own
                                        line above the message bubble, and is
                                        only recognisable by that position)

Deliberately NOT here: any confidence floor, any "is this text good enough to
use" rule, any URL extraction. ocr_confidence is reported so the CALLER can set
that threshold once, in one place, when this is wired into the pipeline.
Filtering here would hide the evidence that decision needs.

Nothing in this module raises. An image that cannot be opened, a Tesseract
binary that is not installed, a missing optional dependency - all of them come
back as a populated "error" string with safe defaults in every other field,
because layer 4 failing must never take the rest of the pipeline down with it.

Tesseract note: pytesseract is a wrapper around a separate Tesseract binary
that has to be installed independently. On Windows it is frequently not on
PATH, so the executable location is read from the TESSERACT_CMD environment
variable when that is set. No path is hardcoded - a machine-specific path in
source would break every other machine.
"""
from __future__ import annotations

import io
import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

# Every third-party import is optional at import time. This module will be
# loaded by the pipeline at startup, so an uninstalled decoder dependency must
# degrade to "layer 4 reports an error" rather than stopping the process from
# booting at all.
try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - depends on the local install
    Image = None  # type: ignore[assignment]
    _PILLOW_ERROR: Optional[str] = f"{type(exc).__name__}: {exc}"
else:
    _PILLOW_ERROR = None

try:
    from pyzbar import pyzbar
except Exception as exc:  # pragma: no cover - depends on the local install
    pyzbar = None  # type: ignore[assignment]
    _PYZBAR_ERROR: Optional[str] = f"{type(exc).__name__}: {exc}"
else:
    _PYZBAR_ERROR = None

try:
    import pytesseract
    from pytesseract import Output as TesseractOutput
except Exception as exc:  # pragma: no cover - depends on the local install
    pytesseract = None  # type: ignore[assignment]
    TesseractOutput = None  # type: ignore[assignment]
    _PYTESSERACT_ERROR: Optional[str] = f"{type(exc).__name__}: {exc}"
else:
    _PYTESSERACT_ERROR = None

# Read when OCR runs, not at import, so setting it in an already-running shell
# session takes effect. Needed when the Tesseract binary is not on PATH, which
# on Windows is the normal case.
TESSERACT_CMD_ENV = "TESSERACT_CMD"

# A "token" for degradation purposes: a run of 3+ alphanumeric characters.
# Runs of 1-2 characters are dropped because OCR noise on a damaged image
# produces them by the dozen ("l", "rn", "0"), which would mask exactly the
# collapse this count exists to detect.
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]{3,}")


def count_tokens(text: str) -> int:
    """
    Number of alphanumeric tokens of length 3 or more in `text`.

    This is the degradation signal, and it exists because OCR confidence is not
    one. Measured on a single image at three quality levels, mean confidence
    fell only 93 -> 85 -> 82 while word recall fell 100% -> 100% -> 30%:
    Tesseract reports confidence only for the words it actually found, so it
    stays high while the text silently disappears. Token count on those same
    images fell 34 -> 34 -> 13, tracking the real loss.

    Reported, never acted on here - the caller owns the threshold.
    """
    if not isinstance(text, str) or not text:
        return 0
    return len(TOKEN_PATTERN.findall(text))


def _empty_result() -> dict:
    """The shape every return value has, with nothing decoded."""
    return {
        "qr_payloads": [],
        "ocr_text": "",
        "ocr_lines": [],
        "ocr_confidence": 0.0,
        "ocr_token_count": 0,
        "width": 0,
        "height": 0,
        "error": None,
    }


def _load_image(image_bytes: bytes) -> Any:
    """
    Open raw bytes as an RGB Pillow image.

    RGB conversion is not cosmetic: pyzbar and Tesseract both behave
    unpredictably on palettised (P), alpha (RGBA) and 1-bit images, and a
    screenshot of an SMS is very often one of those.
    """
    image = Image.open(io.BytesIO(image_bytes))
    image.load()  # force the decode now, so a truncated file fails here not later
    return image.convert("RGB")


def _decode_qr(image: Any) -> List[str]:
    """
    Every QR / barcode symbol in the image, in the order pyzbar found them.

    All symbols are returned, not just the first: an image can carry a decoy
    alongside the real payload, and layer 2 aggregates across links with MAX,
    so it has to see all of them to pick the worst.

    Payloads arrive as bytes and are decoded as UTF-8 with replacement rather
    than discarded - a partly undecodable payload is still evidence, and
    dropping it would be a filtering decision this module does not make.
    """
    payloads: List[str] = []
    for symbol in pyzbar.decode(image):
        data = getattr(symbol, "data", b"")
        text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else str(data)
        if text:
            payloads.append(text)
    return payloads


def _configure_tesseract() -> None:
    """Point pytesseract at an explicit binary when TESSERACT_CMD is set."""
    command = os.environ.get(TESSERACT_CMD_ENV, "").strip()
    if command:
        pytesseract.pytesseract.tesseract_cmd = command


# Tesseract numbers every word by the page / block / paragraph / line it sits
# on. Those four together are a line's identity, and they are the only way back
# from the flat word table to the lines a person actually sees in the picture.
_LINE_FIELDS = ("page_num", "block_num", "par_num", "line_num")


def _line_keys(data: Dict, count: int) -> List[Any]:
    """
    A per-word line identity for each of the `count` rows in Tesseract's table.

    A pytesseract build that does not report the structural columns degrades to
    one single line, which is exactly what ocr_text already was - never an
    error, just no line structure to offer.
    """
    columns = [data.get(field) for field in _LINE_FIELDS]
    columns = [c for c in columns if isinstance(c, (list, tuple)) and len(c) >= count]
    if not columns:
        return [0] * count
    return [tuple(column[index] for column in columns) for index in range(count)]


def _ocr(image: Any) -> Tuple[str, float, List[str]]:
    """
    Run OCR and return (text, mean_word_confidence, lines).

    image_to_data rather than image_to_string, because the per-word confidence
    is the whole point: a blurry photo that OCRs into plausible-looking garbage
    and a clean screenshot both produce a string, and only the confidence tells
    the two apart.

    Tesseract's table has one row per page / block / paragraph / line as well as
    per word, and those structural rows carry conf = -1 with empty text. Skipping
    them is reading the format correctly, not applying a quality threshold: the
    text is joined from every row that actually has text, and the mean is taken
    over words with conf > 0. No word is dropped for scoring badly.

    `lines` is the same words regrouped into the visual lines they came from, so
    " ".join(lines) == ocr_text. It exists because one thing in a screenshot of
    an SMS is identified by WHERE it sits rather than by what it says: the sender
    header, alone on its own line above the message. Flattened into one string it
    becomes indistinguishable from a word of the body text.
    """
    _configure_tesseract()

    data = pytesseract.image_to_data(image, output_type=TesseractOutput.DICT)

    texts = data.get("text", [])
    confs = data.get("conf", [])
    line_keys = _line_keys(data, len(texts))

    words: List[str] = []
    confidences: List[float] = []
    lines: "OrderedDict[Any, List[str]]" = OrderedDict()

    for index, (raw_text, raw_conf) in enumerate(zip(texts, confs)):
        text = (raw_text or "").strip()
        if not text:
            continue
        words.append(text)
        lines.setdefault(line_keys[index], []).append(text)

        # conf comes back as str in some pytesseract versions, int/float in others.
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            continue
        if confidence > 0:
            confidences.append(confidence)

    ocr_text = " ".join(words)
    mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    ocr_lines = [" ".join(line) for line in lines.values()]
    return ocr_text, float(mean_confidence), ocr_lines


def decode_image(image_bytes: bytes) -> dict:
    """
    Extract QR payloads and OCR text from one image.

    image_bytes : the image exactly as received (MMS attachment, uploaded file)

    Returns:
    {
        "qr_payloads":  list[str],   # decoded QR/barcode contents, [] if none
        "ocr_text":     str,         # extracted text, "" if none/low confidence
        "ocr_lines":    list[str],   # the same text split back into visual lines
        "ocr_confidence": float,     # mean word confidence 0-100, 0.0 if no text
        "ocr_token_count": int,      # alphanumeric tokens of length >= 3 in ocr_text
        "width":        int,
        "height":       int,
        "error":        str | None   # reason if decoding failed, else None
    }

    Never raises. "error" is None only when the image opened and both decoders
    ran. When one of the two fails the other's results are still returned and
    "error" names what was lost, so a partial read is never mistaken for a clean
    one that happened to find nothing.
    """
    result = _empty_result()

    if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
        result["error"] = f"expected image bytes, got {type(image_bytes).__name__}"
        return result

    if not image_bytes:
        result["error"] = "empty image bytes"
        return result

    if Image is None:
        result["error"] = f"Pillow unavailable ({_PILLOW_ERROR})"
        return result

    try:
        image = _load_image(bytes(image_bytes))
    except Exception as exc:
        result["error"] = f"could not open image: {type(exc).__name__}: {exc}"
        return result

    result["width"], result["height"] = int(image.width), int(image.height)

    # The two decoders are independent: one failing must not cost us the other.
    problems: List[str] = []

    if pyzbar is None:
        problems.append(f"QR decoding unavailable (pyzbar: {_PYZBAR_ERROR})")
    else:
        try:
            result["qr_payloads"] = _decode_qr(image)
        except Exception as exc:
            problems.append(f"QR decoding failed: {type(exc).__name__}: {exc}")

    if pytesseract is None:
        problems.append(f"OCR unavailable (pytesseract: {_PYTESSERACT_ERROR})")
    else:
        try:
            (result["ocr_text"], result["ocr_confidence"],
             result["ocr_lines"]) = _ocr(image)
            result["ocr_token_count"] = count_tokens(result["ocr_text"])
        except Exception as exc:
            problems.append(f"OCR failed: {type(exc).__name__}: {exc}")

    result["error"] = "; ".join(problems) if problems else None
    return result
