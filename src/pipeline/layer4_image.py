"""
Layer 4 - image decoding and image-only scoring.

Thin wrapper around src/stage4_image/, exactly as layer2_url.py wraps
src/stage2_url/: the scoring logic lives in the stage module, and this file is
what the pipeline imports. The stage module is loaded lazily so that a message
with no image costs nothing here and so an uninstalled decoder dependency
cannot stop the pipeline from booting.

Layer 4 is a DECODER first. Its real output is not image_score - it is the QR
URLs handed to layer 2 and the OCR text handed to layer 1. image_score covers
only what those two cannot see: that there is a QR code at all, that it is a
payment request, that the picture is shaped like a screenshot, that it is dense
with text.

image_score is NOT a fusion feature and must not become one. The fusion model
was trained on a dataset with no images in it, so an l4 coefficient would be
fitted against an all-zero column and would mean nothing. Layer 4 reaches the
verdict through the layer 1 and layer 2 channels, which the fusion already
weights correctly.

Caching
-------
Decoded results are cached on disk, keyed by the SHA-256 of the image bytes,
for the same reason layer 2's URL scores are cached in scripts/10: an image
fetched over the network is not reproducible. The host that served it may be
gone tomorrow, and a decode of a dead link's placeholder scores differently
from a decode of the original. The cache is best-effort - a directory that
cannot be written to costs a little speed, never a result.
"""
from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE4_DIR = PROJECT_ROOT / "src" / "stage4_image"

CACHE_PATH = PROJECT_ROOT / "data" / "processed" / "l4_image_cache.json"

# Bump when the decoder or the scoring rule changes in a way that makes older
# cached entries wrong. Entries under a different version are ignored rather
# than deleted, so rolling back does not throw away the old ones.
#
# 2: per-image results gained "ocr_empty". A version 1 entry does not carry it,
#    and a cache hit returns the stored dict verbatim, so without this bump an
#    older cached image would come back missing the key.
# 3: ocr_empty/ocr_degraded now split on ocr_confidence as well as token count,
#    so a version 2 entry for a destroyed image carries the old, wrong flags -
#    it would still be served as "no text found" rather than "degraded".
CACHE_VERSION = "3"


@lru_cache(maxsize=1)
def _stage4() -> Callable[[bytes], Dict]:
    """Load the stage 4 scorer once, lazily."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.stage4_image.predict import predict_image as stage4_predict_image

    return stage4_predict_image


@lru_cache(maxsize=1)
def _stage4_aggregate() -> Callable[[List[Dict]], Dict]:
    """Load the stage 4 aggregator once, lazily."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.stage4_image.predict import aggregate_images

    return aggregate_images


def image_key(image_bytes: bytes) -> str:
    """Content-addressed cache key: same bytes, same key, on any machine."""
    return hashlib.sha256(bytes(image_bytes)).hexdigest()


@lru_cache(maxsize=1)
def _load_cache() -> Dict[str, Dict]:
    """
    Read the on-disk cache once per process.

    A missing, unreadable or corrupt cache file is not an error - it just means
    nothing is cached yet. Layer 4 must never fail because of its own cache.
    """
    if not CACHE_PATH.exists():
        return {}
    try:
        raw = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    if not isinstance(raw, dict):
        return {}
    if raw.get("version") != CACHE_VERSION:
        return {}

    entries = raw.get("entries")
    return entries if isinstance(entries, dict) else {}


def _save_cache(cache: Dict[str, Dict]) -> None:
    """Write the cache back, best-effort."""
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": CACHE_VERSION, "entries": cache}
        CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError):
        # Unwritable directory, or a result that will not serialise. Neither is
        # worth failing an analysis over.
        pass


def predict_image(image_bytes: bytes, use_cache: bool = True) -> Dict:
    """
    Decode and score one image, reusing a cached decode when the same bytes
    have been seen before.

    Returns the stage 4 result dict. On an unexpected failure it returns the
    same shape with "error" populated, so the pipeline never has to guard this
    call.
    """
    try:
        cache = _load_cache() if use_cache else {}
        key = image_key(image_bytes) if use_cache else None

        if key is not None and key in cache:
            cached = dict(cache[key])
            cached["cached"] = True
            return cached

        result = _stage4()(image_bytes)
        result["cached"] = False

        # Only a clean decode is cached. An error is a fact about this machine -
        # Tesseract not installed yet, a transient read failure - not about the
        # bytes, and caching one would keep returning "pyzbar unavailable" long
        # after pyzbar was installed. The cache exists to hold results that
        # cannot be reproduced, not results that should not be.
        if key is not None and not result.get("error"):
            cache[key] = {k: v for k, v in result.items() if k != "cached"}
            _save_cache(cache)

        return result
    except Exception as exc:
        return {
            "qr_urls": [],
            "qr_payment_payloads": [],
            "qr_other_payloads": [],
            "ocr_text": "",
            "ocr_confidence": 0.0,
            "ocr_token_count": 0,
            "ocr_empty": False,
            "ocr_degraded": False,
            "image_score": 0.0,
            "width": 0,
            "height": 0,
            "notes": [],
            "cached": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def predict_images(images: Optional[List[bytes]], use_cache: bool = True) -> Dict:
    """
    Score every image attached to the message.

    Aggregates with MAX across images and unions their QR URLs, mirroring how
    layer 2 aggregates across links. Returns image_score = None when there was
    no image at all - different from 0.0, which means an image was decoded and
    looked ordinary.
    """
    if not images:
        return _stage4_aggregate()([]) if _stage4_available() else _empty_aggregate()

    results = [predict_image(image, use_cache=use_cache) for image in images]

    try:
        return _stage4_aggregate()(results)
    except Exception as exc:
        aggregate = _empty_aggregate()
        aggregate["results"] = results
        aggregate["notes"] = [f"layer 4 aggregation failed: {type(exc).__name__}: {exc}"]
        return aggregate


def _stage4_available() -> bool:
    """True when the stage 4 module can be imported at all."""
    try:
        _stage4_aggregate()
        return True
    except Exception:
        return False


def _empty_aggregate() -> Dict[str, Any]:
    """The 'no images' result, without needing stage 4 to be importable."""
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
