"""
Layer 2 — URL phishing classification.

Thin wrapper around the already-trained stage 2 model (src/stage2_url/). The
scoring logic there is untouched: text/lexical model at 0.7 weight, live HTML
red flags at 0.3, trusted domains short-circuited to 0.0.

Note this DOES make a real HTTP request per URL (5s timeout) as part of the
HTML check — that is stage 2's existing behaviour.

Content types: that same request already receives a Content-Type header which
used to be discarded. A response that is an image is now handed to layer 4 to
decode rather than to the HTML parser, and layer 2 ABSTAINS on that URL. The
alternative — running four markup red-flag checks over a JPEG — returns false
for all four, which is indistinguishable from a page that was inspected and
found clean. An abstention says "not scored"; a 0.0 would say "safe".
"""
import sys
import warnings
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGE2_DIR = PROJECT_ROOT / "src" / "stage2_url"

if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))


def _fix_tfidf_across_sklearn_versions(vectorizer):
    """
    The stage 2 artifacts were pickled with scikit-learn 1.8.0; this venv runs
    1.4.2 (which is what the layer 1 artifacts were trained with). The two
    versions store the fitted IDF weights under different attribute names:
    1.8 keeps a plain `idf_` array, 1.4 expects the `_idf_diag` sparse diagonal
    that `idf_` is derived from. Result: transform() raises
    "NotFittedError: idf vector is not fitted" even though the vectorizer IS
    fitted and its vocabulary and IDF values are fully intact.

    Rebuilding _idf_diag from the stored idf_ restores exactly the matrix 1.4
    would have built itself, so the scores are unchanged — no retraining, no
    edit to the .pkl files. Remove this if both stages ever run on one
    scikit-learn version.
    """
    transformer = getattr(vectorizer, "_tfidf", None)
    if transformer is None or hasattr(transformer, "_idf_diag"):
        return

    idf = transformer.__dict__.get("idf_")
    if idf is None:
        return

    idf = np.asarray(idf, dtype=np.float64)
    n = idf.shape[0]
    transformer._idf_diag = sp.diags(idf, offsets=0, shape=(n, n), format="csr", dtype=np.float64)
    del transformer.__dict__["idf_"]


@lru_cache(maxsize=1)
def _stage2():
    # Imported lazily so that loading layer 1 alone doesn't pull in stage 2's
    # pickles, and so an SMS with no URL costs nothing here.
    with warnings.catch_warnings():
        # The version mismatch above makes sklearn warn on every unpickle.
        warnings.simplefilter("ignore")
        import predict as stage2_predict

    _fix_tfidf_across_sklearn_versions(stage2_predict.cv)
    return stage2_predict.predict_url_detail


def _predict_url_with_bytes(url: str) -> Tuple[Dict, Optional[bytes]]:
    """
    Score one URL, returning its result dict and any image bytes fetched.

    The bytes are kept OUT of the result dict deliberately: that dict is
    serialised into the pipeline's JSON output, and a few hundred KB of JPEG
    does not belong there.
    """
    try:
        detail = _stage2()(url)
        result = {
            "url": url,
            "phish_score": detail["phish_score"],
            "error": None,
            "abstained": detail["abstained"],
            "note": detail["abstain_reason"],
            "content_type": detail["content_type"],
        }
        return result, detail["image_bytes"]
    except Exception as exc:
        result = {
            "url": url,
            "phish_score": None,
            "error": f"{type(exc).__name__}: {exc}",
            "abstained": False,
            "note": None,
            "content_type": None,
        }
        return result, None


def predict_url(url: str) -> dict:
    """Score a single URL."""
    return _predict_url_with_bytes(url)[0]


def predict_urls(urls: list) -> dict:
    """
    Score every URL extracted from the message.

    A message is only as safe as its most dangerous link, so the aggregate
    handed to the fusion layer is the MAX score across URLs, not the mean.
    Returns url_phish_score = None when there was no URL to score, which is
    different from a score of 0.0 (checked, looks clean).

    A URL that abstained (its response was an image) contributes nothing to the
    max, exactly like one that failed to fetch. Any image bytes collected on the
    way are returned under "fetched_images" for layer 4; the caller is expected
    to remove that key before serialising the result.
    """
    results = []
    fetched_images: List[bytes] = []
    for url in urls:
        result, image_bytes = _predict_url_with_bytes(url)
        results.append(result)
        if image_bytes:
            fetched_images.append(image_bytes)

    aggregate = _aggregate(results)
    aggregate["fetched_images"] = fetched_images
    return aggregate


def _aggregate(results: List[Dict]) -> Dict:
    """Max across per-URL results, shared by predict_urls and merge_results."""
    scored = [r["phish_score"] for r in results if r["phish_score"] is not None]

    if not scored:
        return {"results": results, "url_phish_score": None, "worst_url": None}

    worst = max(results, key=lambda r: r["phish_score"] if r["phish_score"] is not None else -1)
    return {
        "results": results,
        "url_phish_score": max(scored),
        "worst_url": worst["url"],
    }


def merge_results(first: Dict, second: Dict) -> Dict:
    """
    Combine two layer 2 passes into one, re-running the same max aggregation.

    Needed because QR codes are only discovered after the first pass: a URL that
    served an image is decoded by layer 4, and any link inside that image has to
    be scored too. Merging is cheaper than re-scoring the first pass's URLs,
    which would mean fetching every one of them a second time.
    """
    combined = list(first["results"])
    seen = {r["url"] for r in combined}
    combined.extend(r for r in second["results"] if r["url"] not in seen)

    merged = _aggregate(combined)
    merged["fetched_images"] = list(first.get("fetched_images", [])) + list(
        second.get("fetched_images", [])
    )
    return merged
