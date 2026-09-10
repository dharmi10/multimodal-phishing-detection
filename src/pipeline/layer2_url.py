"""
Layer 2 — URL phishing classification.

Thin wrapper around the already-trained stage 2 model (src/stage2_url/). The
scoring logic there is untouched: text/lexical model at 0.7 weight, live HTML
red flags at 0.3, trusted domains short-circuited to 0.0.

Note this DOES make a real HTTP request per URL (5s timeout) as part of the
HTML check — that is stage 2's existing behaviour.
"""
import sys
import warnings
from pathlib import Path
from functools import lru_cache

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
    return stage2_predict.predict_url_score


def predict_url(url: str) -> dict:
    """Score a single URL."""
    try:
        score = float(_stage2()(url))
        return {"url": url, "phish_score": score, "error": None}
    except Exception as exc:
        return {"url": url, "phish_score": None, "error": f"{type(exc).__name__}: {exc}"}


def predict_urls(urls: list) -> dict:
    """
    Score every URL extracted from the message.

    A message is only as safe as its most dangerous link, so the aggregate
    handed to the fusion layer is the MAX score across URLs, not the mean.
    Returns url_phish_score = None when there was no URL to score, which is
    different from a score of 0.0 (checked, looks clean).
    """
    results = [predict_url(u) for u in urls]
    scored = [r["phish_score"] for r in results if r["phish_score"] is not None]

    if not scored:
        return {"results": results, "url_phish_score": None, "worst_url": None}

    worst = max(results, key=lambda r: r["phish_score"] if r["phish_score"] is not None else -1)
    return {
        "results": results,
        "url_phish_score": max(scored),
        "worst_url": worst["url"],
    }
