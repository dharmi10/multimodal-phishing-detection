"""
Builds data/processed/fusion_features.csv - the per-row layer scores that
train_fusion.py consumes.

This was the one missing link in the chain: every other artifact in the
project has a script that produces it, but the fusion feature table did not,
so the fusion model could not be rebuilt from source. This closes that gap.

Run from project root:  python scripts/10_build_fusion_features.py

What each column is
-------------------
sender_id, label, y   straight from the custom dataset, filtered and ordered
                      EXACTLY as retrain_svm_v3.load_custom does - that shared
                      ordering is what lets the out-of-fold layer 1 scores be
                      joined back row-for-row.
n_urls                how many links extract.extract_urls finds in the message
                      body. Diagnostic only; train_fusion.py does not read it.
l1_score              OUT-OF-FOLD layer 1 (v3) score, read from
                      layer1_oof_scores.csv. Not recomputed here: these rows
                      are in v3's training set, so scoring them with the
                      deployed model would leak. retrain_svm_v3.py owns this.
l2_score              stage 2 URL score, max across the message's links.
                      Empty when the message has no link - which the fusion
                      trainer treats as "no signal", not as a safe 0.0.
l3                    NOT a column. train_fusion.py derives layer 3 itself,
                      out-of-fold, from sender_id + label, because an in-fold
                      sender lookup is correct on every row by construction.
l1_score_v2           the older in-fold v2 score, kept purely so the v2 -> v3
                      comparison in the report can still be reproduced.
                      Nothing trains on it.

The layer 2 cache
-----------------
Layer 2 makes a real HTTP request per URL (5s timeout), so a full build takes
a few minutes and - more importantly - is NOT reproducible: a phishing site
that was live when the dataset was collected is usually dead months later, and
a dead fetch scores differently from a live one. Scores are therefore cached
per URL in data/processed/l2_url_scores.csv, which is what makes a rebuild
deterministic. Delete that file, or pass --refresh-l2, to re-fetch.

    python scripts/10_build_fusion_features.py --seed-cache-from-existing

seeds the cache from the current fusion_features.csv instead of fetching. Use
it once when adopting this script on an existing checkout, so the cache holds
the exact scores the deployed fusion model was trained on rather than fresh
ones fetched from a web that has moved on since.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.extract import extract_urls
from src.pipeline.layer2_url import predict_urls

CUSTOM = PROJECT_ROOT / "data" / "custom-dataset_4600.csv.xls"
OOF_SCORES = PROJECT_ROOT / "data" / "processed" / "layer1_oof_scores.csv"
L2_CACHE = PROJECT_ROOT / "data" / "processed" / "l2_url_scores.csv"
OUT = PROJECT_ROOT / "data" / "processed" / "fusion_features.csv"
MODELS = PROJECT_ROOT / "saved_models"

COLUMNS = ["sender_id", "label", "y", "n_urls", "l1_score", "l2_score", "l1_score_v2"]


def load_custom() -> pd.DataFrame:
    """
    The custom dataset, filtered and ordered exactly as retrain_svm_v3 does.

    Kept byte-identical to that function on purpose: if the two ever drift,
    the out-of-fold layer 1 scores would silently attach to the wrong rows.
    """
    df = pd.read_csv(CUSTOM, encoding="latin-1")
    df = df.dropna(subset=["sender_id", "message_text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df.label.isin(["legit", "phishing"])].reset_index(drop=True)
    df["y"] = (df.label == "phishing").astype(int)
    return df


def attach_layer1(df: pd.DataFrame) -> pd.DataFrame:
    """Join the out-of-fold v3 scores, verifying the row alignment holds."""
    if not OOF_SCORES.exists():
        sys.exit(f"missing {OOF_SCORES} - run `python scripts/09_retrain_svm_v3.py` first")

    oof = pd.read_csv(OOF_SCORES)
    if len(oof) != len(df):
        sys.exit(
            f"row count mismatch: {len(df)} custom rows vs {len(oof)} out-of-fold "
            f"scores. Re-run scripts/09_retrain_svm_v3.py - the two must come from the same "
            f"filtered view of the custom dataset."
        )
    if not (oof["sender_id"].values == df["sender_id"].values).all():
        sys.exit(
            "sender_id order differs between the custom dataset and "
            f"{OOF_SCORES.name}. Re-run scripts/09_retrain_svm_v3.py."
        )

    df["l1_score"] = oof["l1_score_v3"].values
    return df


def load_l2_cache() -> dict:
    if not L2_CACHE.exists():
        return {}
    cache = pd.read_csv(L2_CACHE)
    return dict(zip(cache["url"], cache["phish_score"]))


def save_l2_cache(cache: dict) -> None:
    rows = [{"url": u, "phish_score": s} for u, s in sorted(cache.items())]
    pd.DataFrame(rows, columns=["url", "phish_score"]).to_csv(L2_CACHE, index=False)


def seed_cache_from_existing() -> dict:
    """
    Rebuild the per-URL cache from the fusion table that is already on disk.

    Only sound because every row in this dataset carries at most one link, so
    the row's aggregate (a max across URLs) IS that one URL's score. Rows with
    more than one link are skipped rather than guessed at.
    """
    if not OUT.exists():
        sys.exit(f"--seed-cache-from-existing needs {OUT}, which does not exist")

    existing = pd.read_csv(OUT)
    custom = load_custom()
    if len(existing) != len(custom):
        sys.exit(f"{OUT.name} has {len(existing)} rows but the custom dataset has {len(custom)}")

    cache, skipped = {}, 0
    for message, score in zip(custom["message_text"], existing["l2_score"]):
        urls = extract_urls(message)
        if len(urls) == 1 and pd.notna(score):
            cache[urls[0]] = float(score)
        elif len(urls) > 1:
            skipped += 1

    save_l2_cache(cache)
    print(f"seeded layer 2 cache with {len(cache)} URLs from {OUT.name}"
          + (f" ({skipped} multi-link rows skipped)" if skipped else ""))
    return cache


def score_layer2(df: pd.DataFrame, refresh: bool) -> pd.DataFrame:
    """
    Score every message's links, reusing cached scores where available.

    Only the URLs missing from the cache are fetched, so a rebuild after
    adding a handful of dataset rows costs a handful of requests, not 177.
    """
    df["urls"] = df["message_text"].apply(extract_urls)
    df["n_urls"] = df["urls"].apply(len)

    cache = {} if refresh else load_l2_cache()
    wanted = {u for urls in df["urls"] for u in urls}
    missing = sorted(wanted - cache.keys())

    print(f"layer 2: {len(wanted)} unique URLs, {len(wanted) - len(missing)} cached, "
          f"{len(missing)} to fetch")

    if missing:
        print("  (each fetch is a live HTTP request with a 5s timeout)")
        t0 = time.time()
        for i, url in enumerate(missing, 1):
            result = predict_urls([url])
            score = result["url_phish_score"]
            if score is None:
                failure = result["results"][0]["error"]
                print(f"  [{i}/{len(missing)}] FAILED {url}: {failure}")
            else:
                cache[url] = score
            if i % 25 == 0:
                print(f"  [{i}/{len(missing)}] {time.time() - t0:.0f}s elapsed", flush=True)
        save_l2_cache(cache)
        print(f"  done in {time.time() - t0:.0f}s -> {L2_CACHE.name}")

    # Aggregate per message: MAX across links, matching layer2_url.predict_urls.
    # A message whose links all failed to score keeps NaN, same as no link at
    # all - the fusion trainer must not read a failed fetch as "checked, clean".
    def aggregate(urls):
        scored = [cache[u] for u in urls if u in cache]
        return max(scored) if scored else None

    df["l2_score"] = df["urls"].apply(aggregate)
    return df


def score_layer1_v2(df: pd.DataFrame) -> pd.DataFrame:
    """The legacy in-fold v2 score, for the v2-vs-v3 comparison only."""
    import joblib

    model_path = MODELS / "TFIDF_SVM_v2.pkl"
    vec_path = MODELS / "tfidf_vectorizer_v2.pkl"
    if not (model_path.exists() and vec_path.exists()):
        print("v2 artifacts not found - leaving l1_score_v2 empty")
        df["l1_score_v2"] = None
        return df

    from src.preprocessing.clean_text import clean_pipeline

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    cleaned = df["message_text"].apply(clean_pipeline)
    proba = model.predict_proba(vectorizer.transform(cleaned))
    df["l1_score_v2"] = proba[:, list(model.classes_).index(1)]
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--refresh-l2", action="store_true",
                        help="ignore the URL cache and re-fetch every link")
    parser.add_argument("--seed-cache-from-existing", action="store_true",
                        help="build the URL cache from the fusion_features.csv already on disk")
    args = parser.parse_args()

    if args.seed_cache_from_existing:
        seed_cache_from_existing()

    df = load_custom()
    print(f"custom dataset: {len(df)} rows "
          f"({df.y.sum()} phishing / {(1 - df.y).sum()} legit)")

    df = attach_layer1(df)
    df = score_layer2(df, refresh=args.refresh_l2)
    df = score_layer1_v2(df)

    out = df[COLUMNS]
    out.to_csv(OUT, index=False)

    print("\nsignal availability:")
    for column, name in (("l1_score", "l1"), ("l2_score", "l2")):
        n = out[column].notna().sum()
        print(f"  {name}: {n}/{len(out)} present ({n / len(out):.0%})")
    print(f"\nSaved -> {OUT}")
    print("Next: python scripts/11_train_fusion.py")


if __name__ == "__main__":
    main()
