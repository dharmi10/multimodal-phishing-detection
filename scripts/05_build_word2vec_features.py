"""
Word2Vec feature extraction for smishing detection (Approach 2, paper Section IV.C.2).

Mirrors the structure of tfidf_features.py:
    - loads train/test splits from data/processed
    - builds features from cleaned_text
    - saves vectorizer/model + transformed arrays for downstream ML training

Unlike TF-IDF (which is deterministic bag-of-words weighting), Word2Vec is a
trained neural embedding model. We train it FROM SCRATCH on our own training
corpus only (never touching test data), per the paper: "Word2Vec embeddings
were trained from scratch using the combined dataset to capture semantic
relationships between words" (Section IV.C.2).

Word2Vec produces one vector PER WORD, not per message. The paper does not
specify how word vectors are aggregated into a single per-message vector
(this is a real implementation gap, not something we're guessing wrong on).
We use TF-IDF-weighted averaging of word vectors as the message representation,
documented and justified below.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FEATURES_DIR = Path(__file__).resolve().parents[1] / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters — paper does not specify these (Section IV.C.2 only says
# "Word2Vec ... trained from scratch"). Documenting choices here, same as the
# SVD_COMPONENTS decision in tfidf_features.py.
#
# VECTOR_SIZE = 200: mid-range choice. 100 is common for small corpora, 300
#   is common for large corpora (e.g. Google News). Our corpus (~13.8k SMS-
#   length messages) is small-to-mid sized, so 200 balances representational
#   power against overfitting risk on a limited vocabulary.
# WINDOW = 5: SMS messages are short (often <30 tokens), so a window of 5
#   captures most of a message's context without exceeding message length.
# MIN_COUNT = 2: drop hapax legomena (words appearing exactly once) since
#   they contribute noisy, poorly-estimated vectors and inflate vocab size.
# SG = 1 (skip-gram): skip-gram typically outperforms CBOW on smaller
#   datasets and for rarer words, both relevant here (smishing keywords like
#   "OTP", "verify", "suspended" are exactly the rarer, high-signal tokens
#   we most want good vectors for).
# EPOCHS = 20: more than gensim's default (5) since our corpus is small;
#   more passes let embeddings converge better on limited data.
# ---------------------------------------------------------------------------
VECTOR_SIZE = 200
WINDOW = 5
MIN_COUNT = 2
SG = 1
EPOCHS = 20
SEED = 42


def message_to_vector(tokens, w2v_model, word_weights, vector_size):
    """
    Convert a tokenized message into a single vector by TF-IDF-weighted
    averaging of its word vectors.

    Why weighted (not plain) averaging: plain averaging treats "the" and
    "verify" identically, diluting exactly the high-signal smishing keywords
    (see Fig. 2 word cloud: "account", "bank", "verify", "urgent" etc.) with
    common filler words. Weighting each word vector by its TF-IDF score
    before averaging keeps the message vector biased toward its most
    informative tokens, consistent with why the paper uses TF-IDF at all
    (Section IV.C.1: "high-weight TF-IDF is allocated to smishing-related
    words").

    Falls back to a zero vector if none of the message's tokens are in the
    Word2Vec vocabulary (can happen for very short/unusual test messages).
    """
    vectors = []
    weights = []
    for tok in tokens:
        if tok in w2v_model.wv:
            vectors.append(w2v_model.wv[tok])
            weights.append(word_weights.get(tok, 1.0))
    if not vectors:
        return np.zeros(vector_size)
    vectors = np.array(vectors)
    weights = np.array(weights)
    return np.average(vectors, axis=0, weights=weights)


def main():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    train_df["cleaned_text"] = train_df["cleaned_text"].fillna("")
    test_df["cleaned_text"] = test_df["cleaned_text"].fillna("")

    logger.info(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # Tokenize (simple whitespace split since cleaning/lemmatization already
    # happened upstream in preprocessing — cleaned_text is already normalized)
    train_tokens = train_df["cleaned_text"].str.split()
    test_tokens = test_df["cleaned_text"].str.split()

    # --- Train Word2Vec from scratch on TRAIN tokens only ---
    w2v_model = Word2Vec(
        sentences=train_tokens.tolist(),
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=SG,
        epochs=EPOCHS,
        seed=SEED,
        workers=4,
    )
    logger.info(
        f"Word2Vec trained: vocab size = {len(w2v_model.wv)}, "
        f"vector_size = {VECTOR_SIZE}"
    )

    # --- Fit TF-IDF purely to get per-word weights for aggregation ---
    # (Not the same TF-IDF matrix used in Approach 1 — this is only used
    # internally here as a word-importance lookup for weighted averaging.)
    tfidf_for_weights = TfidfVectorizer(min_df=MIN_COUNT)
    tfidf_for_weights.fit(train_df["cleaned_text"])
    word_weights = dict(
        zip(tfidf_for_weights.get_feature_names_out(), tfidf_for_weights.idf_)
    )

    # --- Build message-level vectors for train and test ---
    X_train_w2v = np.array(
        [
            message_to_vector(toks, w2v_model, word_weights, VECTOR_SIZE)
            for toks in train_tokens
        ]
    )
    X_test_w2v = np.array(
        [
            message_to_vector(toks, w2v_model, word_weights, VECTOR_SIZE)
            for toks in test_tokens
        ]
    )

    logger.info(
        f"Word2Vec feature matrix shape - train: {X_train_w2v.shape}, "
        f"test: {X_test_w2v.shape}"
    )

    # --- Save model + features (mirrors tfidf_features.py save pattern) ---
    w2v_model.save(str(FEATURES_DIR / "word2vec_model.model"))

    with open(FEATURES_DIR / "word2vec_word_weights.pkl", "wb") as f:
        pickle.dump(word_weights, f)

    np.save(FEATURES_DIR / "X_train_w2v.npy", X_train_w2v)
    np.save(FEATURES_DIR / "X_test_w2v.npy", X_test_w2v)

    logger.info("Saved Word2Vec model, word weights, and transformed feature arrays")


if __name__ == "__main__":
    main()