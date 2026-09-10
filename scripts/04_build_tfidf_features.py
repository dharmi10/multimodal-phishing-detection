"""
TF-IDF vectorization + max_features feature selection + TruncatedSVD
dimensionality reduction, matching paper's Approach 1 pipeline
(Sections IV.C.1, IV.D, IV.E).

IMPORTANT: vectorizer/SVD are FIT ONLY on the training set, then used to
TRANSFORM the test set. Fitting on the full dataset (including test) would
leak information from test into training - a critical ML mistake.
"""
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "saved_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# --- Hyperparameters NOT specified in the paper - our recommendations ---
MAX_FEATURES = 5000
NGRAM_RANGE = (1, 2)
SVD_COMPONENTS = 1000
RANDOM_SEED = 42


def build_tfidf_features():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")

    # Guard against any NaN cleaned_text that could crash the vectorizer
    train_df["cleaned_text"] = train_df["cleaned_text"].fillna("")
    test_df["cleaned_text"] = test_df["cleaned_text"].fillna("")

    logger.info(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # --- Step 1: TF-IDF vectorization + feature selection (max_features) ---
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
    )

    X_train_tfidf = tfidf.fit_transform(train_df["cleaned_text"])
    X_test_tfidf = tfidf.transform(test_df["cleaned_text"])

    logger.info(f"TF-IDF matrix shape - train: {X_train_tfidf.shape}, test: {X_test_tfidf.shape}")

    # --- Step 2: Dimensionality reduction via TruncatedSVD ---
    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_SEED)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)

    explained_var = svd.explained_variance_ratio_.sum()
    logger.info(f"TruncatedSVD: {SVD_COMPONENTS} components explain {explained_var:.2%} of variance")
    logger.info(f"Reduced matrix shape - train: {X_train_svd.shape}, test: {X_test_svd.shape}")

    # --- Save everything needed for model training + future inference ---
    
    FEATURES_DIR = Path(__file__).resolve().parents[1] / "data" / "features"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# ...at the bottom, replace the save block with:
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(svd, MODELS_DIR / "tfidf_svd.pkl")
    np.save(FEATURES_DIR / "X_train_tfidf_svd.npy", X_train_svd)
    np.save(FEATURES_DIR / "X_test_tfidf_svd.npy", X_test_svd)
    np.save(FEATURES_DIR / "y_train.npy", train_df["label"].values)
    np.save(FEATURES_DIR / "y_test.npy", test_df["label"].values)

    logger.info("Saved TF-IDF vectorizer, SVD model, and transformed feature arrays")

    return X_train_svd, X_test_svd, train_df["label"].values, test_df["label"].values


if __name__ == "__main__":
    build_tfidf_features()