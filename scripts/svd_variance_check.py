"""
Diagnostic: check how explained variance grows with number of SVD
components, to pick a data-driven value instead of guessing.
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
train_df["cleaned_text"] = train_df["cleaned_text"].fillna("")

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(train_df["cleaned_text"])

# Test a range of component counts
for n in [100, 300, 500, 800, 1000, 1500]:
    svd = TruncatedSVD(n_components=n, random_state=42)
    svd.fit(X_train_tfidf)
    print(f"n_components={n}: explained variance = {svd.explained_variance_ratio_.sum():.2%}")