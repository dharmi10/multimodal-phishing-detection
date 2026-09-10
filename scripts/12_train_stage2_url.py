"""
Step 12 — train the layer 2 URL model.

NOTE ON THE NUMBER: this runs *before* step 10, not after it.
10_build_fusion_features.py scores every dataset URL with the artifacts this
script produces, so stage 2 must already exist by then. It is numbered 12
only because steps 01-11 were established first and renumbering them would
invalidate every existing reference. Read the order as: 12 -> 01..09 -> 10 -> 11.

Provenance script. The deployed artifacts were trained externally under
scikit-learn 1.8.0; this venv runs 1.4.2, so re-running this will NOT
reproduce them byte for byte. It is kept so the model is rebuildable and its
training data is documented -- not because it needs to be run. Running it
overwrites the deployed layer 2 model.

Model: TF-IDF over URL tokens + 7 lexical features -> MultinomialNB.
"""
import sys
from pathlib import Path

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import hstack, csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "saved_models"

sys.path.insert(0, str(PROJECT_ROOT / "src" / "stage2_url"))
from features import tokenize_url, extract_lexical_features

# Same seven columns predict.py assembles at inference time, in the same
# order -- the model is positional, so these must not be reordered.
LEXICAL_COLS = [
    'has_ip_address', 'slash_count', 'has_port_number', 'has_at_symbol',
    'hyphen_count', 'url_length', 'domain_length',
]


def main():
    df1 = pd.read_csv(RAW_DIR / "stage2_phishing_urls.csv")
    df2 = pd.read_csv(RAW_DIR / "stage2_extra_good_urls.csv")
    df = pd.concat([df1, df2], ignore_index=True)

    df['tokens'] = df['URL'].apply(tokenize_url)
    df['text_joint'] = df['tokens'].apply(lambda tokens: ' '.join(tokens))

    cv = TfidfVectorizer()
    X_vec = cv.fit_transform(df['text_joint'])

    lexical_df = df['URL'].apply(extract_lexical_features).apply(pd.Series)
    X_lex_sparse = csr_matrix(lexical_df[LEXICAL_COLS].astype(float).values)

    X_combined = hstack([X_vec, X_lex_sparse])
    y = df['Label']

    model = MultinomialNB()
    model.fit(X_combined, y)

    joblib.dump(cv, MODELS_DIR / "stage2_feature_transformer.pkl")
    joblib.dump(model, MODELS_DIR / "stage2_url_model.pkl")

    print("Training complete. Rows used:", len(df))
    print("Class balance:")
    print(y.value_counts())


if __name__ == "__main__":
    main()
