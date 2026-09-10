"""
Retrains layer 1 with transactional SMS added to the legitimate class.

Why: the v2 training set's legit class is almost entirely casual personal
chat from the UK/US public corpora ("What happened in interview?"), while its
smish class is full of formal, urgent, promotional language. A real
transactional message - "Sent Rs.1079.12 From HDFC Bank A/C *5345" - matches
neither, and its formal register pushes it toward smish. That is the source
of every false positive the fusion evaluation showed (precision 0.650).

The custom dataset's 353 legit rows are exactly the missing register, so they
get folded into training.

Leakage: those same rows are the fusion evaluation set, so this script also
produces OUT-OF-FOLD layer 1 scores - for each fold the model is retrained on
train_augmented plus the OTHER folds only, then scores the held-out fold. The
fusion trainer consumes those, so no row is ever scored by a model that saw
it. The deployed v3 model is separately fitted on everything.

Originals are untouched: this writes *_v3.pkl alongside v2.

Run from project root:  python scripts/09_retrain_svm_v3.py
"""
import sys, time, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing.clean_text import clean_pipeline

TRAIN_AUG = PROJECT_ROOT / "data" / "processed" / "train_augmented.csv"
TEST = PROJECT_ROOT / "data" / "processed" / "test.csv"
CUSTOM = PROJECT_ROOT / "data" / "custom-dataset_4600.csv.xls"
OOF_OUT = PROJECT_ROOT / "data" / "processed" / "layer1_oof_scores.csv"
MODELS = PROJECT_ROOT / "saved_models"

SEED, N_FOLDS = 42, 5
SVM_PARAMS = dict(C=10, kernel="rbf", probability=True, random_state=SEED)


def load_custom():
    """The custom dataset, filtered and ordered exactly as the fusion features are."""
    df = pd.read_csv(CUSTOM, encoding="latin-1")
    df = df.dropna(subset=["sender_id", "message_text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df.label.isin(["legit", "phishing"])].reset_index(drop=True)
    df["y"] = (df.label == "phishing").astype(int)
    df["cleaned_text"] = df["message_text"].apply(clean_pipeline)
    return df


def fit_model(texts, labels):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    model = SVC(**SVM_PARAMS).fit(X, labels)
    return model, vectorizer


def smish_proba(model, vectorizer, texts):
    proba = model.predict_proba(vectorizer.transform(texts))
    return proba[:, list(model.classes_).index(1)]


def main():
    base = pd.read_csv(TRAIN_AUG)
    base_text, base_y = base["cleaned_text"].fillna(""), base["label"]
    custom = load_custom()
    print(f"base train: {len(base)}   custom: {len(custom)} "
          f"({custom.y.sum()} phishing / {(1-custom.y).sum()} legit)")

    # --- Out-of-fold scores for the custom rows ---
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.full(len(custom), np.nan)

    for k, (train_idx, test_idx) in enumerate(skf.split(custom, custom.y), 1):
        t0 = time.time()
        texts = pd.concat([base_text, custom.cleaned_text.iloc[train_idx]])
        labels = pd.concat([base_y, custom.y.iloc[train_idx]])
        model, vectorizer = fit_model(texts, labels)
        oof[test_idx] = smish_proba(model, vectorizer, custom.cleaned_text.iloc[test_idx])
        print(f"  fold {k}/{N_FOLDS} done in {time.time()-t0:.0f}s", flush=True)

    custom["l1_score_v3"] = oof
    custom[["sender_id", "label", "y", "l1_score_v3"]].to_csv(OOF_OUT, index=False)
    print(f"\nout-of-fold layer 1 scores -> {OOF_OUT}")

    print("\nlayer 1 on the custom rows (threshold 0.5), out-of-fold:")
    pred = (oof >= 0.5).astype(int)
    print(f"  Accuracy : {accuracy_score(custom.y, pred):.4f}")
    print(f"  Precision: {precision_score(custom.y, pred, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(custom.y, pred, zero_division=0):.4f}")
    print(f"  F1       : {f1_score(custom.y, pred, zero_division=0):.4f}")

    # --- Final deployable model: everything ---
    print("\nfitting final v3 on base + all custom rows...", flush=True)
    t0 = time.time()
    texts = pd.concat([base_text, custom.cleaned_text])
    labels = pd.concat([base_y, custom.y])
    model, vectorizer = fit_model(texts, labels)
    print(f"  done in {time.time()-t0:.0f}s")

    # --- Regression check on the original held-out test set ---
    test = pd.read_csv(TEST)
    test_clean = test["cleaned_text"].fillna("") if "cleaned_text" in test else test["text"].apply(clean_pipeline)
    preds = model.predict(vectorizer.transform(test_clean))
    print("\nv3 on the original untouched test.csv:")
    print(f"  Accuracy : {accuracy_score(test.label, preds):.4f}")
    print(f"  Precision: {precision_score(test.label, preds, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(test.label, preds, zero_division=0):.4f}")
    print(f"  F1       : {f1_score(test.label, preds, zero_division=0):.4f}")

    v2_model = joblib.load(MODELS / "TFIDF_SVM_v2.pkl")
    v2_vec = joblib.load(MODELS / "tfidf_vectorizer_v2.pkl")
    v2_preds = v2_model.predict(v2_vec.transform(test_clean))
    print("v2 on the same test.csv (for comparison):")
    print(f"  Accuracy : {accuracy_score(test.label, v2_preds):.4f}")
    print(f"  Precision: {precision_score(test.label, v2_preds, zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(test.label, v2_preds, zero_division=0):.4f}")
    print(f"  F1       : {f1_score(test.label, v2_preds, zero_division=0):.4f}")

    joblib.dump(model, MODELS / "TFIDF_SVM_v3.pkl")
    joblib.dump(vectorizer, MODELS / "tfidf_vectorizer_v3.pkl")
    print(f"\nSaved: {MODELS/'TFIDF_SVM_v3.pkl'}, {MODELS/'tfidf_vectorizer_v3.pkl'}")
    print("(v2 artifacts left untouched)")


if __name__ == "__main__":
    main()
