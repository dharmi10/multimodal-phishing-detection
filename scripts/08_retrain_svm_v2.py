"""
Retrains TF-IDF vectorizer + SVM on the augmented training set.
Saves as NEW files (doesn't overwrite your original artifacts, in case
you want to compare or roll back).

Run from project root: python scripts/08_retrain_svm_v2.py
"""
import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing.clean_text import clean_pipeline

TRAIN_AUG_PATH = PROJECT_ROOT / "data" / "processed" / "train_augmented.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

# --- Load data ---
train = pd.read_csv(TRAIN_AUG_PATH)
test = pd.read_csv(TEST_PATH)

print("Cleaning text...")
train["cleaned_text"] = train["text"].apply(clean_pipeline)
test["cleaned_text"] = test["text"].apply(clean_pipeline)

# --- Refit TF-IDF (must refit: new vocabulary from augmented examples) ---
print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=5000)  # adjust max_features to match your original if different
X_train = vectorizer.fit_transform(train["cleaned_text"])
X_test = vectorizer.transform(test["cleaned_text"])

y_train = train["label"]
y_test = test["label"]

# --- Retrain SVM with your original best params ---
print("Training SVM (C=10, kernel=rbf)...")
model = SVC(C=10, kernel="rbf", probability=True, random_state=42)
model.fit(X_train, y_train)

# --- Quick eval on held-out test set ---
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
preds = model.predict(X_test)
print(f"\nTest set performance (original test.csv, untouched):")
print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
print(f"Precision: {precision_score(y_test, preds):.4f}")
print(f"Recall:    {recall_score(y_test, preds):.4f}")
print(f"F1:        {f1_score(y_test, preds):.4f}")

# --- Save new artifacts ---
Path("saved_models").mkdir(exist_ok=True)
joblib.dump(model, "saved_models/TFIDF_SVM_v2.pkl")
joblib.dump(vectorizer, "saved_models/tfidf_vectorizer_v2.pkl")
print("\nSaved: saved_models/TFIDF_SVM_v2.pkl, saved_models/tfidf_vectorizer_v2.pkl")