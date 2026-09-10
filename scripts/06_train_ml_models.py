"""
Trains Logistic Regression, Naive Bayes, SVM, Random Forest, XGBoost on
BOTH TF-IDF+SVD (Approach 1) and Word2Vec (Approach 2) features.
Uses class_weight='balanced' (not in paper - our fix for 83.6/16.4 imbalance).
GridSearchCV + 5-fold CV per paper Section IV.F/G.
"""
import numpy as np, joblib, logging, json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR = Path(__file__).resolve().parents[1] / "data" / "features"
MODELS_DIR = Path(__file__).resolve().parents[1] / "saved_models"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
MODELS_DIR.mkdir(exist_ok=True); RESULTS_DIR.mkdir(exist_ok=True)

# Small grids - paper doesn't specify exact grids, kept small given time constraints
GRIDS = {
    "LR": (LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
           {"C": [0.1, 1, 10]}),
    "NB": (GaussianNB(), {"var_smoothing": [1e-9, 1e-8]}),
    "SVM": (SVC(class_weight="balanced", probability=True, random_state=42),
            {"C": [1, 10], "kernel": ["rbf"]}),
    "RF": (RandomForestClassifier(class_weight="balanced", random_state=42),
           {"n_estimators": [100, 200], "max_depth": [None, 20]}),
    "XGB": (XGBClassifier(eval_metric="logloss", random_state=42),
            {"n_estimators": [100, 200], "max_depth": [4, 6]}),
}

def run_approach(name, X_train, X_test, y_train, y_test):
    results = {}
    # XGBoost needs scale_pos_weight for imbalance (no class_weight param)
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    GRIDS["XGB"][0].set_params(scale_pos_weight=neg / pos)

    for model_name, (estimator, grid) in GRIDS.items():
        logger.info(f"[{name}] Training {model_name}...")
        gs = GridSearchCV(estimator, grid, cv=5, scoring="f1", n_jobs=-1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        preds = best.predict(X_test)

        results[model_name] = {
            "best_params": gs.best_params_,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
        }
        joblib.dump(best, MODELS_DIR / f"{name}_{model_name}.pkl")
        logger.info(f"[{name}] {model_name}: {results[model_name]}")

    return results

def main():
    y_train = np.load(FEATURES_DIR / "y_train.npy")
    y_test = np.load(FEATURES_DIR / "y_test.npy")

    all_results = {}

    # --- Approach 1: TF-IDF + SVD ---
    X_train_tfidf = np.load(FEATURES_DIR / "X_train_tfidf_svd.npy")
    X_test_tfidf = np.load(FEATURES_DIR / "X_test_tfidf_svd.npy")
    all_results["TFIDF"] = run_approach("TFIDF", X_train_tfidf, X_test_tfidf, y_train, y_test)

    # --- Approach 2: Word2Vec ---
    # SVM/GaussianNB benefit from non-negative/scaled input; MinMaxScale for consistency
    X_train_w2v = np.load(FEATURES_DIR / "X_train_w2v.npy")
    X_test_w2v = np.load(FEATURES_DIR / "X_test_w2v.npy")
    scaler = MinMaxScaler()
    X_train_w2v_s = scaler.fit_transform(X_train_w2v)
    X_test_w2v_s = scaler.transform(X_test_w2v)
    joblib.dump(scaler, MODELS_DIR / "w2v_scaler.pkl")
    all_results["W2V"] = run_approach("W2V", X_train_w2v_s, X_test_w2v_s, y_train, y_test)

    with open(RESULTS_DIR / "ml_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    main()