"""
Merge Dataset 3 (Kaggle UCI SMS Spam) and Dataset 4 (Mendeley SMS Phishing)
into a single combined raw dataset.

NOTE: Dataset 1 (paper's Ad10sKun/phishing_detection HuggingFace source) and
Dataset 2 (paper only cites "Kaggle," no link given) were both excluded from
this reproduction:
- Dataset 2 could not be identified with confidence from the paper's vague
  description (no name/link provided).
- Dataset 1, once downloaded, was found via EDA (word cloud dominated by
  "ect", "hou", "enron", "kaminski") to actually be Enron corporate email
  data, not SMS text — inconsistent with the paper's SMS-based methodology.
  Multiple searches for genuine alternative smishing-specific SMS datasets
  did not turn up a suitable, verifiably distinct replacement (most
  candidates found were repackaged copies of the same UCI SMS Spam
  Collection already used as Dataset 3, or synthetic/LLM-generated data).

Label convention (standard practice, not specified by the paper beyond
"smish" vs "legitimate"): 1 = smish/malicious, 0 = legitimate.
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset3() -> pd.DataFrame:
    """Kaggle UCI SMS Spam Collection: columns v1 (label), v2 (text), 3 junk columns."""
    df = pd.read_csv(RAW_DIR / "dataset3_uci_spam.csv", encoding="latin-1")
    df = df[["v1", "v2"]].copy()
    df.columns = ["label", "text"]
    df["label"] = df["label"].str.strip().str.lower().map({"spam": 1, "ham": 0})
    df["source"] = "dataset3_uci"
    logger.info(f"Dataset 3 loaded: {len(df)} rows")
    return df

def load_dataset4() -> pd.DataFrame:
    """
    Mendeley SMS Phishing dataset (extended 17,160-row version): columns
    LABEL, TEXT, URL, EMAIL, PHONE. LABEL has 3 classes with inconsistent
    casing: ham, spam/Spam, Smishing/smishing. Per the paper's own Table I
    math (spam + smishing = combined "smish" class), we group spam AND
    smishing together as the malicious class (label=1).
    Uses latin-1 encoding - file contains non-UTF-8 characters (e.g. £).
    """
    df = pd.read_csv(RAW_DIR / "dataset4_mendeley.csv", encoding="latin-1")
    df = df[["LABEL", "TEXT"]].copy()
    df.columns = ["label", "text"]

    normalized = df["label"].str.strip().str.lower()
    df["label"] = normalized.map({"ham": 0, "spam": 1, "smishing": 1})
    df["source"] = "dataset4_mendeley"
    logger.info(f"Dataset 4 loaded: {len(df)} rows")
    return df


def load_dataset5() -> pd.DataFrame:
    """
    Kaggle: ahmadtijjani/phishing-urgency-authority-persuasion
    (phishing_dataset_with_category.csv). 1,000 rows but only 79 unique
    messages - the rest are repeats of the same ~79 templates with brand
    names swapped (e.g. "Urgent! Your Amazon has been compromised..." /
    "Urgent! Your Google has been compromised..."). This is templated/
    synthetic data, not organically collected real-world SMS. We
    deduplicate down to unique texts only, to avoid the model just
    memorizing repeated exact phrasings, and use it as a small
    supplementary malicious-class addition (all rows are phishing/label=1)
    to help offset our combined dataset's class imbalance.
    DEVIATION FROM PAPER: this source is not used in the original paper at
    all - it's our own addition to address the imbalance created by
    excluding the paper's original Dataset 1 (email-mislabeled) and
    Dataset 2 (unidentifiable). Should be documented clearly as such.
    """
    df = pd.read_csv(RAW_DIR / "dataset5_persuasion.csv", encoding="latin-1")
    df = df[["text", "label"]].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["text"])
    logger.info(f"Dataset 5: deduplicated from {before} to {len(df)} unique rows")

    df["label"] = 1  # all rows are phishing/malicious
    df["source"] = "dataset5_persuasion"
    logger.info(f"Dataset 5 loaded: {len(df)} rows")
    return df

def merge_all() -> pd.DataFrame:
    d3 = load_dataset3()
    d4 = load_dataset4()
    d5 = load_dataset5()

    combined = pd.concat([d3, d4, d5], ignore_index=True)

    logger.info(f"\n{'='*50}")
    logger.info(f"COMBINED DATASET SUMMARY (before cleaning)")
    logger.info(f"{'='*50}")
    logger.info(f"Total rows: {len(combined)}")
    logger.info(f"\nPer-source breakdown:")
    print(combined.groupby(["source", "label"]).size().unstack(fill_value=0))
    logger.info(f"\nOverall label distribution:")
    print(combined["label"].value_counts())

    out_path = PROCESSED_DIR / "combined_raw.csv"
    combined.to_csv(out_path, index=False)
    logger.info(f"\nSaved combined dataset to {out_path}")

    return combined

if __name__ == "__main__":
    merge_all()