"""
Data cleaning pipeline matching the paper's Section IV.B methodology:
1. Missing value handling
2. Duplicate removal
3. URL removal
4. Phone number removal
5. Special character removal
6. Extra whitespace removal
7. Tokenization
8. Lemmatization
(Partial) stop word removal — paper explicitly says full stopword removal
is risky since words like 'verify', 'urgent', 'now', 'will' carry
smishing-relevant signal (Section IV.B.8).
"""
import pandas as pd
import re
import string
import logging
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

lemmatizer = WordNetLemmatizer()

# --- Decision point: which stopwords to KEEP despite normally being removed ---
# The paper explicitly names these examples as too important to remove:
# 'is', 'just', 'now', 'will', 'verify', 'urgent' (Section IV.B.8)
# Paper does NOT give a full list — this exact set below is MY RECOMMENDATION,
# extending the paper's named examples with other common smishing-signal words,
# based on standard smishing/phishing keyword literature. This is a judgment
# call you should document clearly as your own addition in the paper/report.
SMISHING_SIGNAL_WORDS = {
    "is", "just", "now", "will", "verify", "urgent",
    "click", "confirm", "account", "suspended", "免", "won",
    "free", "call", "text", "reply", "claim", "update",
}

STOP_WORDS = set(stopwords.words("english")) - SMISHING_SIGNAL_WORDS

URL_PATTERN = re.compile(r"http\S+|www\.\S+")
PHONE_PATTERN = re.compile(r"\b\d{10,}\b|\+?\d[\d\-\s]{7,}\d")


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def remove_phone_numbers(text: str) -> str:
    return PHONE_PATTERN.sub(" ", text)


def remove_special_characters(text: str) -> str:
    # Keep only letters, digits, and whitespace; drop punctuation/symbols
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)


def remove_extra_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize_and_lemmatize(text: str) -> str:
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def clean_pipeline(text: str) -> str:
    """Apply all cleaning steps in the paper's exact order."""
    if not isinstance(text, str):
        return ""
    text = remove_urls(text)
    text = remove_phone_numbers(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)
    text = tokenize_and_lemmatize(text)
    return text


def clean_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path.name}")

    # 1. Missing value handling (paper 1.a): drop rows with missing text;
    #    fill missing label if text exists (paper doesn't say WITH what —
    #    my recommendation: drop these too, since an unknown label is
    #    unusable for supervised learning, safer than guessing a label)
    before = len(df)
    df = df.dropna(subset=["text"])
    df = df.dropna(subset=["label"])
    logger.info(f"After missing-value handling: {len(df)} rows (dropped {before - len(df)})")
    # 2. Duplicate removal (paper 1.b)
    # IMPORTANT DECISION: dedup PER SOURCE DATASET, not on the combined set.
    # Table II in the paper reports duplicate-drop counts per individual
    # dataset (before merging), and those drops are small (~1-6% per source).
    # Deduplicating on the globally combined set instead caused Dataset 4 to
    # lose ~82% of its rows, because it shares many verbatim ham messages
    # with Datasets 1/3 (common public SMS corpora reuse the same underlying
    # sources). Per-source dedup matches the paper's reported numbers far
    # more closely and avoids this cross-dataset collision artifact.
    before = len(df)
    df = df.groupby("source", group_keys=False).apply(
        lambda g: g.drop_duplicates(subset=["text"])
    )
    logger.info(f"After per-source duplicate removal: {len(df)} rows (dropped {before - len(df)})")

    # 3-8. Apply the full cleaning pipeline (URL, phone, special char, whitespace, tokenize, lemmatize)
    logger.info("Applying cleaning pipeline (URL/phone/special-char removal, tokenization, lemmatization)...")
    df["cleaned_text"] = df["text"].apply(clean_pipeline)

    # Drop rows that became empty after cleaning (e.g., a message that was ONLY a URL)
    before = len(df)
    df = df[df["cleaned_text"].str.strip() != ""]
    logger.info(f"After removing empty-after-cleaning rows: {len(df)} rows (dropped {before - len(df)})")

    df.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned dataset to {output_path}")

    return df
