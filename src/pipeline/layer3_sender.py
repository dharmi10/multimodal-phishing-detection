"""
Layer 3 — sender ID reputation.

A strict lookup against the custom sender dataset. The sender ID is matched
against the table and gets back exactly what the dataset says about it:
legit, phishing, or unknown. No heuristics, no pattern matching, no
generalisation — a sender that is not in the table returns "unknown" with a
score of None, which the fusion layer must treat as "no information" rather
than as a safe 0.0.

Only the sender_id and label columns are read; message_text and url are
ignored by design (those layers have their own models).

Note the dataset file is named .xls but is actually a CSV (ISO-8859/latin-1,
CRLF), so it is read with read_csv, not read_excel.
"""
from pathlib import Path
from functools import lru_cache

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SENDER_DATASET = PROJECT_ROOT / "data" / "custom-dataset_4600.csv.xls"

# What each dataset label maps to on the 0..1 phishing scale.
LABEL_SCORES = {
    "phishing": 1.0,
    "legit": 0.0,
}


def normalize_sender(sender_id: str) -> str:
    """Whitespace- and case-insensitive matching; nothing else is altered."""
    if not isinstance(sender_id, str):
        return ""
    return sender_id.strip().upper()


@lru_cache(maxsize=1)
def load_sender_table() -> dict:
    """
    Build {normalized_sender_id: label} from the custom dataset.

    The file is still being filled in, so rows missing a sender_id or a label
    are skipped rather than treated as errors. A sender that appears twice
    under conflicting labels is dropped entirely — guessing which row is
    right would be worse than reporting "unknown".
    """
    df = pd.read_csv(SENDER_DATASET, encoding="latin-1", usecols=["sender_id", "label"])

    df = df.dropna(subset=["sender_id", "label"])
    df["sender_id"] = df["sender_id"].astype(str).map(normalize_sender)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    df = df[(df["sender_id"] != "") & df["label"].isin(LABEL_SCORES)]

    table = {}
    conflicting = set()
    for sender, label in zip(df["sender_id"], df["label"]):
        if sender in table and table[sender] != label:
            conflicting.add(sender)
        table[sender] = label

    for sender in conflicting:
        table.pop(sender, None)

    return table


def lookup_sender(sender_id: str) -> dict:
    """Look one sender ID up in the dataset."""
    table = load_sender_table()

    if not sender_id:
        return {
            "sender_id": None,
            "status": "missing",
            "sender_phish_score": None,
            "note": "no sender ID was supplied with the message",
        }

    key = normalize_sender(sender_id)
    label = table.get(key)

    if label is None:
        return {
            "sender_id": sender_id,
            "status": "unknown",
            "sender_phish_score": None,
            "note": "sender ID not present in the custom dataset",
        }

    return {
        "sender_id": sender_id,
        "status": label,
        "sender_phish_score": LABEL_SCORES[label],
        "note": None,
    }


def table_stats() -> dict:
    """Coverage summary of the lookup table, for sanity checks."""
    table = load_sender_table()
    counts = {}
    for label in table.values():
        counts[label] = counts.get(label, 0) + 1
    return {"unique_senders": len(table), "by_label": counts}
