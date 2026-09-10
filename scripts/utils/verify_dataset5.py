"""
Inspect Dataset 5 (Kaggle: ahmadtijjani/phishing-urgency-authority-persuasion)
to confirm column names, label values, and message style before merging.
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def verify_dataset5():
    path = RAW_DIR / "dataset5_persuasion.csv"

    # Try latin-1 first since every dataset so far has needed it at some point
    df = pd.read_csv(path, encoding="latin-1")

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 10 rows:")
    print(df.head(10))

    print(f"\nLabel value counts:")
    print(df["label"].value_counts(dropna=False))

    print(f"\nCategory value counts (if present):")
    if "category" in df.columns:
        print(df["category"].value_counts(dropna=False))

    # Check message length distribution to confirm SMS-style (short), not email-style (long)
    print(f"\nText length stats (in words):")
    print(df["text"].str.split().apply(len).describe())

if __name__ == "__main__":
    verify_dataset5()