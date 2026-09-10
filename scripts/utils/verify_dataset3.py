"""
Inspect Dataset 3 (Kaggle UCI SMS Spam Collection) to confirm
column names and label values before merging in Step 3.
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def verify_dataset3():
    path = RAW_DIR / "dataset3_uci_spam.csv"

    # This specific Kaggle file is known to use Latin-1 encoding, not UTF-8.
    # If encoding="utf-8" (pandas default) is used instead, this line will
    # throw a UnicodeDecodeError - that's expected, not a bug in our code.
    df = pd.read_csv(path, encoding="latin-1")

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df.head())

    # The UCI file's label column is typically named 'v1' (label) and 'v2' (text),
    # but let's confirm rather than assume.
    label_col = df.columns[0]
    print(f"\nValue counts for column '{label_col}':")
    print(df[label_col].value_counts())

if __name__ == "__main__":
    verify_dataset3()