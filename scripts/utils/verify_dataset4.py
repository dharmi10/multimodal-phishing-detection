"""
Inspect Dataset 4 (Mendeley SMS Phishing dataset) to confirm
column names and label values before merging in Step 3.
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

def verify_dataset4():
    path = RAW_DIR / "dataset4_mendeley.csv"
    df = pd.read_csv(path)

    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df.head())

    # Try the last column as label - adjust if this looks wrong
    label_col = "LABEL"
    print(f"\nValue counts for column '{label_col}':")
    print(df[label_col].value_counts())

if __name__ == "__main__":
    verify_dataset4()