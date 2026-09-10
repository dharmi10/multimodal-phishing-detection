"""
Download Dataset 1 (Ad10sKun/phishing_detection) from Hugging Face
and save as raw CSV for later merging.
"""
from datasets import load_dataset, concatenate_datasets
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_dataset1():
    ds = load_dataset("Ad10sKun/phishing_detection")
    print(ds)

    # The source dataset ships with train/test splits (13,320 + 3,330 = 16,650).
    # The paper treats this as ONE 16,650-record dataset and performs its own
    # 80/20 split later on the full COMBINED dataset (Section IV.B.9).
    # So we combine both splits here to avoid double-splitting.
    full_ds = concatenate_datasets([ds["train"], ds["test"]])
    df = full_ds.to_pandas()

    out_path = RAW_DIR / "dataset1_huggingface.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

if __name__ == "__main__":
    download_dataset1()