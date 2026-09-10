"""
Step 2 - apply the cleaning pipeline to the merged raw dataset.

    python scripts/02_clean_dataset.py

    data/processed/combined_raw.csv  ->  data/processed/combined_cleaned.csv

The cleaning logic itself lives in src/preprocessing/clean_text.py rather
than here, because it is not a build step only: layer 1 calls the very same
clean_pipeline() on every incoming SMS at inference time. Keeping one
implementation is what guarantees a message is cleaned identically whether it
is being trained on or scored.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.clean_text import clean_dataset

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    input_path = PROCESSED_DIR / "combined_raw.csv"
    output_path = PROCESSED_DIR / "combined_cleaned.csv"

    if not input_path.exists():
        sys.exit(f"missing {input_path} - run `python scripts/01_merge_datasets.py` first")

    df = clean_dataset(input_path, output_path)

    print(f"\n{'='*50}")
    print("FINAL CLEANED DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"Total rows: {len(df)}")
    print("\nPer-source breakdown:")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0))
    print("\nOverall label distribution:")
    print(df["label"].value_counts())
    print("\nSample cleaned text (first 5):")
    print(df[["text", "cleaned_text"]].head())
    print("\nNext: python scripts/03_train_test_split.py")


if __name__ == "__main__":
    main()
