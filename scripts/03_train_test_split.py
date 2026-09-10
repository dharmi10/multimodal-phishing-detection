"""
Stratified 80/20 train/test split on the cleaned combined dataset,
matching the paper's Section IV.B.9 methodology (80% train / 20% test).

Uses stratification (not in the paper, but standard practice) to preserve
our dataset's 83.6%/16.4% class imbalance in both splits.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

RANDOM_SEED = 42  # fixed seed for reproducibility, per project-wide reproducibility standard


def split_data():
    df = pd.read_csv(PROCESSED_DIR / "combined_cleaned.csv")
    logger.info(f"Loaded {len(df)} rows for splitting")

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )

    logger.info(f"\nTrain set: {len(train_df)} rows")
    print(train_df["label"].value_counts(normalize=True))

    logger.info(f"\nTest set: {len(test_df)} rows")
    print(test_df["label"].value_counts(normalize=True))

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"\nSaved train set to {train_path}")
    logger.info(f"Saved test set to {test_path}")

    return train_df, test_df


if __name__ == "__main__":
    split_data()