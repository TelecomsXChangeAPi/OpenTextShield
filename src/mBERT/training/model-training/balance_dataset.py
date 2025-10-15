"""
Balance the augmented dataset by subsampling ham samples to improve class distribution.
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def balance_dataset(input_csv: str, output_csv: str, target_ham_samples: int = 50000):
    """Balance the dataset by reducing ham samples."""
    logger.info(f"Loading dataset from {input_csv}")
    df = pd.read_csv(input_csv)

    logger.info("Original class distribution:")
    print(df['label'].value_counts())

    # Separate by class
    ham_df = df[df['label'] == 'ham']
    spam_df = df[df['label'] == 'spam']
    phishing_df = df[df['label'] == 'phishing']

    logger.info(f"Ham samples: {len(ham_df)}")
    logger.info(f"Spam samples: {len(spam_df)}")
    logger.info(f"Phishing samples: {len(phishing_df)}")

    # Subsample ham
    if len(ham_df) > target_ham_samples:
        ham_df = ham_df.sample(n=target_ham_samples, random_state=42)
        logger.info(f"Subsampled ham to {target_ham_samples} samples")

    # Combine balanced dataset
    balanced_df = pd.concat([ham_df, spam_df, phishing_df], ignore_index=True)

    # Shuffle
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save with UTF-8 encoding
    balanced_df.to_csv(output_csv, index=False, encoding='utf-8')
    logger.info(f"Balanced dataset saved to {output_csv}")

    # Report final stats
    print("\nBalanced Dataset Statistics:")
    print(balanced_df['label'].value_counts())
    total = len(balanced_df)
    print(f"\nTotal samples: {total}")

    print("\nClass distribution:")
    for label, count in balanced_df['label'].value_counts().items():
        pct = count / total * 100
        print(f"{label}: {count} ({pct:.1f}%)")

    return balanced_df

if __name__ == "__main__":
    balance_dataset(
        "dataset/sms_spam_phishing_dataset_v2.3_augmented.csv",
        "dataset/sms_spam_phishing_dataset_v2.4_balanced.csv",
        target_ham_samples=50000
    )