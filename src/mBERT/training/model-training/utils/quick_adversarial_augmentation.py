"""
Quick adversarial data augmentation for testing improvements
"""

import pandas as pd
import random
from adversarial_data_generator import AdversarialDataGenerator

def create_quick_augmented_dataset():
    """Create a smaller augmented dataset for quick testing."""
    generator = AdversarialDataGenerator()

    # Load smaller sample of original data
    original_df = pd.read_csv('../dataset/sms_spam_phishing_dataset_v2.4_combined.csv', nrows=5000)

    print(f"Loaded {len(original_df)} original samples")

    # Augment with smaller factor for speed
    augmented_df = generator.augment_dataset(original_df, augmentation_factor=2)

    # Save the augmented dataset
    output_path = '../dataset/sms_spam_phishing_dataset_v2.4_quick_adversarial.csv'
    generator.save_augmented_dataset(augmented_df, output_path)

    print(f"Created augmented dataset with {len(augmented_df)} samples")
    return output_path

if __name__ == "__main__":
    create_quick_augmented_dataset()