#!/usr/bin/env python3
"""
Dataset Cleaning Script for OpenTextShield SMS Spam/Phishing Dataset

This script cleans the SMS spam/phishing dataset by:
1. Removing corrupted entries with invalid labels
2. Fixing obvious mislabeling (e.g., verification codes labeled as phishing)
3. Removing entries with malformed text
4. Providing statistics on the cleaned dataset
"""

import re
from collections import Counter, defaultdict
import argparse
import csv

def load_dataset(file_path):
    """Load the dataset from CSV format."""
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Skip header
        next(f, None)

        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse CSV format: text,label
            # Handle quoted fields properly
            if line.startswith('"') and line.endswith('"'):
                # Simple case: entire line is quoted
                parts = line.rsplit(',', 1)
                if len(parts) == 2:
                    text = parts[0].strip('"')
                    label = parts[1].strip().lower()
                else:
                    continue
            else:
                # Handle cases where text might contain commas
                parts = line.rsplit(',', 1)
                if len(parts) == 2:
                    text = parts[0].strip()
                    label = parts[1].strip().lower()
                else:
                    continue

            # Only keep valid labels
            if label in ['ham', 'spam', 'phishing']:
                data.append({'text': text, 'label': label})

    return data

def clean_text(text):
    """Clean and normalize text."""
    if not isinstance(text, str):
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove common corrupted patterns
    text = re.sub(r'\\+$', '', text)  # Remove trailing backslashes
    text = re.sub(r'^\s*\d+\|', '', text)  # Remove line number prefixes that might remain

    return text

def is_valid_entry(text, label):
    """Check if an entry is valid."""
    if not text or len(text.strip()) < 3:
        return False

    # Skip entries that look like they're from other datasets or corrupted
    corrupted_patterns = [
        r'^\s*\d+\s*$',  # Just numbers
        r'^\s*\(\w+\)\s*$',  # Just parenthesized codes
        r'^\s*\|.*\|',  # Pipe-separated content that looks malformed
        r'voicespin.*account balance',  # Test/bogus account balance messages
        r'^\s*[\(\)\w]+\s*$',  # Just alphanumeric codes
    ]

    for pattern in corrupted_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False

    return True

def fix_mislabeling(text, label):
    """Fix obvious mislabeling based on content analysis."""
    text_lower = text.lower()

    # Verification codes should be ham, not phishing
    verification_patterns = [
        r'verification code',
        r'code is \d+',
        r'your code is',
        r'otp',
        r'one-time password',
        r'security code',
        r'authenticator',
        r'2fa',
        r'two-factor',
    ]

    for pattern in verification_patterns:
        if re.search(pattern, text_lower) and label == 'phishing':
            return 'ham'

    # Banking/transaction confirmations should be ham
    if 'transfer' in text_lower and 'security code' in text_lower and label == 'phishing':
        return 'ham'

    return label

def clean_dataset(data):
    """Clean the dataset."""
    print(f"Original dataset size: {len(data)}")

    # Clean text and validate entries
    cleaned_data = []
    seen_texts = set()

    for item in data:
        text = clean_text(item['text'])
        label = item['label']

        # Skip invalid entries
        if not is_valid_entry(text, label):
            continue

        # Fix mislabeling
        label = fix_mislabeling(text, label)

        # Skip duplicates
        if text in seen_texts:
            continue

        seen_texts.add(text)
        cleaned_data.append({'text': text, 'label': label})

    print(f"After cleaning: {len(cleaned_data)}")
    return cleaned_data

def print_statistics(data):
    """Print dataset statistics."""
    print("\n=== DATASET STATISTICS ===")
    print(f"Total samples: {len(data)}")

    label_counts = Counter(item['label'] for item in data)
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(data)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    ham_count = label_counts.get('ham', 0)
    spam_count = label_counts.get('spam', 0)
    phishing_count = label_counts.get('phishing', 0)

    if spam_count > 0:
        print(f"\nClass imbalance ratio (ham/spam): {ham_count / spam_count:.2f}")
    if phishing_count > 0:
        print(f"Class imbalance ratio (ham/phishing): {ham_count / phishing_count:.2f}")

    # Text length statistics
    text_lengths = [len(item['text']) for item in data]
    if text_lengths:
        print("\nText length statistics:")
        print(f"  Mean: {sum(text_lengths) / len(text_lengths):.1f} characters")
        print(f"  Median: {sorted(text_lengths)[len(text_lengths) // 2]:.1f} characters")
        print(f"  Min: {min(text_lengths)} characters")
        print(f"  Max: {max(text_lengths)} characters")

def balance_dataset(data, target_samples=None):
    """Balance the dataset by undersampling the majority class."""
    label_counts = Counter(item['label'] for item in data)

    if target_samples is None:
        # Balance to the size of the smallest class
        target_samples = min(label_counts.values())

    print(f"\nBalancing dataset to {target_samples} samples per class...")

    # Group by label
    label_groups = defaultdict(list)
    for item in data:
        label_groups[item['label']].append(item)

    balanced_data = []
    for label, items in label_groups.items():
        if len(items) > target_samples:
            # Undersample with simple random sampling (no numpy/pandas)
            import random
            random.seed(42)
            sampled_items = random.sample(items, target_samples)
        else:
            sampled_items = items
        balanced_data.extend(sampled_items)

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(balanced_data)

    print(f"Balanced dataset size: {len(balanced_data)}")
    return balanced_data

def save_dataset(data, output_path):
    """Save the dataset to CSV."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(data)

def main():
    parser = argparse.ArgumentParser(description='Clean SMS spam/phishing dataset')
    parser.add_argument('--input', default='dataset/sms_spam_phishing_dataset_v2.1.csv',
                       help='Input dataset file path')
    parser.add_argument('--output', default='dataset/sms_spam_phishing_dataset_v2.1_cleaned.csv',
                       help='Output cleaned dataset file path')
    parser.add_argument('--balance', action='store_true',
                       help='Balance the dataset by undersampling majority classes')
    parser.add_argument('--balance-target', type=int,
                       help='Target number of samples per class when balancing')

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.input}...")
    data = load_dataset(args.input)

    # Clean dataset
    data = clean_dataset(data)

    # Print statistics
    print_statistics(data)

    # Balance if requested
    if args.balance or args.balance_target:
        target = args.balance_target
        if target is None:
            label_counts = Counter(item['label'] for item in data)
            target = min(label_counts.values())
        data = balance_dataset(data, target)

        print("\n=== BALANCED DATASET STATISTICS ===")
        print_statistics(data)

    # Save cleaned dataset
    output_path = args.output
    save_dataset(data, output_path)
    print(f"\nCleaned dataset saved to {output_path}")

if __name__ == '__main__':
    main()

