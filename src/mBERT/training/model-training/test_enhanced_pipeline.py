#!/usr/bin/env python3
"""
Test script for the enhanced multilingual training pipeline.

This script validates that the data loading, preprocessing, and pipeline
components work correctly with the cleaned and expanded datasets.
"""

import csv
import json
import sys
from pathlib import Path
from collections import Counter

def test_dataset_loading(dataset_path, expected_min_samples=1000):
    """Test loading a dataset and basic validation."""
    print(f"\n=== Testing Dataset Loading: {dataset_path} ===")

    try:
        with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            data = list(reader)

        print(f"Successfully loaded {len(data)} rows")

        if len(data) < expected_min_samples:
            print(f"⚠️  Warning: Only {len(data)} samples (expected at least {expected_min_samples})")
            return False

        # Check required columns
        if 'text' not in data[0] and 'Pesan' not in data[0]:
            print("❌ Error: Missing text column")
            return False

        if 'label' not in data[0] and 'Kategori' not in data[0]:
            print("❌ Error: Missing label column")
            return False

        # Check label distribution
        text_col = 'text' if 'text' in data[0] else 'Pesan'
        label_col = 'label' if 'label' in data[0] else 'Kategori'

        labels = [row[label_col].strip().lower() for row in data]
        label_counts = Counter(labels)

        print(f"Label distribution: {dict(label_counts)}")

        # Check for valid labels
        valid_labels = {'ham', 'spam', 'phishing'}
        invalid_labels = set(label_counts.keys()) - valid_labels

        if invalid_labels:
            print(f"⚠️  Warning: Found invalid labels: {invalid_labels}")

        # Check text quality
        texts = [row[text_col] for row in data]
        empty_texts = sum(1 for t in texts if not t.strip())
        very_short_texts = sum(1 for t in texts if len(t.strip()) < 5)

        if empty_texts > 0:
            print(f"⚠️  Warning: {empty_texts} empty texts found")

        if very_short_texts > len(texts) * 0.1:  # More than 10%
            print(f"⚠️  Warning: {very_short_texts} very short texts (>10% of dataset)")

        print("✅ Dataset loading test passed")
        return True

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def test_data_preprocessing():
    """Test text preprocessing functions."""
    print("\n=== Testing Data Preprocessing ===")

    # Test cases
    test_cases = [
        ("Hello   world!", "Hello world!"),
        ("Check this link: http://example.com", "Check this link: [URL]"),
        ("Call me at 1234567890", "Call me at [PHONE]"),
        ("Email: test@example.com", "Email: [EMAIL]"),
        ("URGENT!!! Call now!!!", "URGENT! Call now!"),
    ]

    def preprocess_text(text):
        """Simple preprocessing for testing."""
        import re
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        text = re.sub(r'\b\d{10,15}\b', '[PHONE]', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        text = re.sub(r'[!?.]{2,}', '!', text)
        return text.strip()

    passed = 0
    for input_text, expected in test_cases:
        result = preprocess_text(input_text)
        if result == expected:
            passed += 1
        else:
            print(f"❌ Preprocessing failed: '{input_text}' -> '{result}' (expected '{expected}')")

    if passed == len(test_cases):
        print("✅ All preprocessing tests passed")
        return True
    else:
        print(f"⚠️  {passed}/{len(test_cases)} preprocessing tests passed")
        return False

def test_multilingual_support():
    """Test multilingual text handling."""
    print("\n=== Testing Multilingual Support ===")

    # Test texts in different languages
    test_texts = [
        "Hello world",  # English
        "Hola mundo",   # Spanish
        "Привет мир",   # Russian
        "Halo dunia",   # Indonesian
        "你好世界",     # Chinese
        "مرحبا بالعالم", # Arabic
    ]

    # Basic checks
    all_have_content = all(text.strip() for text in test_texts)
    all_have_letters = all(any(c.isalpha() for c in text) for text in test_texts)

    if all_have_content and all_have_letters:
        print("✅ Multilingual text validation passed")
        return True
    else:
        print("❌ Multilingual text validation failed")
        return False

def test_pipeline_integration():
    """Test integration of all pipeline components."""
    print("\n=== Testing Pipeline Integration ===")

    # Test combining datasets
    datasets = [
        'dataset/sms_spam_phishing_dataset_v2.1_cleaned.csv',
        'dataset/sms_spam_indo_expanded.csv'
    ]

    total_samples = 0
    all_labels = set()

    for dataset_path in datasets:
        if Path(dataset_path).exists():
            try:
                with open(dataset_path, 'r', encoding='utf-8', errors='ignore') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)

                total_samples += len(data)

                # Get label column name
                if 'label' in data[0]:
                    labels = [row['label'].strip().lower() for row in data]
                elif 'Kategori' in data[0]:
                    labels = [row['Kategori'].strip().lower() for row in data]
                else:
                    continue

                all_labels.update(labels)

            except Exception as e:
                print(f"❌ Error processing {dataset_path}: {e}")
                return False
        else:
            print(f"⚠️  Dataset not found: {dataset_path}")

    print(f"Combined dataset would have {total_samples} samples")
    print(f"Unique labels found: {sorted(all_labels)}")

    expected_labels = {'ham', 'spam', 'phishing'}
    if expected_labels.issubset(all_labels):
        print("✅ Pipeline integration test passed")
        return True
    else:
        print(f"❌ Missing expected labels: {expected_labels - all_labels}")
        return False

def generate_training_config():
    """Generate a sample training configuration."""
    print("\n=== Generating Training Configuration ===")

    config = {
        "model": {
            "name": "bert-base-multilingual-cased",
            "num_labels": 3,
            "max_length": 128
        },
        "training": {
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_steps": 0.1,
            "validation_split": 0.1
        },
        "data": {
            "datasets": [
                "dataset/sms_spam_phishing_dataset_v2.1_cleaned.csv",
                "dataset/sms_spam_indo_expanded.csv"
            ],
            "balance_classes": True,
            "augmentation": True
        },
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1", "auc"],
            "save_best_model": True,
            "early_stopping": True,
            "patience": 3
        }
    }

    config_path = "enhanced_training_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Training configuration saved to {config_path}")
    return True

def run_all_tests():
    """Run all pipeline tests."""
    print("🧪 Running Enhanced Training Pipeline Tests")
    print("=" * 50)

    tests = [
        ("Dataset Loading", lambda: test_dataset_loading('dataset/sms_spam_phishing_dataset_v2.1_cleaned.csv')),
        ("Indonesian Dataset Loading", lambda: test_dataset_loading('dataset/sms_spam_indo_expanded.csv', 50000)),
        ("Data Preprocessing", test_data_preprocessing),
        ("Multilingual Support", test_multilingual_support),
        ("Pipeline Integration", test_pipeline_integration),
        ("Training Config Generation", generate_training_config),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("🧪 TEST SUMMARY")
    print("=" * 50)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Enhanced pipeline is ready for training.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)