#!/usr/bin/env python3
"""
Dataset Validation Tool for OpenTextShield SMS Spam/Phishing Dataset

This script validates datasets for:
1. Label consistency and validity
2. Text quality and length checks
3. Duplicate detection
4. Class balance analysis
5. Language detection (basic)
6. Suspicious pattern detection
"""

import re
import csv
from collections import Counter, defaultdict
import argparse
import sys

def load_dataset(file_path):
    """Load dataset from CSV format."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'text' in row and 'label' in row:
                    data.append({
                        'text': row['text'].strip(),
                        'label': row['label'].strip().lower()
                    })
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

    return data

def validate_labels(data):
    """Validate label consistency."""
    print("=== LABEL VALIDATION ===")

    valid_labels = {'ham', 'spam', 'phishing'}
    label_counts = Counter()
    invalid_labels = Counter()

    for item in data:
        label = item['label']
        if label in valid_labels:
            label_counts[label] += 1
        else:
            invalid_labels[label] += 1

    print(f"Valid labels found: {dict(label_counts)}")
    if invalid_labels:
        print(f"Invalid labels found: {dict(invalid_labels)}")
        return False

    total = sum(label_counts.values())
    print("Label distribution:")
    for label, count in label_counts.items():
        percentage = (count / total) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    return True

def validate_text_quality(data):
    """Validate text quality."""
    print("\n=== TEXT QUALITY VALIDATION ===")

    issues = {
        'empty_text': 0,
        'too_short': 0,
        'too_long': 0,
        'only_numbers': 0,
        'only_punctuation': 0,
        'suspicious_urls': 0,
        'encoding_issues': 0,
        'malformed_quotes': 0
    }

    text_lengths = []

    for item in data:
        text = item['text']

        if not text:
            issues['empty_text'] += 1
            continue

        text_lengths.append(len(text))

        if len(text) < 5:
            issues['too_short'] += 1

        if len(text) > 1000:
            issues['too_long'] += 1

        if re.match(r'^\s*\d+\s*$', text):
            issues['only_numbers'] += 1

        if re.match(r'^\s*[^\w\s]+\s*$', text):
            issues['only_punctuation'] += 1

        if 'http://' in text or 'https://' in text or 'bit.ly' in text or 'tinyurl.com' in text:
            issues['suspicious_urls'] += 1

        # Check for encoding issues (replacement characters)
        if '�' in text:
            issues['encoding_issues'] += 1

        # Check for malformed quotes
        if ('"' in text and text.count('"') % 2 != 0) or ('\'' in text and text.count('\'') % 2 != 0):
            issues['malformed_quotes'] += 1

    print("Text quality issues:")
    for issue, count in issues.items():
        if count > 0:
            percentage = (count / len(data)) * 100
            print(f"  {issue}: {count} ({percentage:.2f}%)")

    if text_lengths:
        print("\nText length statistics:")
        print(f"  Mean: {sum(text_lengths) / len(text_lengths):.1f} characters")
        print(f"  Median: {sorted(text_lengths)[len(text_lengths) // 2]:.1f} characters")
        print(f"  Min: {min(text_lengths)} characters")
        print(f"  Max: {max(text_lengths)} characters")

    return issues

def detect_duplicates(data):
    """Detect duplicate entries."""
    print("\n=== DUPLICATE DETECTION ===")

    text_counts = Counter(item['text'] for item in data)
    duplicates = {text: count for text, count in text_counts.items() if count > 1}

    if duplicates:
        print(f"Found {len(duplicates)} duplicate texts")
        print("Most common duplicates:")
        sorted_duplicates = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)
        for text, count in sorted_duplicates[:10]:
            print(f"  {count}x: {text[:50]}{'...' if len(text) > 50 else ''}")
    else:
        print("No duplicates found")

    return duplicates

def analyze_class_balance(data):
    """Analyze class balance and provide recommendations."""
    print("\n=== CLASS BALANCE ANALYSIS ===")

    label_counts = Counter(item['label'] for item in data)
    total = sum(label_counts.values())

    if len(label_counts) < 2:
        print("Warning: Dataset has fewer than 2 classes")
        return

    # Calculate imbalance ratios
    ham_count = label_counts.get('ham', 0)
    spam_count = label_counts.get('spam', 0)
    phishing_count = label_counts.get('phishing', 0)

    print("Imbalance analysis:")
    if spam_count > 0:
        ratio = ham_count / spam_count
        print(f"  Ham/Spam ratio: {ratio:.2f}")
        if ratio > 5:
            print("    ⚠️  Severe imbalance: ham class dominates spam")
        elif ratio > 3:
            print("    ⚠️  Moderate imbalance: consider balancing")

    if phishing_count > 0:
        ratio = ham_count / phishing_count
        print(f"  Ham/Phishing ratio: {ratio:.2f}")
        if ratio > 5:
            print("    ⚠️  Severe imbalance: ham class dominates phishing")
        elif ratio > 3:
            print("    ⚠️  Moderate imbalance: consider balancing")

    # Recommendations
    min_samples = min(label_counts.values())
    max_samples = max(label_counts.values())

    if max_samples / min_samples > 3:
        print(f"\nRecommendations:")
        print(f"  Consider balancing to ~{min_samples} samples per class")
        print("  Options: undersample majority class or oversample minority classes")

def detect_suspicious_patterns(data):
    """Detect potentially suspicious or problematic patterns."""
    print("\n=== SUSPICIOUS PATTERN DETECTION ===")

    patterns = {
        'test_messages': 0,
        'placeholder_text': 0,
        'lorem_ipsum': 0,
        'repeated_chars': 0,
        'all_caps': 0,
        'excessive_spaces': 0
    }

    for item in data:
        text = item['text'].lower()

        # Test messages
        if any(word in text for word in ['test', 'testing', 'sample', 'example']):
            patterns['test_messages'] += 1

        # Placeholder text
        if any(word in text for word in ['lorem', 'ipsum', 'placeholder', 'dummy']):
            patterns['placeholder_text'] += 1

        # Lorem ipsum
        if 'lorem ipsum' in text:
            patterns['lorem_ipsum'] += 1

        # Repeated characters
        if re.search(r'(.)\1{4,}', text):
            patterns['repeated_chars'] += 1

        # All caps (long messages)
        if len(text) > 20 and text.isupper():
            patterns['all_caps'] += 1

        # Excessive spaces
        if '  ' in item['text']:  # Double spaces
            patterns['excessive_spaces'] += 1

    print("Suspicious patterns detected:")
    for pattern, count in patterns.items():
        if count > 0:
            percentage = (count / len(data)) * 100
            print(f"  {pattern}: {count} ({percentage:.2f}%)")

    return patterns

def basic_language_detection(data):
    """Basic language detection using character patterns."""
    print("\n=== BASIC LANGUAGE DETECTION ===")

    languages = {
        'english': 0,
        'cyrillic': 0,  # Russian, Serbian, etc.
        'arabic': 0,
        'chinese': 0,
        'other': 0
    }

    for item in data:
        text = item['text']

        if re.search(r'[а-яё]', text, re.IGNORECASE):  # Cyrillic
            languages['cyrillic'] += 1
        elif re.search(r'[\u0600-\u06FF]', text):  # Arabic
            languages['arabic'] += 1
        elif re.search(r'[\u4e00-\u9fff]', text):  # Chinese
            languages['chinese'] += 1
        elif re.match(r'^[a-zA-Z\s\.,!?\'"-]+$', text):  # Mostly English characters
            languages['english'] += 1
        else:
            languages['other'] += 1

    total = sum(languages.values())
    print("Language distribution (approximate):")
    for lang, count in languages.items():
        if count > 0:
            percentage = (count / total) * 100
            print(f"  {lang}: {count} ({percentage:.1f}%)")

    return languages

def generate_report(data, output_file=None):
    """Generate a comprehensive validation report."""
    print("\n=== VALIDATION REPORT ===")

    all_checks_passed = True

    # Run all validations
    labels_valid = validate_labels(data)
    if not labels_valid:
        all_checks_passed = False

    quality_issues = validate_text_quality(data)
    duplicates = detect_duplicates(data)
    analyze_class_balance(data)
    suspicious_patterns = detect_suspicious_patterns(data)
    languages = basic_language_detection(data)

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Total samples: {len(data)}")

    if all_checks_passed:
        print("✅ All basic validation checks passed")
    else:
        print("❌ Some validation checks failed")

    # Quality score (simple heuristic)
    quality_score = 100
    quality_score -= len(duplicates) * 0.1  # Penalize duplicates
    quality_score -= quality_issues['empty_text'] * 0.5
    quality_score -= quality_issues['encoding_issues'] * 0.2
    quality_score = max(0, min(100, quality_score))

    print(f"Estimated dataset quality score: {quality_score:.1f}/100")

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Dataset Validation Report\n")
            f.write("========================\n\n")
            f.write(f"Total samples: {len(data)}\n")
            f.write(f"Quality score: {quality_score:.1f}/100\n\n")

            f.write("Label distribution:\n")
            label_counts = Counter(item['label'] for item in data)
            for label, count in label_counts.items():
                percentage = (count / len(data)) * 100
                f.write(f"  {label}: {count} ({percentage:.1f}%)\n")

            f.write("\nIssues found:\n")
            for issue, count in quality_issues.items():
                if count > 0:
                    f.write(f"  {issue}: {count}\n")

        print(f"\nReport saved to {output_file}")

    return all_checks_passed

def main():
    parser = argparse.ArgumentParser(description='Validate SMS spam/phishing dataset')
    parser.add_argument('input_file', help='Input dataset file path')
    parser.add_argument('--report', help='Output report file path')
    parser.add_argument('--strict', action='store_true',
                       help='Exit with error code if validation fails')

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.input_file}...")
    data = load_dataset(args.input_file)

    if not data:
        print("Error: No data loaded")
        sys.exit(1)

    # Run validation
    validation_passed = generate_report(data, args.report)

    # Exit with appropriate code
    if args.strict and not validation_passed:
        sys.exit(1)

if __name__ == '__main__':
    main()