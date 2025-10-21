#!/usr/bin/env python3
"""
Benchmark script for OpenTextShield mBERT
Tests the actual OpenTextShield API
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Configuration
OTS_API_URL = "http://localhost:8002/predict/"
TEST_DATASET_FILE = Path(__file__).parent / "test_dataset.json"
RESULTS_FILE = Path(__file__).parent / "results_opentextshield.json"

def load_test_dataset() -> Dict:
    """Load the test dataset."""
    with open(TEST_DATASET_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def classify_text(text: str) -> Tuple[Dict, float]:
    """
    Classify a single text using OpenTextShield.

    Returns:
        Tuple of (response_data, response_time)
    """
    start_time = time.time()

    payload = {
        "text": text,
        "model": "ots-mbert"
    }

    try:
        response = requests.post(OTS_API_URL, json=payload, timeout=60)
        response.raise_for_status()

        elapsed_time = time.time() - start_time

        result = response.json()

        return result, elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "error": str(e),
            "label": "error",
            "probability": 0.0
        }, elapsed_time

def run_benchmark() -> Dict:
    """Run the complete benchmark."""
    print("=" * 80)
    print("OpenTextShield Benchmark - mBERT SMS Classification")
    print("=" * 80)
    print()

    # Load resources
    print("Loading test dataset...")
    dataset = load_test_dataset()

    print(f"Dataset loaded: {len(dataset['samples'])} samples")
    print(f"Distribution: {dataset['metadata']['distribution']}")
    print()

    # Initialize results
    results = {
        "metadata": {
            "model": "OpenTextShield mBERT v2.5",
            "test_date": datetime.utcnow().isoformat() + "Z",
            "dataset": dataset['metadata'],
            "system_info": {
                "platform": "M4 Mac Mini",
                "api_endpoint": OTS_API_URL
            }
        },
        "predictions": [],
        "statistics": {
            "total_samples": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "errors": 0,
            "accuracy": 0.0,
            "avg_response_time": 0.0,
            "avg_processing_time": 0.0,
            "min_response_time": float('inf'),
            "max_response_time": 0.0,
            "by_class": {}
        }
    }

    # Run tests
    response_times = []
    processing_times = []

    for idx, sample in enumerate(dataset['samples'], 1):
        sample_id = sample['id']
        text = sample['text']
        ground_truth = sample['ground_truth']

        print(f"[{idx}/{len(dataset['samples'])}] Testing {sample_id} ({ground_truth})...")

        prediction, response_time = classify_text(text)
        response_times.append(response_time)

        predicted_label = prediction.get('label', 'error')
        is_correct = predicted_label == ground_truth

        # Get model processing time (internal)
        model_processing_time = prediction.get('processing_time', 0.0)
        if model_processing_time > 0:
            processing_times.append(model_processing_time)

        result_entry = {
            "sample_id": sample_id,
            "text": text,
            "ground_truth": ground_truth,
            "prediction": predicted_label,
            "probability": prediction.get('probability', 0.0),
            "response_time": response_time,
            "model_processing_time": model_processing_time,
            "correct": is_correct,
            "language": sample['language'],
            "category": sample['category'],
            "model_info": prediction.get('model_info', {})
        }

        results['predictions'].append(result_entry)

        # Update statistics
        results['statistics']['total_samples'] += 1
        if is_correct:
            results['statistics']['correct_predictions'] += 1
            print(f"  ✓ Correct: {predicted_label} (confidence: {prediction.get('probability', 0.0):.2f}, time: {response_time:.3f}s)")
        elif predicted_label == 'error':
            results['statistics']['errors'] += 1
            print(f"  ✗ Error: {prediction.get('error', 'Unknown error')}")
        else:
            results['statistics']['incorrect_predictions'] += 1
            print(f"  ✗ Incorrect: predicted {predicted_label}, expected {ground_truth} (time: {response_time:.3f}s)")

        print()

    # Calculate final statistics
    total = results['statistics']['total_samples']
    correct = results['statistics']['correct_predictions']

    results['statistics']['accuracy'] = (correct / total * 100) if total > 0 else 0.0
    results['statistics']['avg_response_time'] = sum(response_times) / len(response_times) if response_times else 0.0
    results['statistics']['avg_processing_time'] = sum(processing_times) / len(processing_times) if processing_times else 0.0
    results['statistics']['min_response_time'] = min(response_times) if response_times else 0.0
    results['statistics']['max_response_time'] = max(response_times) if response_times else 0.0

    # Per-class statistics
    for label in ['ham', 'spam', 'phishing']:
        class_samples = [p for p in results['predictions'] if p['ground_truth'] == label]
        class_correct = [p for p in class_samples if p['correct']]

        tp = len([p for p in class_correct if p['prediction'] == label])
        fp = len([p for p in results['predictions'] if p['prediction'] == label and p['ground_truth'] != label])
        fn = len([p for p in class_samples if p['prediction'] != label])
        tn = len([p for p in results['predictions'] if p['prediction'] != label and p['ground_truth'] != label])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results['statistics']['by_class'][label] = {
            "total_samples": len(class_samples),
            "correct": len(class_correct),
            "accuracy": len(class_correct) / len(class_samples) * 100 if class_samples else 0.0,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "true_negatives": tn
            }
        }

    # Save results
    print("=" * 80)
    print("Saving results...")
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {RESULTS_FILE}")
    print()

    # Print summary
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {results['statistics']['incorrect_predictions']}")
    print(f"Errors: {results['statistics']['errors']}")
    print(f"Overall Accuracy: {results['statistics']['accuracy']:.2f}%")
    print(f"Avg Response Time: {results['statistics']['avg_response_time']:.3f}s")
    print(f"Avg Model Processing Time: {results['statistics']['avg_processing_time']:.3f}s")
    print(f"Min Response Time: {results['statistics']['min_response_time']:.3f}s")
    print(f"Max Response Time: {results['statistics']['max_response_time']:.3f}s")
    print()
    print("Per-Class Metrics:")
    for label in ['ham', 'spam', 'phishing']:
        stats = results['statistics']['by_class'][label]
        print(f"  {label.upper()}:")
        print(f"    Accuracy: {stats['accuracy']:.2f}%")
        print(f"    Precision: {stats['precision']:.3f}")
        print(f"    Recall: {stats['recall']:.3f}")
        print(f"    F1-Score: {stats['f1_score']:.3f}")
    print("=" * 80)

    return results

if __name__ == "__main__":
    run_benchmark()
