#!/usr/bin/env python3
"""
Benchmark script for GPT-OSS-20B SMS Classification
Tests the GPT-OSS-20B model configured with OpenTextShield prompt
"""

import json
import time
import requests
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Configuration
GPT_OSS_API_URL = "http://0.0.0.0:1234/v1/chat/completions"
SYSTEM_PROMPT_FILE = Path(__file__).parent / "opentextshield_prompt.txt"
TEST_DATASET_FILE = Path(__file__).parent / "test_dataset.json"
RESULTS_FILE = Path(__file__).parent / "results_gpt_oss.json"

def load_system_prompt() -> str:
    """Load the OpenTextShield system prompt."""
    with open(SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
        return f.read()

def load_test_dataset() -> Dict:
    """Load the test dataset."""
    with open(TEST_DATASET_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def classify_text(text: str, system_prompt: str) -> Tuple[Dict, float, str]:
    """
    Classify a single text using GPT-OSS-20B.

    Returns:
        Tuple of (parsed_response, response_time, raw_content)
    """
    start_time = time.time()

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,
        "max_tokens": 500,
        "stream": False
    }

    try:
        response = requests.post(GPT_OSS_API_URL, json=payload, timeout=60)
        response.raise_for_status()

        elapsed_time = time.time() - start_time

        result = response.json()
        content = result['choices'][0]['message']['content']

        # Parse JSON from content (strip markdown code blocks if present)
        content_clean = content.strip()
        if content_clean.startswith('```json'):
            content_clean = content_clean[7:]
        if content_clean.startswith('```'):
            content_clean = content_clean[3:]
        if content_clean.endswith('```'):
            content_clean = content_clean[:-3]
        content_clean = content_clean.strip()

        parsed = json.loads(content_clean)

        return parsed, elapsed_time, content

    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "error": str(e),
            "label": "error",
            "probability": 0.0
        }, elapsed_time, str(e)

def run_benchmark() -> Dict:
    """Run the complete benchmark."""
    print("=" * 80)
    print("GPT-OSS-20B Benchmark - OpenTextShield SMS Classification")
    print("=" * 80)
    print()

    # Load resources
    print("Loading system prompt and test dataset...")
    system_prompt = load_system_prompt()
    dataset = load_test_dataset()

    print(f"Dataset loaded: {len(dataset['samples'])} samples")
    print(f"Distribution: {dataset['metadata']['distribution']}")
    print()

    # Initialize results
    results = {
        "metadata": {
            "model": "openai/gpt-oss-20b",
            "test_date": datetime.utcnow().isoformat() + "Z",
            "dataset": dataset['metadata'],
            "system_info": {
                "platform": "M4 Mac Mini",
                "api_endpoint": GPT_OSS_API_URL
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
            "min_response_time": float('inf'),
            "max_response_time": 0.0,
            "by_class": {}
        }
    }

    # Run tests
    response_times = []

    for idx, sample in enumerate(dataset['samples'], 1):
        sample_id = sample['id']
        text = sample['text']
        ground_truth = sample['ground_truth']

        print(f"[{idx}/{len(dataset['samples'])}] Testing {sample_id} ({ground_truth})...")

        prediction, response_time, raw_content = classify_text(text, system_prompt)
        response_times.append(response_time)

        predicted_label = prediction.get('label', 'error')
        is_correct = predicted_label == ground_truth

        result_entry = {
            "sample_id": sample_id,
            "text": text,
            "ground_truth": ground_truth,
            "prediction": predicted_label,
            "probability": prediction.get('probability', 0.0),
            "response_time": response_time,
            "correct": is_correct,
            "language": sample['language'],
            "category": sample['category'],
            "raw_response": raw_content
        }

        results['predictions'].append(result_entry)

        # Update statistics
        results['statistics']['total_samples'] += 1
        if is_correct:
            results['statistics']['correct_predictions'] += 1
            print(f"  ✓ Correct: {predicted_label} (confidence: {prediction.get('probability', 0.0):.2f}, time: {response_time:.2f}s)")
        elif predicted_label == 'error':
            results['statistics']['errors'] += 1
            print(f"  ✗ Error: {prediction.get('error', 'Unknown error')}")
        else:
            results['statistics']['incorrect_predictions'] += 1
            print(f"  ✗ Incorrect: predicted {predicted_label}, expected {ground_truth} (time: {response_time:.2f}s)")

        print()

    # Calculate final statistics
    total = results['statistics']['total_samples']
    correct = results['statistics']['correct_predictions']

    results['statistics']['accuracy'] = (correct / total * 100) if total > 0 else 0.0
    results['statistics']['avg_response_time'] = sum(response_times) / len(response_times) if response_times else 0.0
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
