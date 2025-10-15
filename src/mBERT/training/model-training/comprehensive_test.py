"""
Comprehensive testing script for OpenTextShield mBERT model.

Tests accuracy across languages, edge cases, and identifies weak points.
"""

import requests
import pandas as pd
import json
import time
from collections import defaultdict, Counter
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTester:
    """Comprehensive model testing class."""

    def __init__(self, api_url: str = "http://localhost:8002/predict/"):
        self.api_url = api_url
        self.results = []

    def test_sample(self, text: str, true_label: str) -> Dict:
        """Test a single sample."""
        payload = {
            "text": text,
            "model": "ots-mbert"
        }

        try:
            start_time = time.time()
            response = requests.post(self.api_url, json=payload, timeout=10)
            processing_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                predicted_label = result.get('label')
                probability = result.get('probability', 0)

                correct = predicted_label == true_label
                confidence = probability if predicted_label == true_label else 1 - probability

                return {
                    'text': text,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'probability': probability,
                    'correct': correct,
                    'confidence': confidence,
                    'processing_time': processing_time,
                    'error': None
                }
            else:
                return {
                    'text': text,
                    'true_label': true_label,
                    'predicted_label': None,
                    'probability': 0,
                    'correct': False,
                    'confidence': 0,
                    'processing_time': processing_time,
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                'text': text,
                'true_label': true_label,
                'predicted_label': None,
                'probability': 0,
                'correct': False,
                'confidence': 0,
                'processing_time': 0,
                'error': str(e)
            }

    def test_dataset(self, dataset_path: str, max_samples: int = None) -> pd.DataFrame:
        """Test entire dataset."""
        logger.info(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)

        if max_samples:
            df = df.sample(n=max_samples, random_state=42)

        logger.info(f"Testing {len(df)} samples")

        results = []
        for idx, row in df.iterrows():
            result = self.test_sample(row['text'], row['label'])
            results.append(result)

            if (idx + 1) % 50 == 0:
                logger.info(f"Tested {idx + 1}/{len(df)} samples")

        self.results_df = pd.DataFrame(results)
        return self.results_df

    def analyze_results(self) -> Dict:
        """Analyze test results."""
        if not hasattr(self, 'results_df'):
            raise ValueError("Run test_dataset first")

        df = self.results_df

        # Basic metrics
        total_samples = len(df)
        correct_predictions = df['correct'].sum()
        accuracy = correct_predictions / total_samples

        # Per-class metrics
        class_metrics = {}
        for label in ['ham', 'spam', 'phishing']:
            class_df = df[df['true_label'] == label]
            if len(class_df) > 0:
                class_correct = class_df['correct'].sum()
                class_accuracy = class_correct / len(class_df)
                class_metrics[label] = {
                    'accuracy': class_accuracy,
                    'samples': len(class_df),
                    'correct': class_correct
                }

        # Error analysis
        errors = df[~df['correct']].copy()
        error_patterns = self._analyze_errors(errors)

        # Confidence analysis
        low_confidence = df[df['confidence'] < 0.8]

        # Processing time stats
        avg_time = df['processing_time'].mean()
        max_time = df['processing_time'].max()

        return {
            'total_samples': total_samples,
            'accuracy': accuracy,
            'class_metrics': class_metrics,
            'error_analysis': error_patterns,
            'low_confidence_count': len(low_confidence),
            'avg_processing_time': avg_time,
            'max_processing_time': max_time,
            'errors_df': errors
        }

    def _analyze_errors(self, errors_df: pd.DataFrame) -> Dict:
        """Analyze error patterns."""
        patterns = {
            'misclassifications': defaultdict(list),
            'common_errors': Counter(),
            'text_lengths': [],
            'low_confidence_errors': 0
        }

        for _, row in errors_df.iterrows():
            true = row['true_label']
            pred = row['predicted_label']
            text = row['text']

            patterns['misclassifications'][f"{true}_to_{pred}"].append(text)
            patterns['common_errors'][(true, pred)] += 1
            patterns['text_lengths'].append(len(text))

            if row['confidence'] < 0.8:
                patterns['low_confidence_errors'] += 1

        return patterns

    def test_edge_cases(self) -> Dict:
        """Test edge cases."""
        edge_cases = {
            'short_messages': [
                ("Hi", "ham"),
                ("OK", "ham"),
                ("Win!", "spam"),
                ("Call", "spam")
            ],
            'long_messages': [
                ("This is a very long message that contains a lot of text and should test how the model handles lengthy SMS messages that might contain multiple sentences and various information." * 3, "ham")
            ],
            'special_chars': [
                ("🎉 Win $1000! Click here: http://bit.ly/123 🎉", "spam"),
                ("Hello!!! How are you??? 😊", "ham"),
                ("URGENT: Your account #12345 is locked!!!", "phishing")
            ],
            'numbers_only': [
                ("123456", "ham"),
                ("Your code is 123456", "ham")
            ],
            'urls': [
                ("Check this: https://google.com", "ham"),
                ("Win prize at http://fake.com/prize", "phishing")
            ],
            'mixed_languages': [
                ("Hello, cómo estás? Guten Tag!", "ham"),
                ("Win 1000€! Cliquez ici: http://spam.fr", "spam")
            ]
        }

        edge_results = {}
        for category, samples in edge_cases.items():
            logger.info(f"Testing {category}")
            results = []
            for text, true_label in samples:
                result = self.test_sample(text, true_label)
                results.append(result)
            edge_results[category] = results

        return edge_results

    def test_multilingual(self, language_samples: Dict[str, List[Tuple[str, str]]]) -> Dict:
        """Test multilingual capabilities."""
        multilingual_results = {}
        for language, samples in language_samples.items():
            logger.info(f"Testing {language}")
            results = []
            for text, true_label in samples:
                result = self.test_sample(text, true_label)
                results.append(result)
            multilingual_results[language] = results

        return multilingual_results

    def generate_report(self, analysis: Dict, edge_results: Dict, multilingual_results: Dict) -> str:
        """Generate comprehensive test report."""
        report = []
        report.append("# OpenTextShield mBERT Comprehensive Test Report")
        report.append("")

        report.append("## Overall Performance")
        report.append(f"- Total Samples: {analysis['total_samples']}")
        report.append(f"- Accuracy: {analysis['accuracy']:.4f}")
        report.append(f"- Average Processing Time: {analysis['avg_processing_time']:.4f}s")
        report.append(f"- Max Processing Time: {analysis['max_processing_time']:.4f}s")
        report.append("")

        report.append("## Per-Class Performance")
        for label, metrics in analysis['class_metrics'].items():
            report.append(f"- {label.upper()}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['samples']})")
        report.append("")

        report.append("## Error Analysis")
        report.append(f"- Total Errors: {len(analysis['errors_df'])}")
        report.append(f"- Low Confidence Predictions: {analysis['low_confidence_count']}")
        report.append(f"- Low Confidence Errors: {analysis['error_analysis']['low_confidence_errors']}")
        report.append("")

        report.append("### Common Misclassifications")
        for (true, pred), count in analysis['error_analysis']['common_errors'].most_common(5):
            report.append(f"- {true} → {pred}: {count} times")
        report.append("")

        report.append("## Edge Cases")
        for category, results in edge_results.items():
            correct = sum(1 for r in results if r['correct'])
            total = len(results)
            accuracy = correct / total if total > 0 else 0
            report.append(f"- {category}: {accuracy:.2f} ({correct}/{total})")
        report.append("")

        report.append("## Multilingual Performance")
        for language, results in multilingual_results.items():
            correct = sum(1 for r in results if r['correct'])
            total = len(results)
            accuracy = correct / total if total > 0 else 0
            report.append(f"- {language}: {accuracy:.2f} ({correct}/{total})")
        report.append("")

        report.append("## Recommendations")
        report.append("1. **Weak Areas Identified**:")
        if analysis['error_analysis']['common_errors']:
            most_common = analysis['error_analysis']['common_errors'].most_common(1)[0]
            report.append(f"   - Most common error: {most_common[0][0]} misclassified as {most_common[0][1]} ({most_common[1]} times)")

        report.append("2. **Data Augmentation Needed**:")
        report.append("   - Generate more samples for error-prone categories")
        report.append("   - Add multilingual data for underrepresented languages")
        report.append("   - Include more edge cases in training")

        report.append("3. **Model Improvements**:")
        report.append("   - Fine-tune on augmented dataset")
        report.append("   - Consider ensemble methods for uncertain predictions")
        report.append("   - Implement confidence thresholding")

        return "\n".join(report)

def main():
    tester = ModelTester()

    # Test main dataset
    logger.info("Starting comprehensive testing")
    results_df = tester.test_dataset("dataset/test_subset.csv", max_samples=500)
    analysis = tester.analyze_results()

    # Test edge cases
    edge_results = tester.test_edge_cases()

    # Test multilingual (sample languages)
    language_samples = {
        'Spanish': [
            ("Hola, ¿cómo estás?", "ham"),
            ("¡Gana 1000€! Haz clic aquí: http://spam.es", "spam"),
            ("Tu cuenta ha sido bloqueada. Inicia sesión en http://banco-falso.com", "phishing")
        ],
        'French': [
            ("Salut, comment ça va?", "ham"),
            ("Gagnez 1000€! Cliquez ici: http://spam.fr", "spam"),
            ("Votre compte est verrouillé. Connectez-vous à http://banque-fausse.fr", "phishing")
        ],
        'German': [
            ("Hallo, wie geht's?", "ham"),
            ("Gewinne 1000€! Klicke hier: http://spam.de", "spam"),
            ("Ihr Konto wurde gesperrt. Melden Sie sich bei http://bank-falsch.de an", "phishing")
        ],
        'Arabic': [
            ("مرحبا، كيف حالك؟", "ham"),
            ("اربح 1000 دولار! اضغط هنا: http://spam.ar", "spam"),
            ("تم قفل حسابك. سجل الدخول إلى http://bank-fake.ar", "phishing")
        ],
        'Chinese': [
            ("你好，你怎么样？", "ham"),
            ("赢取1000美元！点击这里：http://spam.cn", "spam"),
            ("您的账户已被锁定。请登录http://bank-fake.cn", "phishing")
        ]
    }
    multilingual_results = tester.test_multilingual(language_samples)

    # Generate report
    report = tester.generate_report(analysis, edge_results, multilingual_results)

    with open("comprehensive_test_report.md", "w") as f:
        f.write(report)

    logger.info("Comprehensive testing completed. Report saved to comprehensive_test_report.md")

    # Print summary
    print(f"Overall Accuracy: {analysis['accuracy']:.4f}")
    print(f"Total Samples: {analysis['total_samples']}")
    print(f"Errors: {len(analysis['errors_df'])}")
    print("Report generated: comprehensive_test_report.md")

if __name__ == "__main__":
    main()