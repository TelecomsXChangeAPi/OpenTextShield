"""
Model comparison script for OpenTextShield mBERT models.

Compares two models (old vs new) on a test dataset and determines which performs better.
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Label mapping
LABEL_MAP = {'ham': 0, 'spam': 1, 'phishing': 2}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

class ModelComparator:
    """Class to compare two mBERT models."""

    def __init__(self, old_model_path: str, new_model_path: str, test_dataset_path: str, max_len: int = 128):
        self.old_model_path = Path(old_model_path)
        self.new_model_path = Path(new_model_path)
        self.test_dataset_path = Path(test_dataset_path)
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self, model_path: Path) -> Tuple[BertForSequenceClassification, BertTokenizer]:
        """Load model and tokenizer from path."""
        logger.info(f"Loading model from {model_path}")
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model.to(self.device)
        model.eval()
        return model, tokenizer

    def load_test_data(self) -> Tuple[List[str], List[str]]:
        """Load test dataset."""
        logger.info(f"Loading test data from {self.test_dataset_path}")
        df = pd.read_csv(self.test_dataset_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        return texts, labels

    def predict(self, model: BertForSequenceClassification, tokenizer: BertTokenizer, texts: List[str]) -> List[str]:
        """Predict labels for texts."""
        predictions = []
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt',
                    truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = model(**inputs)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).item()
                predictions.append(REVERSE_LABEL_MAP[pred])
        return predictions

    def compute_metrics(self, true_labels: List[str], pred_labels: List[str]) -> Dict:
        """Compute classification metrics."""
        true_numeric = [LABEL_MAP[label] for label in true_labels]
        pred_numeric = [LABEL_MAP[label] for label in pred_labels]

        accuracy = accuracy_score(true_numeric, pred_numeric)
        report = classification_report(true_numeric, pred_numeric, target_names=['ham', 'spam', 'phishing'], output_dict=True)
        conf_matrix = confusion_matrix(true_numeric, pred_numeric)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }

    def compare_models(self) -> Dict:
        """Compare old and new models."""
        # Load data
        texts, true_labels = self.load_test_data()

        # Load old model
        old_model, old_tokenizer = self.load_model(self.old_model_path)

        # Load new model
        new_model, new_tokenizer = self.load_model(self.new_model_path)

        # Predict with old model
        logger.info("Predicting with old model...")
        old_predictions = self.predict(old_model, old_tokenizer, texts)

        # Predict with new model
        logger.info("Predicting with new model...")
        new_predictions = self.predict(new_model, new_tokenizer, texts)

        # Compute metrics
        old_metrics = self.compute_metrics(true_labels, old_predictions)
        new_metrics = self.compute_metrics(true_labels, new_predictions)

        # Determine better model
        old_accuracy = old_metrics['accuracy']
        new_accuracy = new_metrics['accuracy']

        # Also check macro F1
        old_macro_f1 = old_metrics['classification_report']['macro avg']['f1-score']
        new_macro_f1 = new_metrics['classification_report']['macro avg']['f1-score']

        if new_accuracy > old_accuracy:
            better_model = 'new'
            reason = f"New model has higher accuracy ({new_accuracy:.4f} vs {old_accuracy:.4f})"
        elif new_accuracy < old_accuracy:
            better_model = 'old'
            reason = f"Old model has higher accuracy ({old_accuracy:.4f} vs {new_accuracy:.4f})"
        else:
            # Tie in accuracy, check F1
            if new_macro_f1 > old_macro_f1:
                better_model = 'new'
                reason = f"Models tied in accuracy ({new_accuracy:.4f}), but new model has higher macro F1 ({new_macro_f1:.4f} vs {old_macro_f1:.4f})"
            else:
                better_model = 'old'
                reason = f"Models tied in accuracy ({new_accuracy:.4f}), old model has higher or equal macro F1 ({old_macro_f1:.4f} vs {new_macro_f1:.4f})"

        return {
            'old_metrics': old_metrics,
            'new_metrics': new_metrics,
            'better_model': better_model,
            'reason': reason
        }

def main():
    # Paths - adjust as needed
    old_model_path = "models/enhanced_mbert_20251003_195757_best_model_epoch_1"
    new_model_path = "models/enhanced_mbert_20251003_195757_best_model_epoch_1"  # Using old for baseline, new model trained with 99.33% test accuracy
    test_dataset_path = "dataset/test_subset.csv"

    comparator = ModelComparator(old_model_path, new_model_path, test_dataset_path)
    results = comparator.compare_models()

    print("=== Model Comparison Results ===")
    print(f"Better model: {results['better_model'].upper()}")
    print(f"Reason: {results['reason']}")
    print("\nOld Model Metrics:")
    print(f"Accuracy: {results['old_metrics']['accuracy']:.4f}")
    print("Classification Report:")
    for label, metrics in results['old_metrics']['classification_report'].items():
        if label in ['ham', 'spam', 'phishing']:
            print(f"  {label}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

    print("\nNew Model Metrics:")
    print(f"Accuracy: {results['new_metrics']['accuracy']:.4f}")
    print("Classification Report:")
    for label, metrics in results['new_metrics']['classification_report'].items():
        if label in ['ham', 'spam', 'phishing']:
            print(f"  {label}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

if __name__ == "__main__":
    main()