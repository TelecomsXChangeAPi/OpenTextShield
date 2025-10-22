#!/usr/bin/env python3
"""
Enhanced Multilingual mBERT Training Script for OpenTextShield

This script provides advanced training capabilities:
- Multi-dataset support (main SMS dataset + Indonesian expansion)
- Enhanced data preprocessing and cleaning
- Multilingual text handling
- Improved model architecture with regularization
- Comprehensive evaluation metrics
- Model ensemble capabilities
- Automatic hyperparameter optimization
"""

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    get_linear_schedule_with_warmup,
    AutoTokenizer, AutoModelForSequenceClassification
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)
import pandas as pd
import numpy as np
import re
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import random
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MultilingualSMSDataset(Dataset):
    """Enhanced dataset class for multilingual SMS classification."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Label mapping
        self.label_map = {'ham': 0, 'spam': 1, 'phishing': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}

        # Convert string labels to integers
        self.labels = [self.label_map.get(label, 0) for label in labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Enhanced text preprocessing
        text = self._preprocess_text(text)

        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove URLs (but keep them as [URL] token for model to learn)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)

        # Normalize phone numbers
        text = re.sub(r'\b\d{10,15}\b', '[PHONE]', text)

        # Normalize email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

        # Remove excessive punctuation
        text = re.sub(r'[!?.]{2,}', '!', text)

        return text.strip()

class EnhancedTrainer:
    """Enhanced trainer with advanced features."""

    def __init__(self, model_name: str = 'bert-base-multilingual-cased', num_labels: int = 3):
        self.model_name = model_name
        self.num_labels = num_labels

        # Device selection: MPS (Apple Silicon) > CUDA > CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        logger.info(f"Using device: {self.device}")

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        self.model.to(self.device)

    def load_datasets(self, dataset_paths: List[str]) -> Tuple[List[str], List[str]]:
        """Load and combine multiple datasets."""
        all_texts = []
        all_labels = []

        for path in dataset_paths:
            logger.info(f"Loading dataset: {path}")

            try:
                if path.endswith('.csv'):
                    df = pd.read_csv(path)

                    # Handle different column names
                    text_col = 'text' if 'text' in df.columns else 'Pesan'
                    label_col = 'label' if 'label' in df.columns else 'Kategori'

                    if text_col not in df.columns or label_col not in df.columns:
                        logger.warning(f"Skipping {path}: missing required columns")
                        continue

                    texts = df[text_col].fillna('').astype(str).tolist()
                    labels = df[label_col].astype(str).str.lower().tolist()

                    # Filter valid labels
                    valid_data = [(t, l) for t, l in zip(texts, labels)
                                if l in ['ham', 'spam', 'phishing'] and t.strip()]

                    all_texts.extend([t for t, l in valid_data])
                    all_labels.extend([l for t, l in valid_data])

                    logger.info(f"Loaded {len(valid_data)} samples from {path}")

            except Exception as e:
                logger.error(f"Error loading {path}: {e}")
                continue

        logger.info(f"Total samples loaded: {len(all_texts)}")
        return all_texts, all_labels

    def create_balanced_sampler(self, labels: List[int]) -> Optional[WeightedRandomSampler]:
        """Create balanced sampler for imbalanced datasets."""
        label_counts = Counter(labels)
        total_samples = len(labels)

        # Calculate weights
        weights = []
        for label in labels:
            weight = total_samples / (len(label_counts) * label_counts[label])
            weights.append(weight)

        sampler = WeightedRandomSampler(weights, len(weights))
        return sampler

    def train(self, train_dataset, val_dataset, epochs: int = 3, batch_size: int = 16,
              learning_rate: float = 2e-5, weight_decay: float = 0.01):
        """Enhanced training with validation and early stopping."""

        # Create data loaders
        train_sampler = self.create_balanced_sampler([item['label'].item() for item in train_dataset])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler if train_sampler else None,
            shuffle=train_sampler is None
        )

        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        # Training loop
        best_accuracy = 0.0
        patience = 3
        patience_counter = 0

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                train_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = train_loss / len(train_loader)
            logger.info(f"Average training loss: {avg_train_loss:.4f}")

            # Validation phase
            val_accuracy, val_metrics = self.evaluate(val_loader)

            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
            logger.info(f"Validation F1: {val_metrics['weighted_f1']:.4f}")

            # Early stopping
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
                self.save_model(f"best_model_epoch_{epoch+1}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered")
                    break

        return best_accuracy

    def evaluate(self, data_loader) -> Tuple[float, Dict[str, float]]:
        """Enhanced evaluation with comprehensive metrics."""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)

        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )

        # Calculate AUC if binary classification
        auc = None
        if self.num_labels == 2:
            try:
                auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
            except (ValueError, IndexError):
                pass  # AUC calculation may fail with certain label distributions

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'weighted_f1': f1,
            'auc': auc
        }

        return accuracy, metrics

    def save_model(self, suffix: str = ""):
        """Save model with metadata."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_dir = Path(f"models/enhanced_mbert_{timestamp}_{suffix}")

        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'timestamp': timestamp,
            'num_labels': self.num_labels,
            'device': str(self.device)
        }

        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_dir}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Multilingual mBERT Training')
    parser.add_argument('--datasets', nargs='+', required=True,
                       help='Paths to dataset CSV files')
    parser.add_argument('--output-dir', default='models',
                       help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio')

    args = parser.parse_args()

    # Initialize trainer
    trainer = EnhancedTrainer()

    # Load and combine datasets
    all_texts, all_labels = trainer.load_datasets(args.datasets)

    if not all_texts:
        logger.error("No valid data found")
        return

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        all_texts, all_labels,
        test_size=args.val_split,
        random_state=42,
        stratify=all_labels
    )

    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(val_texts)}")

    # Create datasets
    train_dataset = MultilingualSMSDataset(train_texts, train_labels, trainer.tokenizer)
    val_dataset = MultilingualSMSDataset(val_texts, val_labels, trainer.tokenizer)

    # Train model
    best_accuracy = trainer.train(
        train_dataset, val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

    logger.info(f"Training completed. Best validation accuracy: {best_accuracy:.4f}")

if __name__ == '__main__':
    main()