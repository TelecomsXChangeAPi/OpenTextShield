"""
Incremental Learning Training Script for OpenTextShield mBERT.

This script implements Parameter-Efficient Fine-Tuning (PEFT) using LoRA
to enable incremental learning without full model retraining.

Features:
- LoRA adapters for efficient fine-tuning
- Load existing adapters for continued learning
- Minimal compute requirements
- Preserves base model weights
"""

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import chardet
import time
import logging
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from config import training_config, get_device


# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'incremental_training_{training_config.model_version}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IncrementalTrainer:
    """Trainer for incremental learning with LoRA adapters."""

    def __init__(self, config=training_config):
        self.config = config
        self.device = get_device()
        self.tokenizer = None
        self.base_model = None
        self.model = None  # PEFT model
        self.optimizer = None

        # LoRA configuration
        self.lora_config = LoraConfig(
            r=8,  # Low-rank dimension
            lora_alpha=16,
            target_modules=["query", "key", "value", "dense"],  # BERT attention layers
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS"  # Sequence classification
        )

        # Training metrics
        self.metrics = {
            "train_losses": [],
            "val_losses": [],
            "val_accuracies": [],
            "training_time": 0,
            "best_val_accuracy": 0,
            "best_epoch": 0
        }

        logger.info(f"Incremental trainer initialized with device: {self.device}")
        logger.info(f"LoRA config: r={self.lora_config.r}, alpha={self.lora_config.lora_alpha}")

    def load_and_prepare_data(self, dataset_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and prepare incremental dataset."""
        try:
            dataset_path = dataset_path or self.config.get_latest_dataset()
            with open(dataset_path, 'rb') as f:
                result = chardet.detect(f.read())
                file_encoding = result['encoding']

            logger.info(f"Loading incremental dataset from: {dataset_path}")

            try:
                df = pd.read_csv(dataset_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(dataset_path, encoding='latin1')

            # Validate structure
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset must contain 'text' and 'label' columns")

            # Clean data
            df = df.dropna(subset=['text', 'label'])
            df['label'] = df['label'].map(self.config.label_mapping)

            if df['label'].isna().sum() > 0:
                raise ValueError("Some labels could not be mapped")

            logger.info(f"Incremental dataset loaded: {len(df)} samples")
            logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

            # Split data (smaller validation for incremental)
            train_df, temp_df = train_test_split(
                df,
                test_size=0.3,  # 70% train, 30% val/test
                random_state=self.config.random_state,
                stratify=df['label']
            )

            val_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,  # 15% val, 15% test
                random_state=self.config.random_state,
                stratify=temp_df['label']
            )

            logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

            return train_df, val_df, test_df

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def create_data_loader(self, df: pd.DataFrame, batch_size: Optional[int] = None) -> DataLoader:
        """Create data loader for incremental training."""
        from torch.utils.data import Dataset

        class TextDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, item):
                text = str(self.texts[item])
                label = self.labels[item]

                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    return_token_type_ids=False,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt',
                    truncation=True
                )

                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }

        batch_size = batch_size or self.config.batch_size

        dataset = TextDataset(
            texts=df['text'].to_numpy(),
            labels=df['label'].to_numpy(),
            tokenizer=self.tokenizer,
            max_len=self.config.max_length
        )

        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    def initialize_model(self):
        """Initialize base model and PEFT adapters."""
        try:
            logger.info(f"Loading tokenizer: {self.config.model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(self.config.model_name)

            logger.info(f"Loading base model: {self.config.model_name}")
            self.base_model = BertForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels
            )

            # Check for existing trained base model
            base_model_path = self.config.get_model_save_path()
            if base_model_path.exists():
                logger.info(f"Loading existing base model weights from {base_model_path}")
                state_dict = torch.load(base_model_path, map_location=self.device, weights_only=True)
                self.base_model.load_state_dict(state_dict)
                logger.info("Base model weights loaded")
            else:
                logger.info("Using base BERT model (no existing weights found)")

            # Check for existing LoRA adapters
            adapter_path = self.config.base_dir / f"adapters_{self.config.model_version}"
            if adapter_path.exists():
                logger.info(f"Loading existing LoRA adapters from {adapter_path}")
                self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
                logger.info("Existing adapters loaded for continued training")
            else:
                logger.info("Creating new LoRA adapters")
                self.model = get_peft_model(self.base_model, self.lora_config)
                logger.info(f"Trainable parameters: {self.model.print_trainable_parameters()}")

            self.model.to(self.device)

            # Optimizer (only train LoRA parameters)
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate * 0.1  # Lower LR for adapters
            )

            logger.info("PEFT model initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def validate_model(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate PEFT model."""
        self.model.eval()
        total_loss = 0
        predictions, true_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                total_loss += outputs.loss.item()
                predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(true_labels, predictions)

        return avg_loss, accuracy

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train one epoch with LoRA."""
        self.model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs}, "
                    f"Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        return total_loss / len(train_loader)

    def train(self, dataset_path: Optional[str] = None) -> Dict[str, Any]:
        """Main incremental training loop."""
        try:
            # Load incremental data
            train_df, val_df, test_df = self.load_and_prepare_data(dataset_path)

            # Initialize PEFT model
            self.initialize_model()

            # Create data loaders
            train_loader = self.create_data_loader(train_df)
            val_loader = self.create_data_loader(val_df)
            test_loader = self.create_data_loader(test_df)

            # Training loop (fewer epochs for incremental)
            num_epochs = min(self.config.num_epochs, 5)  # Limit epochs for incremental
            start_time = time.time()
            logger.info("Starting incremental training with LoRA")

            for epoch in range(num_epochs):
                train_loss = self.train_epoch(train_loader, epoch)
                val_loss, val_accuracy = self.validate_model(val_loader)

                self.metrics["train_losses"].append(train_loss)
                self.metrics["val_losses"].append(val_loss)
                self.metrics["val_accuracies"].append(val_accuracy)

                if val_accuracy > self.metrics["best_val_accuracy"]:
                    self.metrics["best_val_accuracy"] = val_accuracy
                    self.metrics["best_epoch"] = epoch

                    # Save best adapters
                    adapter_path = self.config.base_dir / f"adapters_{self.config.model_version}_best"
                    self.model.save_pretrained(adapter_path)
                    logger.info(f"Best adapters saved: {val_accuracy:.4f} accuracy")

                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.4f}"
                )

            # Training completed
            end_time = time.time()
            self.metrics["training_time"] = end_time - start_time

            logger.info(f"Incremental training completed in {self.metrics['training_time']:.2f} seconds")
            logger.info(f"Best validation accuracy: {self.metrics['best_val_accuracy']:.4f}")

            # Final evaluation
            test_results = self.evaluate_model(test_loader)

            # Save final adapters
            final_adapter_path = self.config.base_dir / f"adapters_{self.config.model_version}"
            self.model.save_pretrained(final_adapter_path)
            logger.info(f"Final adapters saved to: {final_adapter_path}")

            # Save metadata
            self.save_training_metadata(test_results)

            return {
                "metrics": self.metrics,
                "test_results": test_results,
                "adapter_path": str(final_adapter_path)
            }

        except Exception as e:
            logger.error(f"Incremental training failed: {e}")
            raise

    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate the PEFT model."""
        logger.info("Evaluating PEFT model on test set")

        self.model.eval()
        predictions, true_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

        accuracy = accuracy_score(true_labels, predictions)
        label_names = ['ham', 'spam', 'phishing']
        class_report = classification_report(
            true_labels, predictions, target_names=label_names, output_dict=True
        )

        logger.info(f"Test Accuracy: {accuracy:.4f}")

        return {
            "accuracy": accuracy,
            "classification_report": class_report
        }

    def save_training_metadata(self, test_results: Dict[str, Any]):
        """Save incremental training metadata."""
        metadata = {
            "model_version": self.config.model_version,
            "training_type": "incremental_peft",
            "lora_config": {
                "r": self.lora_config.r,
                "lora_alpha": self.lora_config.lora_alpha,
                "target_modules": self.lora_config.target_modules,
                "lora_dropout": self.lora_config.lora_dropout
            },
            "config": self.config.to_dict(),
            "device": str(self.device),
            "training_metrics": self.metrics,
            "test_results": test_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        metadata_path = self.config.base_dir / f"incremental_training_metadata_{self.config.model_version}.json"

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Incremental training metadata saved to: {metadata_path}")


def main(dataset_path: Optional[str] = None):
    """Main incremental training function."""
    try:
        logger.info("Starting OpenTextShield incremental training with PEFT")

        trainer = IncrementalTrainer()
        results = trainer.train(dataset_path)

        logger.info("Incremental training completed successfully!")
        logger.info(f"Final test accuracy: {results['test_results']['accuracy']:.4f}")

        return True

    except Exception as e:
        logger.error(f"Incremental training failed: {e}")
        return False


if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    success = main(dataset_path)
    exit(0 if success else 1)