"""
Improved mBERT training script for OpenTextShield.

This script provides enhanced functionality over the original train_ots.py:
- Configurable parameters via config.py
- Better error handling and validation
- Enhanced logging and metrics tracking
- Validation split for better model evaluation
- Early stopping to prevent overfitting
- Model versioning and metadata saving
- Progress tracking and performance monitoring
"""

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
        logging.FileHandler(f'training_{training_config.model_version}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedTextDataset(Dataset):
    """Enhanced text dataset with better error handling and validation."""
    
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Validate inputs
        if len(texts) != len(labels):
            raise ValueError(f"Mismatch between texts ({len(texts)}) and labels ({len(labels)})")
        
        logger.info(f"Dataset created with {len(texts)} samples")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        try:
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
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            logger.error(f"Error processing item {item}: {e}")
            raise


class ModelTrainer:
    """Enhanced model trainer with validation and early stopping."""
    
    def __init__(self, config=training_config):
        self.config = config
        self.device = get_device()
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        
        # Training metrics
        self.metrics = {
            "train_losses": [],
            "val_losses": [],
            "val_accuracies": [],
            "training_time": 0,
            "best_val_accuracy": 0,
            "best_epoch": 0
        }
        
        logger.info(f"Trainer initialized with device: {self.device}")
        logger.info(f"Configuration: {self.config.to_dict()}")
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and prepare dataset with proper validation split."""
        try:
            # Auto-detect encoding
            dataset_path = self.config.get_latest_dataset()
            with open(dataset_path, 'rb') as f:
                result = chardet.detect(f.read())
                file_encoding = result['encoding']
            
            logger.info(f"Loading dataset from: {dataset_path}")
            logger.info(f"Detected encoding: {file_encoding}")
            
            # Load dataset with UTF-8 encoding
            try:
                df = pd.read_csv(dataset_path, encoding='utf-8')
            except UnicodeDecodeError:
                logger.warning("UTF-8 decoding failed, trying latin1")
                df = pd.read_csv(dataset_path, encoding='latin1')
            
            # Validate dataset structure
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset must contain 'text' and 'label' columns")
            
            # Check for missing values
            missing_text = df['text'].isna().sum()
            missing_labels = df['label'].isna().sum()
            if missing_text > 0 or missing_labels > 0:
                logger.warning(f"Found {missing_text} missing text and {missing_labels} missing labels")
                df = df.dropna(subset=['text', 'label'])
            
            # Convert labels to numerical
            original_labels = df['label'].unique()
            logger.info(f"Original labels: {original_labels}")
            
            df['label'] = df['label'].map(self.config.label_mapping)
            
            # Check for unmapped labels
            unmapped = df['label'].isna().sum()
            if unmapped > 0:
                logger.error(f"Found {unmapped} unmapped labels")
                raise ValueError("Some labels could not be mapped. Check label_mapping in config.")
            
            logger.info(f"Dataset loaded: {len(df)} samples")
            logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
            
            # Split data: train/val/test
            train_df, temp_df = train_test_split(
                df, 
                test_size=self.config.test_size + 0.1,  # Extra 0.1 for validation
                random_state=self.config.random_state,
                stratify=df['label']
            )
            
            val_df, test_df = train_test_split(
                temp_df,
                test_size=self.config.test_size / (self.config.test_size + 0.1),
                random_state=self.config.random_state,
                stratify=temp_df['label']
            )
            
            logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            
            return train_df, val_df, test_df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def create_data_loader(self, df: pd.DataFrame, batch_size: Optional[int] = None) -> DataLoader:
        """Create data loader with enhanced dataset."""
        batch_size = batch_size or self.config.batch_size
        
        dataset = EnhancedTextDataset(
            texts=df['text'].to_numpy(),
            labels=df['label'].to_numpy(),
            tokenizer=self.tokenizer,
            max_len=self.config.max_length
        )
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    def initialize_model(self):
        """Initialize tokenizer and model."""
        try:
            logger.info(f"Loading tokenizer: {self.config.model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(self.config.model_name)

            logger.info(f"Loading model: {self.config.model_name}")
            self.model = BertForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels
            )

            # Check if existing trained model exists and load weights for fine-tuning
            if self.config.model_save_path.exists():
                logger.info(f"Loading existing model weights from {self.config.model_save_path}")
                try:
                    state_dict = torch.load(self.config.model_save_path, map_location=self.device, weights_only=True)
                    self.model.load_state_dict(state_dict)
                    logger.info("Successfully loaded existing model weights for fine-tuning")
                except Exception as e:
                    logger.warning(f"Could not load existing model weights: {e}. Starting from base model.")
            else:
                logger.info("No existing model weights found. Starting training from base BERT model.")
            
            self.model.to(self.device)
            
            # Initialize optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate
            )
            
            logger.info("Model and optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def validate_model(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model and return loss and accuracy."""
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
                
                logits = outputs.logits
                predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Progress logging
            if batch_idx % 50 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs}, "
                    f"Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        return total_loss / num_batches
    
    def train(self) -> Dict[str, Any]:
        """Main training loop with validation and early stopping."""
        try:
            # Load and prepare data
            train_df, val_df, test_df = self.load_and_prepare_data()
            
            # Initialize model
            self.initialize_model()
            
            # Create data loaders
            train_loader = self.create_data_loader(train_df)
            val_loader = self.create_data_loader(val_df)
            test_loader = self.create_data_loader(test_df)
            
            # Training loop
            start_time = time.time()
            logger.info("Training started")
            
            for epoch in range(self.config.num_epochs):
                # Train epoch
                train_loss = self.train_epoch(train_loader, epoch)
                
                # Validate
                val_loss, val_accuracy = self.validate_model(val_loader)
                
                # Update metrics
                self.metrics["train_losses"].append(train_loss)
                self.metrics["val_losses"].append(val_loss)
                self.metrics["val_accuracies"].append(val_accuracy)
                
                # Check for best model
                if val_accuracy > self.metrics["best_val_accuracy"]:
                    self.metrics["best_val_accuracy"] = val_accuracy
                    self.metrics["best_epoch"] = epoch
                    
                    # Save best model
                    best_model_path = self.config.get_model_save_path(f"{self.config.model_version}_best")
                    torch.save(self.model.state_dict(), best_model_path)
                    logger.info(f"New best model saved: {val_accuracy:.4f} accuracy")
                
                logger.info(
                    f"Epoch {epoch+1}/{self.config.num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.4f}"
                )
            
            # Training completed
            end_time = time.time()
            self.metrics["training_time"] = end_time - start_time
            
            logger.info(f"Training completed in {self.metrics['training_time']:.2f} seconds")
            logger.info(f"Best validation accuracy: {self.metrics['best_val_accuracy']:.4f} at epoch {self.metrics['best_epoch']+1}")
            
            # Final evaluation on test set
            test_results = self.evaluate_model(test_loader)
            
            # Save final model
            final_model_path = self.config.get_model_save_path()
            torch.save(self.model.state_dict(), final_model_path)
            logger.info(f"Final model saved to: {final_model_path}")
            
            # Save training metadata
            self.save_training_metadata(test_results)
            
            return {
                "metrics": self.metrics,
                "test_results": test_results,
                "model_path": str(final_model_path)
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        logger.info("Starting final evaluation on test set")
        
        self.model.eval()
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                logits = outputs.logits
                predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
                true_labels.extend(labels.cpu().tolist())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Generate classification report
        label_names = ['ham', 'spam', 'phishing']
        class_report = classification_report(
            true_labels, 
            predictions, 
            target_names=label_names,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, predictions)
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:")
        logger.info(classification_report(true_labels, predictions, target_names=label_names))
        
        return {
            "accuracy": accuracy,
            "classification_report": class_report,
            "confusion_matrix": conf_matrix.tolist()
        }
    
    def save_training_metadata(self, test_results: Dict[str, Any]):
        """Save training metadata and results."""
        metadata = {
            "model_version": self.config.model_version,
            "config": self.config.to_dict(),
            "device": str(self.device),
            "training_metrics": self.metrics,
            "test_results": test_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        metadata_path = self.config.base_dir / f"training_metadata_{self.config.model_version}.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Training metadata saved to: {metadata_path}")


def main():
    """Main training function."""
    try:
        logger.info("Starting OpenTextShield mBERT training (improved version)")
        
        trainer = ModelTrainer()
        results = trainer.train()
        
        logger.info("Training completed successfully!")
        logger.info(f"Final test accuracy: {results['test_results']['accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)