"""
Configuration management for mBERT MLX Apple Silicon training.
Centralizes all hardcoded values for better maintainability.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


class TrainingConfig:
    """Configuration class for training parameters and paths."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        
        # Dataset Configuration
        self.dataset_dir = self.base_dir / "dataset"
        self.current_dataset = "sms_spam_phishing_dataset_v2.1.csv"
        self.dataset_path = self.dataset_dir / self.current_dataset
        
        # Model Configuration
        self.model_name = "bert-base-multilingual-cased"
        self.num_labels = 3
        self.max_length = 128
        
        # Training Parameters
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.num_epochs = 3
        self.test_size = 0.2
        self.random_state = 42
        
        # Model Save Configuration
        self.model_version = "2.1"
        self.model_filename = f"mbert_ots_model_{self.model_version}.pth"
        self.model_save_path = self.base_dir / self.model_filename
        
        # MLX Configuration
        self.mlx_model_path = self.base_dir / "converted_bert.npz"
        
        # Label Mapping
        self.label_mapping = {
            'ham': 0,
            'spam': 1,
            'phishing': 2
        }
        
        # Device Configuration (Apple Silicon optimization)
        self.device_preference = ["mps", "cuda", "cpu"]  # Preferred order
        
    def get_latest_dataset(self) -> Path:
        """Get the path to the latest dataset file."""
        if self.dataset_path.exists():
            return self.dataset_path
        
        # Fallback to latest available dataset
        dataset_files = sorted(self.dataset_dir.glob("sms_spam_phishing_dataset_v*.csv"))
        if dataset_files:
            return dataset_files[-1]  # Return the latest version
        
        raise FileNotFoundError(f"No dataset files found in {self.dataset_dir}")
    
    def get_model_save_path(self, version: Optional[str] = None) -> Path:
        """Get model save path, optionally with custom version."""
        if version:
            filename = f"mbert_ots_model_{version}.pth"
            return self.base_dir / filename
        return self.model_save_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for logging/debugging."""
        return {
            "dataset_path": str(self.dataset_path),
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "model_save_path": str(self.model_save_path)
        }


class TranslationConfig:
    """Configuration for dataset translation tasks."""
    
    def __init__(self):
        # OpenAI Configuration (use environment variables for security)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = "gpt-3.5-turbo"
        
        # Translation Parameters
        self.target_language = "Spanish"  # Can be made configurable
        self.batch_size = 1  # For rate limiting
        self.delay_between_requests = 1.0  # seconds
        
        # File paths
        self.base_dir = Path(__file__).parent
        self.input_dataset = self.base_dir / "dataset" / "sms_spam_phishing_dataset_v2.1.csv"
        self.output_dataset = self.base_dir / f"translated_dataset_v2.1_{self.target_language.upper()}.csv"
        
    def validate(self) -> bool:
        """Validate translation configuration."""
        if not self.openai_api_key:
            print("Warning: OPENAI_API_KEY environment variable not set")
            return False
        
        if not self.input_dataset.exists():
            print(f"Error: Input dataset not found: {self.input_dataset}")
            return False
        
        return True


# Global configuration instances
training_config = TrainingConfig()
translation_config = TranslationConfig()


def get_device():
    """Get the best available device for training/inference."""
    import torch
    
    for device_name in training_config.device_preference:
        if device_name == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif device_name == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        elif device_name == "cpu":
            return torch.device("cpu")
    
    return torch.device("cpu")  # Fallback