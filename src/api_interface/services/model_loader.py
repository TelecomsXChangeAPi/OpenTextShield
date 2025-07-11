"""
Model loading and management for OpenTextShield API.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from transformers import AutoTokenizer, BertForSequenceClassification, BertConfig

from ..config.settings import settings
from ..utils.logging import logger
from ..utils.exceptions import ModelLoadError, ModelNotFoundError


class ModelManager:
    """Manages loading and access to ML models."""
    
    def __init__(self):
        self.mbert_models: Dict[str, torch.nn.Module] = {}
        self.mbert_tokenizers: Dict[str, Any] = {}
        self.device = self._detect_device()
        
    def _detect_device(self) -> torch.device:
        """Detect the best available device for PyTorch."""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA GPU")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        
        return device
    
    def load_mbert_models(self) -> None:
        """Load all configured mBERT models."""
        logger.info("Loading mBERT models...")
        
        for model_name, config in settings.mbert_model_configs.items():
            try:
                model_path = settings.models_base_path / config["path"]
                
                if not model_path.exists():
                    logger.warning(f"Model file not found: {model_path}")
                    continue
                
                # Load configuration
                bert_config = BertConfig.from_pretrained(
                    config["tokenizer"],
                    num_labels=int(config["num_labels"])
                )
                
                # Create and load model
                model = BertForSequenceClassification(bert_config)
                model.load_state_dict(
                    torch.load(model_path, map_location=self.device, weights_only=True)
                )
                model.eval()
                model = model.to(self.device)
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"])
                
                self.mbert_models[model_name] = model
                self.mbert_tokenizers[model_name] = tokenizer
                
                logger.info(f"Successfully loaded mBERT model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load mBERT model {model_name}: {str(e)}")
                raise ModelLoadError(model_name, {"error": str(e)})
    
    def load_all_models(self) -> None:
        """Load all models."""
        self.load_mbert_models()
        
        logger.info("Model loading completed")
    
    def get_mbert_model(self, model_name: str) -> Tuple[torch.nn.Module, Any]:
        """
        Get mBERT model and tokenizer.
        
        Args:
            model_name: Name of the mBERT model
            
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            ModelNotFoundError: If model is not loaded
        """
        if model_name not in self.mbert_models:
            available_models = list(self.mbert_models.keys())
            raise ModelNotFoundError(
                f"mBERT model '{model_name}' not available. "
                f"Available models: {available_models}"
            )
        
        return self.mbert_models[model_name], self.mbert_tokenizers[model_name]
    
    def is_model_available(self, model_type: str, model_name: Optional[str] = None) -> bool:
        """
        Check if a model is available.
        
        Args:
            model_type: Type of model ('mbert' only)
            model_name: Specific model name (for mBERT models)
            
        Returns:
            True if model is available, False otherwise
        """
        if model_type == "mbert":
            return model_name in self.mbert_models if model_name else bool(self.mbert_models)
        
        return False
    
    def get_model_status(self) -> Dict[str, bool]:
        """Get status of all models."""
        status = {}
        
        for model_name in settings.mbert_model_configs.keys():
            status[f"mbert_{model_name}"] = model_name in self.mbert_models
        
        return status


# Global model manager instance
model_manager = ModelManager()