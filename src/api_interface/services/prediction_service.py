"""
Prediction service for OpenTextShield API.
"""

import time
import torch
import numpy as np
from typing import Tuple, Dict, Any

from ..config.settings import settings
from ..utils.logging import logger
from ..utils.exceptions import PredictionError, ModelNotFoundError
from ..models.request_models import PredictionRequest, ModelType
from ..models.response_models import PredictionResponse, ModelInfo, ClassificationLabel
from .model_loader import model_manager


class PredictionService:
    """Service for handling text classification predictions."""
    
    def __init__(self):
        self.label_map = {0: 'ham', 1: 'spam', 2: 'phishing'}
    
    def preprocess_text(self, text: str, tokenizer: Any, max_len: int = None) -> Dict[str, torch.Tensor]:
        """
        Preprocess text for BERT model input.
        
        Args:
            text: Input text
            tokenizer: BERT tokenizer
            max_len: Maximum sequence length
            
        Returns:
            Tokenized and encoded text
        """
        max_length = max_len or settings.max_text_length
        
        return tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
    
    def predict_with_mbert(
        self,
        text: str,
        model_name: str
    ) -> Tuple[str, float, float, ModelInfo]:
        """
        Perform prediction using mBERT model.
        
        Args:
            text: Text to classify
            model_name: Name of the mBERT model to use
            
        Returns:
            Tuple of (label, probability, processing_time, model_info)
        """
        start_time = time.time()
        
        try:
            # Get model and tokenizer
            model, tokenizer = model_manager.get_mbert_model(model_name)
            
            # Preprocess text
            inputs = self.preprocess_text(text, tokenizer)
            inputs = {k: v.to(model_manager.device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
                probability = torch.nn.functional.softmax(outputs.logits, dim=1).max().item()
            
            # Map prediction to label
            label = self.label_map[prediction]
            
            processing_time = time.time() - start_time
            
            model_info = ModelInfo(
                name="OTS_mBERT",
                version="2.1",
                author="TelecomsXChange (TCXC)",
                last_training="2024-03-20"
            )
            
            logger.info(
                f"mBERT prediction completed: {label} "
                f"(confidence: {probability:.3f}, time: {processing_time:.3f}s)"
            )
            
            return label, probability, processing_time, model_info
            
        except Exception as e:
            logger.error(f"mBERT prediction failed: {str(e)}")
            raise PredictionError({"model": model_name, "error": str(e)})
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Perform text classification prediction.
        
        Args:
            request: Prediction request
            
        Returns:
            Prediction response
            
        Raises:
            PredictionError: If prediction fails
            ModelNotFoundError: If requested model is not available
        """
        logger.info(f"Processing prediction request: model={request.model}, text_length={len(request.text)}")
        
        try:
            if request.model == ModelType.OTS_MBERT:
                # Use default mBERT version
                mbert_version = settings.default_mbert_version
                
                # Check if model is available
                if not model_manager.is_model_available("mbert", mbert_version):
                    available_models = list(model_manager.mbert_models.keys())
                    raise ModelNotFoundError(
                        f"OpenTextShield mBERT model '{mbert_version}' not available. "
                        f"Available models: {available_models}"
                    )
                
                label, probability, processing_time, model_info = self.predict_with_mbert(
                    request.text, mbert_version
                )
            
            else:
                raise PredictionError({"error": f"Unsupported model type: {request.model}"})
            
            # Create response
            response = PredictionResponse(
                label=ClassificationLabel(label),
                probability=probability,
                processing_time=processing_time,
                model_info=model_info
            )
            
            return response
            
        except (ModelNotFoundError, PredictionError):
            # Re-raise known exceptions
            raise
        except Exception as e:
            logger.error(f"Unexpected error during prediction: {str(e)}")
            raise PredictionError({"error": str(e)})


# Global prediction service instance
prediction_service = PredictionService()