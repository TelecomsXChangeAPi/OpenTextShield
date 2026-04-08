"""
Prediction service for OpenTextShield API.

Uses asyncio.run_in_executor to offload synchronous model inference
to a thread pool, keeping the FastAPI event loop responsive for
health checks and concurrent request handling.
"""

import asyncio
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, Any

from ..config.settings import settings
from ..utils.logging import logger
from ..utils.exceptions import PredictionError, ModelNotFoundError
from ..models.request_models import PredictionRequest, ModelType
from ..models.response_models import PredictionResponse, ModelInfo, ClassificationLabel
from .model_loader import model_manager

# Import enhanced preprocessor
try:
    from .enhanced_preprocessing import EnhancedPreprocessor
    enhanced_preprocessor = EnhancedPreprocessor()
    USE_ENHANCED_PREPROCESSING = True
except ImportError:
    logger.warning("Enhanced preprocessor not available, using standard preprocessing")
    USE_ENHANCED_PREPROCESSING = False

# Thread pool for inference — sized to allow concurrent predictions
# without over-subscribing GPU/CPU resources
_inference_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ots-inference")


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

        return tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

    def _predict_with_mbert_sync(
        self,
        text: str,
        model_name: str
    ) -> Tuple[str, float, float, ModelInfo]:
        """
        Perform prediction using mBERT model (synchronous, runs in thread pool).

        Args:
            text: Text to classify
            model_name: Name of the mBERT model to use

        Returns:
            Tuple of (label, probability, processing_time, model_info)
        """
        start_time = time.time()

        try:
            # Get model, tokenizer, and version
            model, tokenizer, model_version = model_manager.get_mbert_model(model_name)

            # Enhanced preprocessing if available
            if USE_ENHANCED_PREPROCESSING:
                processed_text, features = enhanced_preprocessor.preprocess_text(text)
                logger.info(f"Enhanced preprocessing features: {features}")
            else:
                processed_text = text

            # Preprocess text
            inputs = self.preprocess_text(processed_text, tokenizer)
            inputs = {k: v.to(model_manager.device) for k, v in inputs.items()}

            # Make prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                prediction = torch.argmax(logits, dim=1).item()
                # Get the probability of the predicted class
                probability = probabilities[0][prediction].item()

            # Map prediction to label
            label = self.label_map[prediction]

            processing_time = time.time() - start_time

            model_info = ModelInfo(
                name="OTS_mBERT",
                version=model_version,  # Use version from model manager
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

        Offloads the synchronous model inference to a thread pool executor
        so the event loop remains responsive for health checks and other
        concurrent requests.

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

                # Run synchronous inference in thread pool to avoid blocking event loop
                loop = asyncio.get_running_loop()
                label, probability, processing_time, model_info = await loop.run_in_executor(
                    _inference_executor,
                    self._predict_with_mbert_sync,
                    request.text,
                    mbert_version,
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