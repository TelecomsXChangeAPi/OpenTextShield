"""
Pydantic response models for OpenTextShield API.
"""

from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field


class ClassificationLabel(str, Enum):
    """Classification labels."""
    HAM = "ham"
    SPAM = "spam"
    PHISHING = "phishing"


class ModelInfo(BaseModel):
    """Model information."""
    
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    author: str = Field(default="TelecomsXChange (TCXC)", description="Model author")
    last_training: str = Field(..., description="Last training date")


class PredictionResponse(BaseModel):
    """Response model for text prediction."""
    
    label: ClassificationLabel = Field(..., description="Predicted classification")
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the prediction"
    )
    processing_time: float = Field(
        ...,
        ge=0.0,
        description="Processing time in seconds"
    )
    model_info: ModelInfo = Field(..., description="Information about the model used")
    
    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "label": "spam",
                "probability": 0.95,
                "processing_time": 0.15,
                "model_info": {
                    "name": "OTS_mBERT",
                    "version": "bert-base-multilingual-cased",
                    "author": "TelecomsXChange (TCXC)",
                    "last_training": "2024-03-20"
                }
            }
        }
    }


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""
    
    message: str = Field(..., description="Confirmation message")
    feedback_id: Optional[str] = Field(None, description="Feedback identifier")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "Feedback received successfully",
                "feedback_id": "feedback_123"
            }
        }
    }


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    timestamp: str = Field(..., description="Response timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "version": "2.1.0",
                "models_loaded": {
                    "mbert_multilingual": True
                },
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    }


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "VALIDATION_ERROR",
                "message": "Invalid input provided",
                "details": {"field": "text"},
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    }