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
    """Model information.

    Note the two distinct version axes: ``version`` is the *model* semantic
    version (e.g. ``2.7``, the trained weights), which is independent of the
    API/platform version reported by the health endpoint (e.g. ``2.10.0``).
    ``architecture`` names the underlying network and never carries a version.
    """

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model semantic version (e.g. '2.7') — distinct from the API version")
    architecture: str = Field(
        default="bert-base-multilingual-cased",
        description="Underlying model architecture (not a version)"
    )
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
                    "version": "2.7",
                    "architecture": "bert-base-multilingual-cased",
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


class ModelVersion(BaseModel):
    """Deployed model identity, on its own version axis.

    Kept separate from the API/platform version so consumers can tell at a
    glance which trained weights are live (``version``) independently of the
    service release (``HealthResponse.api_version``).
    """

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model semantic version (e.g. '2.7')")
    architecture: str = Field(
        default="bert-base-multilingual-cased",
        description="Underlying model architecture (not a version)"
    )


class HealthResponse(BaseModel):
    """Response model for health check.

    Exposes both version axes explicitly: ``api_version`` (the service/platform
    release, e.g. ``2.10.0``) and ``model`` (the deployed weights, e.g. ``2.7``).
    ``version`` is retained as a backward-compatible alias of ``api_version``.
    """

    status: str = Field(..., description="Service status")
    api_version: str = Field(..., description="API / platform version (e.g. '2.10.0')")
    # DEPRECATED: alias of api_version, retained for backward compatibility.
    # Planned for removal in the next major release (v3.0.0); consumers should
    # migrate to `api_version`.
    version: str = Field(
        ...,
        description="DEPRECATED alias of api_version (kept for backward compatibility; removal targeted for v3.0.0)"
    )
    model: Optional[ModelVersion] = Field(
        None, description="Deployed model identity (separate version axis)"
    )
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    timestamp: str = Field(..., description="Response timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "api_version": "2.10.0",
                "version": "2.10.0",
                "model": {
                    "name": "OTS_mBERT",
                    "version": "2.7",
                    "architecture": "bert-base-multilingual-cased"
                },
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