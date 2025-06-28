"""
Pydantic request models for OpenTextShield API.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ModelType(str, Enum):
    """Supported model types."""
    OTS_MBERT = "ots-mbert"


class PredictionRequest(BaseModel):
    """Request model for text prediction."""
    
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Text to classify for spam/phishing detection"
    )
    model: ModelType = Field(
        default=ModelType.OTS_MBERT,
        description="Model type to use for prediction (OpenTextShield mBERT)"
    )
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        """Validate text input."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Congratulations! You've won $1000. Click here to claim your prize!",
                "model": "ots-mbert"
            }
        }
    }


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""
    
    content: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Original text that was classified"
    )
    feedback: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User feedback about the classification"
    )
    thumbs_up: bool = Field(
        default=False,
        description="Whether user agrees with the classification"
    )
    thumbs_down: bool = Field(
        default=False,
        description="Whether user disagrees with the classification"
    )
    user_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Optional user identifier"
    )
    model: ModelType = Field(
        ...,
        description="Model that was used for the original prediction"
    )
    
    @field_validator('content', 'feedback')
    @classmethod
    def validate_non_empty(cls, v):
        """Validate non-empty strings."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or only whitespace")
        return v.strip()
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "content": "Free money! Click here!",
                "feedback": "This was correctly identified as spam",
                "thumbs_up": True,
                "thumbs_down": False,
                "user_id": "user123",
                "model": "ots-mbert"
            }
        }
    }