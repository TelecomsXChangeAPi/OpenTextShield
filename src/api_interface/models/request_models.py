"""
Pydantic request models for OpenTextShield API.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ModelType(str, Enum):
    """Supported model types."""
    OTS_MBERT = "ots-mbert"


class BatchPredictionRequest(BaseModel):
    """
    Request model for batched text prediction.

    Use this endpoint when you can batch messages client-side (e.g. from an
    SMPP proxy collecting messages over a short window). A single HTTP request
    classifies all provided texts in one GPU forward pass, eliminating the
    server-side batch-wait penalty and the client-side connection pool
    pressure that comes with one HTTP request per message.
    """

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=256,
        description=(
            "Batch of texts to classify. Hard cap of 256 to keep a single "
            "GPU pass bounded; tune max_batch_size upstream if you want to "
            "raise this."
        ),
    )
    model: ModelType = Field(
        default=ModelType.OTS_MBERT,
        description="Model to use. Applied to every text in the batch.",
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v):
        """Reject empty strings and trim whitespace."""
        cleaned = []
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"texts[{i}] cannot be empty or whitespace-only")
            if len(text) > 512:
                raise ValueError(f"texts[{i}] exceeds 512 character limit")
            cleaned.append(text.strip())
        return cleaned

    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": [
                    "Hey, want to grab lunch tomorrow?",
                    "FREE iPhone! Click http://scam.ly/win",
                    "URGENT verify your account http://fake-bank.com",
                ],
                "model": "ots-mbert",
            }
        }
    }


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