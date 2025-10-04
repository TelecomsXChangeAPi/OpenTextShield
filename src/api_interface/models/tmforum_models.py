"""
TMForum-compliant models for AI Inference Job Management API (TMF922).

This implements a TMForum standard API for AI inference services,
specifically designed for text classification tasks.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class TMForumInferenceJobState(str, Enum):
    """TMForum AI Inference Job states."""
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "inProgress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TMForumInferenceJobPriority(str, Enum):
    """TMForum AI Inference Job priorities."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class TMForumInferenceInput(BaseModel):
    """TMForum AI Inference Job input specification."""
    inputType: str = Field(..., description="Type of input data (e.g., 'text', 'image')")
    inputFormat: str = Field(..., description="Format of input data")
    inputData: Dict[str, Any] = Field(..., description="Input data payload")
    inputMetadata: Optional[Dict[str, Any]] = Field(None, description="Additional input metadata")


class TMForumInferenceOutput(BaseModel):
    """TMForum AI Inference Job output specification."""
    outputType: str = Field(..., description="Type of output data")
    outputFormat: str = Field(..., description="Format of output data")
    outputData: Dict[str, Any] = Field(..., description="Output data payload")
    outputMetadata: Optional[Dict[str, Any]] = Field(None, description="Additional output metadata")
    confidence: Optional[float] = Field(None, description="Confidence score (0.0-1.0)")


class TMForumInferenceModel(BaseModel):
    """TMForum AI Model specification."""
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    type: str = Field(..., description="Model type (e.g., 'bert', 'transformer')")
    description: Optional[str] = Field(None, description="Model description")
    capabilities: List[str] = Field(default_factory=list, description="Model capabilities")


class TMForumInferenceJob(BaseModel):
    """TMForum AI Inference Job resource."""
    id: Optional[str] = Field(None, description="Unique identifier of the job")
    href: Optional[str] = Field(None, description="Hyperlink reference")
    state: TMForumInferenceJobState = Field(..., description="Current state of the job")
    priority: TMForumInferenceJobPriority = Field(TMForumInferenceJobPriority.NORMAL, description="Job priority")

    # Input/Output
    input: TMForumInferenceInput = Field(..., description="Job input specification")
    output: Optional[TMForumInferenceOutput] = Field(None, description="Job output (when completed)")

    # Model specification
    model: TMForumInferenceModel = Field(..., description="AI model to use for inference")

    # Timing
    creationDate: Optional[datetime] = Field(None, description="Job creation timestamp")
    startDate: Optional[datetime] = Field(None, description="Job start timestamp")
    completionDate: Optional[datetime] = Field(None, description="Job completion timestamp")

    # Metadata
    name: Optional[str] = Field(None, description="Human-readable job name")
    description: Optional[str] = Field(None, description="Job description")
    category: Optional[str] = Field(None, description="Job category")
    externalId: Optional[str] = Field(None, description="External system identifier")

    # Processing details
    processingTimeMs: Optional[int] = Field(None, description="Processing time in milliseconds")
    errorMessage: Optional[str] = Field(None, description="Error message if job failed")

    # TMForum standard fields
    baseType: str = Field("AIInferenceJob", description="Base type of the resource")
    type: str = Field("TextClassificationInferenceJob", description="Type of the resource")
    schemaLocation: Optional[str] = Field(None, description="Schema location")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "inference-job-123",
                "state": "completed",
                "priority": "normal",
                "input": {
                    "inputType": "text",
                    "inputFormat": "plain",
                    "inputData": {
                        "text": "Congratulations! You've won $1000. Click here to claim!"
                    }
                },
                "output": {
                    "outputType": "classification",
                    "outputFormat": "json",
                    "outputData": {
                        "label": "spam",
                        "probability": 0.95
                    },
                    "confidence": 0.95
                },
                "model": {
                    "id": "ots-mbert",
                    "name": "OpenTextShield mBERT",
                    "version": "2.1",
                    "type": "bert",
                    "capabilities": ["text-classification", "multilingual"]
                },
                "creationDate": "2024-01-15T10:30:00Z",
                "completionDate": "2024-01-15T10:30:15Z",
                "processingTimeMs": 150,
                "name": "SMS Spam Classification",
                "type": "TextClassificationInferenceJob"
            }
        }
    }


class TMForumInferenceJobCreate(BaseModel):
    """Request model for creating TMForum AI Inference Job."""
    priority: TMForumInferenceJobPriority = Field(TMForumInferenceJobPriority.NORMAL, description="Job priority")
    input: TMForumInferenceInput = Field(..., description="Job input specification")
    model: TMForumInferenceModel = Field(..., description="AI model to use for inference")
    name: Optional[str] = Field(None, description="Human-readable job name")
    description: Optional[str] = Field(None, description="Job description")
    category: Optional[str] = Field(None, description="Job category")
    externalId: Optional[str] = Field(None, description="External system identifier")

    model_config = {
        "json_schema_extra": {
            "example": {
                "priority": "normal",
                "input": {
                    "inputType": "text",
                    "inputFormat": "plain",
                    "inputData": {
                        "text": "Free money! Click here now!"
                    }
                },
                "model": {
                    "id": "ots-mbert",
                    "name": "OpenTextShield mBERT",
                    "version": "2.1",
                    "type": "bert",
                    "capabilities": ["text-classification", "multilingual"]
                },
                "name": "SMS Classification Request",
                "description": "Classify SMS message for spam/phishing detection"
            }
        }
    }


class TMForumInferenceJobUpdate(BaseModel):
    """Request model for updating TMForum AI Inference Job."""
    state: Optional[TMForumInferenceJobState] = Field(None, description="Update job state")
    priority: Optional[TMForumInferenceJobPriority] = Field(None, description="Update job priority")
    name: Optional[str] = Field(None, description="Update job name")
    description: Optional[str] = Field(None, description="Update job description")


class TMForumInferenceJobList(BaseModel):
    """Response model for listing TMForum AI Inference Jobs."""
    jobs: List[TMForumInferenceJob] = Field(..., description="List of inference jobs")


class TMForumError(BaseModel):
    """TMForum standard error response."""
    code: str = Field(..., description="Error code")
    reason: str = Field(..., description="Error reason")
    message: str = Field(..., description="Error message")
    status: str = Field(..., description="HTTP status")
    referenceError: Optional[str] = Field(None, description="Reference to error details")
    baseType: str = Field("Error", description="Base type")
    type: str = Field("TMForumError", description="Error type")