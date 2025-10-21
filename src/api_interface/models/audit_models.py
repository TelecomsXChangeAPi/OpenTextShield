"""
Pydantic models for audit log entries.

Provides type-safe models for all audit entry types with validation.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AuditEntryType(str, Enum):
    """Types of audit log entries."""

    PREDICTION = "prediction"
    FEEDBACK = "feedback"
    ACCESS_DENIED = "access_denied"
    SYSTEM_EVENT = "system_event"


class BaseAuditEntry(BaseModel):
    """Base model for all audit entries."""

    timestamp: str = Field(..., description="ISO 8601 timestamp in UTC")
    entry_type: AuditEntryType = Field(..., description="Type of audit entry")
    client_ip: str = Field(..., description="Client IP address")


class PredictionAuditEntry(BaseAuditEntry):
    """Audit entry for classification predictions."""

    entry_type: AuditEntryType = AuditEntryType.PREDICTION
    text: str = Field(..., description="Text that was classified (full or processed)")
    text_length: int = Field(..., ge=0, description="Length of original text")
    text_hash: str = Field(..., description="SHA-256 hash of original text")
    label: str = Field(..., description="Classification label (ham, spam, phishing)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    model: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    processing_time_ms: float = Field(..., ge=0, description="Processing time in milliseconds")
    redaction_applied: bool = Field(False, description="Whether PII redaction was applied")
    redacted_entities: Optional[List[str]] = Field(
        None, description="List of entity types that were redacted"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-10-20T14:30:45.123456Z",
                "entry_type": "prediction",
                "client_ip": "192.168.1.100",
                "text": "Congratulations! You've won $1000...",
                "text_length": 75,
                "text_hash": "sha256:a3f5b2c1...",
                "label": "spam",
                "confidence": 0.9523,
                "model": "ots-mbert",
                "model_version": "2.5",
                "processing_time_ms": 152.34,
                "redaction_applied": False,
            }
        }


class FeedbackAuditEntry(BaseAuditEntry):
    """Audit entry for user feedback submissions."""

    entry_type: AuditEntryType = AuditEntryType.FEEDBACK
    feedback_id: str = Field(..., description="Unique feedback identifier")
    text: str = Field(..., description="Original text that was classified")
    text_hash: str = Field(..., description="SHA-256 hash of original text")
    original_label: str = Field(..., description="Original classification label")
    user_feedback: str = Field(..., description="User's feedback text")
    thumbs_up: bool = Field(..., description="Whether user agreed with classification")
    thumbs_down: bool = Field(..., description="Whether user disagreed with classification")
    model: str = Field(..., description="Model that was used for classification")
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    redaction_applied: bool = Field(False, description="Whether PII redaction was applied")
    redacted_entities: Optional[List[str]] = Field(None, description="Redacted entity types")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-10-20T14:31:12.456789Z",
                "entry_type": "feedback",
                "client_ip": "192.168.1.100",
                "feedback_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "text": "Free money! Click here!",
                "text_hash": "sha256:b1c2d3e4...",
                "original_label": "spam",
                "user_feedback": "Correctly identified",
                "thumbs_up": True,
                "thumbs_down": False,
                "model": "ots-mbert",
                "user_id": "operator_123",
                "redaction_applied": False,
            }
        }


class AccessDeniedAuditEntry(BaseAuditEntry):
    """Audit entry for security events (access denied, unauthorized)."""

    entry_type: AuditEntryType = AuditEntryType.ACCESS_DENIED
    endpoint: str = Field(..., description="API endpoint that was accessed")
    reason: str = Field(..., description="Reason for access denial")
    attempted_action: Optional[str] = Field(None, description="HTTP method or action attempted")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-10-20T14:32:00.789012Z",
                "entry_type": "access_denied",
                "client_ip": "203.0.113.45",
                "endpoint": "/predict/",
                "reason": "IP address not in allowlist",
                "attempted_action": "POST",
            }
        }


class SystemEventAuditEntry(BaseAuditEntry):
    """Audit entry for system events (startup, shutdown, config changes)."""

    entry_type: AuditEntryType = AuditEntryType.SYSTEM_EVENT
    client_ip: str = Field("system", description="Always 'system' for system events")
    event_type: str = Field(
        ..., description="Event type (startup, shutdown, model_load, error, etc.)"
    )
    message: str = Field(..., description="Event message")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-10-20T08:00:00.000000Z",
                "entry_type": "system_event",
                "client_ip": "system",
                "event_type": "startup",
                "message": "OpenTextShield API started (v2.5.0)",
                "metadata": {
                    "host": "0.0.0.0",
                    "port": 8002,
                    "audit_enabled": True,
                    "audit_text_storage": "full",
                },
            }
        }


# Union type for API responses
AuditEntry = (
    PredictionAuditEntry
    | FeedbackAuditEntry
    | AccessDeniedAuditEntry
    | SystemEventAuditEntry
)
