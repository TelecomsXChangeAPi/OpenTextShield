"""
TMForum-compliant Event Management API (TMF688) models.

This implements TMForum standard TMF688 Event Management API for audit logging,
providing enterprise-grade event/audit trail capabilities for telecom integration.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class TMForumEventType(str, Enum):
    """TMF688 Event types for audit logging."""
    PREDICTION = "PredictionEvent"
    FEEDBACK = "FeedbackEvent"
    SECURITY = "SecurityEvent"
    SYSTEM = "SystemEvent"


class TMForumEventPriority(str, Enum):
    """TMF688 Event priorities."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TMForumEventState(str, Enum):
    """TMF688 Event states."""
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    PROCESSED = "processed"


class Characteristic(BaseModel):
    """TMForum Characteristic for flexible key-value pairs."""
    name: str = Field(..., description="Name of the characteristic")
    value: str = Field(..., description="Value of the characteristic")
    valueType: Optional[str] = Field(None, description="Data type of the value")

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "confidence",
                "value": "0.9997",
                "valueType": "float"
            }
        }
    }


class RelatedParty(BaseModel):
    """TMForum RelatedParty reference."""
    id: str = Field(..., description="Unique identifier of the related party")
    role: Optional[str] = Field(None, description="Role of the related party")
    name: Optional[str] = Field(None, description="Name of the related party")
    referredType: str = Field(..., alias="@referredType", description="Type of the related party")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "id": "127.0.0.1",
                "role": "client",
                "@referredType": "IPAddress"
            }
        }
    }


class RelatedEntity(BaseModel):
    """TMForum RelatedEntity reference."""
    id: str = Field(..., description="Unique identifier of the related entity")
    href: Optional[str] = Field(None, description="Reference to the related entity")
    name: Optional[str] = Field(None, description="Name of the related entity")
    referredType: str = Field(..., alias="@referredType", description="Type of the related entity")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "id": "OTS_mBERT",
                "name": "OpenTextShield mBERT v2.5",
                "@referredType": "AIModel"
            }
        }
    }


class RelatedObject(BaseModel):
    """TMForum RelatedObject reference."""
    id: str = Field(..., description="Unique identifier of the related object")
    involvement: Optional[str] = Field(None, description="Type of involvement")
    referredType: str = Field(..., alias="@referredType", description="Type of the related object")

    model_config = {
        "populate_by_name": True
    }


class TMForumEvent(BaseModel):
    """TMForum TMF688 Event resource for audit logging."""

    # TMForum standard identifiers
    id: str = Field(..., description="Unique event identifier (UUID)")
    href: str = Field(..., description="Hyperlink reference to this event")

    # Event identification
    eventId: str = Field(..., description="Business event identifier")
    eventType: TMForumEventType = Field(..., description="Type of event")

    # Temporal information
    eventTime: datetime = Field(..., description="When event occurred")
    timeOccurred: datetime = Field(..., description="Actual occurrence time")

    # Event details
    title: str = Field(..., description="Human-readable event title")
    description: Optional[str] = Field(None, description="Event description")
    priority: TMForumEventPriority = Field(TMForumEventPriority.MEDIUM, description="Event priority")
    state: TMForumEventState = Field(TMForumEventState.ACKNOWLEDGED, description="Event state")

    # Correlation and tracking
    correlationId: Optional[str] = Field(None, description="Correlation identifier for related events")

    # Event source and reporting
    source: RelatedParty = Field(..., description="Event source (client IP)")
    reportingSystem: RelatedEntity = Field(..., description="System that reported the event")

    # Flexible characteristics
    characteristic: List[Characteristic] = Field(default_factory=list, description="Event characteristics")

    # Related entities
    relatedParty: Optional[List[RelatedParty]] = Field(None, description="Related parties")
    relatedObject: Optional[List[RelatedObject]] = Field(None, description="Related objects")

    # TMForum standard fields
    baseType: str = Field("Event", alias="@baseType", description="Base type of the resource")
    type: str = Field(..., alias="@type", description="Specific type of the event")
    schemaLocation: Optional[str] = Field(None, alias="@schemaLocation", description="Schema location")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "id": "evt-550e8400-e29b-41d4-a716-446655440000",
                "href": "/tmf-api/event/evt-550e8400-e29b-41d4-a716-446655440000",
                "eventId": "pred-2025-10-21-001",
                "eventType": "PredictionEvent",
                "eventTime": "2025-10-21T04:31:31.007Z",
                "timeOccurred": "2025-10-21T04:31:31.007Z",
                "title": "Text Classification Prediction",
                "description": "SMS message classified as spam",
                "priority": "medium",
                "state": "acknowledged",
                "source": {
                    "id": "127.0.0.1",
                    "role": "client",
                    "@referredType": "IPAddress"
                },
                "reportingSystem": {
                    "id": "OTS_mBERT",
                    "name": "OpenTextShield mBERT v2.5",
                    "@referredType": "AIModel"
                },
                "characteristic": [
                    {"name": "label", "value": "spam", "valueType": "string"},
                    {"name": "confidence", "value": "0.9997", "valueType": "float"}
                ],
                "@baseType": "Event",
                "@type": "TextClassificationPredictionEvent"
            }
        }
    }


class TMForumEventCreate(BaseModel):
    """Request model for creating TMForum Event."""
    eventType: TMForumEventType = Field(..., description="Type of event")
    title: str = Field(..., description="Human-readable event title")
    description: Optional[str] = Field(None, description="Event description")
    priority: TMForumEventPriority = Field(TMForumEventPriority.MEDIUM, description="Event priority")
    source: RelatedParty = Field(..., description="Event source")
    reportingSystem: RelatedEntity = Field(..., description="Reporting system")
    characteristic: List[Characteristic] = Field(default_factory=list, description="Event characteristics")
    correlationId: Optional[str] = Field(None, description="Correlation identifier")

    model_config = {
        "json_schema_extra": {
            "example": {
                "eventType": "PredictionEvent",
                "title": "Manual Audit Entry",
                "description": "Manually created audit event",
                "priority": "low",
                "source": {
                    "id": "192.168.1.100",
                    "role": "client",
                    "@referredType": "IPAddress"
                },
                "reportingSystem": {
                    "id": "ManualEntry",
                    "@referredType": "System"
                },
                "characteristic": [
                    {"name": "reason", "value": "Manual correction", "valueType": "string"}
                ]
            }
        }
    }


class TMForumEventList(BaseModel):
    """Paginated list of TMForum Events."""
    events: List[TMForumEvent] = Field(default_factory=list, description="List of events")
    totalCount: int = Field(..., description="Total number of events matching the query")
    hasMore: bool = Field(..., description="Whether there are more results available")
    next: Optional[str] = Field(None, description="Link to next page of results")

    model_config = {
        "json_schema_extra": {
            "example": {
                "events": [],
                "totalCount": 150,
                "hasMore": True,
                "next": "/tmf-api/event?offset=100&limit=100"
            }
        }
    }


class TMForumEventStatistics(BaseModel):
    """TMForum Event statistics resource."""
    totalEvents: int = Field(..., description="Total number of events")
    eventsByType: Dict[str, int] = Field(default_factory=dict, description="Events grouped by type")
    eventsByPriority: Dict[str, int] = Field(default_factory=dict, description="Events grouped by priority")
    eventsByModel: Optional[Dict[str, int]] = Field(None, description="Events grouped by model")
    averageConfidence: Optional[float] = Field(None, description="Average confidence for predictions")
    uniqueSourceCount: int = Field(..., description="Number of unique event sources")
    timeRange: Optional[Dict[str, str]] = Field(None, description="Time range of events")

    model_config = {
        "json_schema_extra": {
            "example": {
                "totalEvents": 1523,
                "eventsByType": {
                    "PredictionEvent": 1450,
                    "FeedbackEvent": 52,
                    "SecurityEvent": 15,
                    "SystemEvent": 6
                },
                "eventsByPriority": {
                    "low": 1458,
                    "medium": 50,
                    "high": 15,
                    "critical": 0
                },
                "eventsByModel": {
                    "OTS_mBERT": 1450
                },
                "averageConfidence": 0.9856,
                "uniqueSourceCount": 42,
                "timeRange": {
                    "start": "2025-10-21T00:00:00Z",
                    "end": "2025-10-21T23:59:59Z"
                }
            }
        }
    }


class TMForumEventError(BaseModel):
    """TMForum standard error response."""
    code: str = Field(..., description="Error code")
    reason: str = Field(..., description="Error reason")
    message: str = Field(..., description="Error message")
    status: str = Field(..., description="HTTP status code")
    referenceError: Optional[str] = Field(None, description="Reference to error documentation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "code": "EVENT_NOT_FOUND",
                "reason": "Event not found",
                "message": "No event exists with the specified ID",
                "status": "404"
            }
        }
    }
