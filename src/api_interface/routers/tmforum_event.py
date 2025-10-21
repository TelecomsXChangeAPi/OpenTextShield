"""
TMForum-compliant Event Management API (TMF688) router.

This implements TMForum standard TMF688 Event Management API for audit logging,
providing enterprise-grade event/audit trail capabilities for telecom integration.
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse

from ..models.tmforum_event_models import (
    TMForumEvent,
    TMForumEventCreate,
    TMForumEventList,
    TMForumEventStatistics,
    TMForumEventError,
    TMForumEventType,
    TMForumEventPriority
)
from ..services.audit_service import audit_service
from ..middleware.security import verify_ip_address
from ..config.settings import settings
from ..utils.logging import logger

router = APIRouter(
    prefix="/tmf-api/event",
    tags=["TMForum Event Management (TMF688)"],
    dependencies=[Depends(verify_ip_address)]
)


@router.post(
    "",
    response_model=TMForumEvent,
    status_code=201,
    responses={
        201: {"model": TMForumEvent, "description": "Event created successfully"},
        400: {"model": TMForumEventError, "description": "Bad Request"},
        403: {"model": TMForumEventError, "description": "Forbidden"},
        422: {"model": TMForumEventError, "description": "Validation Error"},
        503: {"model": TMForumEventError, "description": "Audit system not available"},
    },
    summary="Create Event",
    description="""
    Create a new audit event manually.

    This endpoint follows TMForum TMF688 Event Management standard.
    Events are normally created automatically by the system, but this endpoint
    allows manual event creation for special cases.
    """
)
async def create_event(request_data: TMForumEventCreate) -> TMForumEvent:
    """
    Create a new audit event.

    Args:
        request_data: Event creation request

    Returns:
        Created event with ID and timestamps

    Raises:
        HTTPException: For validation or processing errors
    """
    if not settings.audit_enabled:
        raise HTTPException(
            status_code=503,
            detail=TMForumEventError(
                code="AUDIT_NOT_ENABLED",
                reason="Audit logging not enabled",
                message="Audit logging is not enabled on this instance",
                status="503"
            ).model_dump()
        )

    try:
        logger.info(f"Creating TMForum event: {request_data.eventType}")
        event = audit_service.create_event_tmf688(request_data)
        logger.info(f"Created TMForum event: {event.id}")
        return event

    except Exception as e:
        logger.error(f"Error creating TMForum event: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=TMForumEventError(
                code="EVENT_CREATION_ERROR",
                reason="Failed to create event",
                message=f"An error occurred while creating the event: {str(e)}",
                status="500"
            ).model_dump()
        )


@router.get(
    "",
    response_model=TMForumEventList,
    responses={
        200: {"model": TMForumEventList, "description": "Successful Response"},
        400: {"model": TMForumEventError, "description": "Bad Request"},
        403: {"model": TMForumEventError, "description": "Forbidden"},
        503: {"model": TMForumEventError, "description": "Audit system not available"},
    },
    summary="List Events",
    description="""
    Query and retrieve audit events with optional filtering.

    This endpoint follows TMForum TMF688 Event Management standard.
    Supports filtering by event type, time range, source, priority, and pagination.

    Returns events in reverse chronological order (most recent first).
    """
)
async def list_events(
    request: Request,
    eventType: Optional[str] = Query(None, description="Filter by event type (PredictionEvent, FeedbackEvent, SecurityEvent, SystemEvent)"),
    eventTime_gte: Optional[str] = Query(None, alias="eventTime.gte", description="Events after this time (ISO 8601)"),
    eventTime_lte: Optional[str] = Query(None, alias="eventTime.lte", description="Events before this time (ISO 8601)"),
    source_id: Optional[str] = Query(None, alias="source.id", description="Filter by source (client IP)"),
    reportingSystem_id: Optional[str] = Query(None, alias="reportingSystem.id", description="Filter by reporting system (model name)"),
    priority: Optional[str] = Query(None, description="Filter by priority (low, medium, high, critical)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of results"),
    offset: int = Query(0, ge=0, description="Number of results to skip"),
    fields: Optional[str] = Query(None, description="Comma-separated list of fields to include")
) -> TMForumEventList:
    """
    List and query audit events.

    Args:
        request: FastAPI request object
        eventType: Filter by event type
        eventTime_gte: Filter events after this time
        eventTime_lte: Filter events before this time
        source_id: Filter by source ID (client IP)
        reportingSystem_id: Filter by reporting system (model)
        priority: Filter by priority
        limit: Max results to return
        offset: Number of results to skip
        fields: Field selection (sparse fieldsets)

    Returns:
        Paginated list of events

    Raises:
        HTTPException: For various error conditions
    """
    if not settings.audit_enabled:
        raise HTTPException(
            status_code=503,
            detail=TMForumEventError(
                code="AUDIT_NOT_ENABLED",
                reason="Audit logging not enabled",
                message="Audit logging is not enabled on this instance",
                status="503"
            ).model_dump()
        )

    try:
        # Parse datetime parameters
        time_gte = None
        time_lte = None

        if eventTime_gte:
            try:
                time_gte = datetime.fromisoformat(eventTime_gte.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=TMForumEventError(
                        code="INVALID_TIME_FORMAT",
                        reason="Invalid time format",
                        message=f"Invalid eventTime.gte format: {eventTime_gte}. Use ISO 8601 format.",
                        status="400"
                    ).model_dump()
                )

        if eventTime_lte:
            try:
                time_lte = datetime.fromisoformat(eventTime_lte.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=TMForumEventError(
                        code="INVALID_TIME_FORMAT",
                        reason="Invalid time format",
                        message=f"Invalid eventTime.lte format: {eventTime_lte}. Use ISO 8601 format.",
                        status="400"
                    ).model_dump()
                )

        # Validate event type if provided
        if eventType and eventType not in [e.value for e in TMForumEventType]:
            raise HTTPException(
                status_code=400,
                detail=TMForumEventError(
                    code="INVALID_EVENT_TYPE",
                    reason="Invalid event type",
                    message=f"Invalid eventType: {eventType}. Must be one of: {', '.join([e.value for e in TMForumEventType])}",
                    status="400"
                ).model_dump()
            )

        # Validate priority if provided
        if priority and priority not in [p.value for p in TMForumEventPriority]:
            raise HTTPException(
                status_code=400,
                detail=TMForumEventError(
                    code="INVALID_PRIORITY",
                    reason="Invalid priority",
                    message=f"Invalid priority: {priority}. Must be one of: {', '.join([p.value for p in TMForumEventPriority])}",
                    status="400"
                ).model_dump()
            )

        logger.info(f"Querying TMForum events: type={eventType}, limit={limit}, offset={offset}")

        # Query events from audit service
        events, total_count = audit_service.query_events_tmf688(
            event_type=eventType,
            time_gte=time_gte,
            time_lte=time_lte,
            source_id=source_id,
            reporting_system=reportingSystem_id,
            priority=priority,
            limit=limit,
            offset=offset
        )

        # Calculate pagination
        has_more = (offset + len(events)) < total_count
        next_url = None
        if has_more:
            next_offset = offset + limit
            base_url = str(request.url).split('?')[0]
            query_params = []
            if eventType:
                query_params.append(f"eventType={eventType}")
            if eventTime_gte:
                query_params.append(f"eventTime.gte={eventTime_gte}")
            if eventTime_lte:
                query_params.append(f"eventTime.lte={eventTime_lte}")
            if source_id:
                query_params.append(f"source.id={source_id}")
            if reportingSystem_id:
                query_params.append(f"reportingSystem.id={reportingSystem_id}")
            if priority:
                query_params.append(f"priority={priority}")
            query_params.append(f"offset={next_offset}")
            query_params.append(f"limit={limit}")
            next_url = f"{base_url}?{'&'.join(query_params)}"

        return TMForumEventList(
            events=events,
            totalCount=total_count,
            hasMore=has_more,
            next=next_url
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying TMForum events: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=TMForumEventError(
                code="EVENT_QUERY_ERROR",
                reason="Failed to query events",
                message=f"An error occurred while querying events: {str(e)}",
                status="500"
            ).model_dump()
        )


@router.get(
    "/statistics",
    response_model=TMForumEventStatistics,
    responses={
        200: {"model": TMForumEventStatistics, "description": "Successful Response"},
        403: {"model": TMForumEventError, "description": "Forbidden"},
        503: {"model": TMForumEventError, "description": "Audit system not available"},
    },
    summary="Get Event Statistics",
    description="""
    Retrieve statistics and analytics from audit events.

    This is a custom endpoint that provides aggregated statistics
    following TMForum patterns.
    """
)
async def get_event_statistics() -> TMForumEventStatistics:
    """
    Get audit event statistics.

    Returns:
        Event statistics and analytics

    Raises:
        HTTPException: For various error conditions
    """
    if not settings.audit_enabled:
        raise HTTPException(
            status_code=503,
            detail=TMForumEventError(
                code="AUDIT_NOT_ENABLED",
                reason="Audit logging not enabled",
                message="Audit logging is not enabled on this instance",
                status="503"
            ).model_dump()
        )

    try:
        logger.info("Calculating TMForum event statistics")
        stats = audit_service.get_event_statistics_tmf688()
        return stats

    except Exception as e:
        logger.error(f"Error calculating TMForum event statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=TMForumEventError(
                code="STATISTICS_ERROR",
                reason="Failed to calculate statistics",
                message=f"An error occurred while calculating statistics: {str(e)}",
                status="500"
            ).model_dump()
        )


@router.get(
    "/{event_id}",
    response_model=TMForumEvent,
    responses={
        200: {"model": TMForumEvent, "description": "Successful Response"},
        403: {"model": TMForumEventError, "description": "Forbidden"},
        404: {"model": TMForumEventError, "description": "Event not found"},
        503: {"model": TMForumEventError, "description": "Audit system not available"},
    },
    summary="Get Event",
    description="""
    Retrieve a specific audit event by ID.

    This endpoint follows TMForum TMF688 Event Management standard.
    """
)
async def get_event(event_id: str) -> TMForumEvent:
    """
    Get a specific audit event by ID.

    Args:
        event_id: Event UUID

    Returns:
        Event details

    Raises:
        HTTPException: If event not found or other errors
    """
    if not settings.audit_enabled:
        raise HTTPException(
            status_code=503,
            detail=TMForumEventError(
                code="AUDIT_NOT_ENABLED",
                reason="Audit logging not enabled",
                message="Audit logging is not enabled on this instance",
                status="503"
            ).model_dump()
        )

    try:
        logger.info(f"Retrieving TMForum event: {event_id}")
        event = audit_service.get_event_by_id(event_id)

        if not event:
            raise HTTPException(
                status_code=404,
                detail=TMForumEventError(
                    code="EVENT_NOT_FOUND",
                    reason="Event not found",
                    message=f"No event exists with ID: {event_id}",
                    status="404"
                ).model_dump()
            )

        return event

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving TMForum event: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=TMForumEventError(
                code="EVENT_RETRIEVAL_ERROR",
                reason="Failed to retrieve event",
                message=f"An error occurred while retrieving the event: {str(e)}",
                status="500"
            ).model_dump()
        )
