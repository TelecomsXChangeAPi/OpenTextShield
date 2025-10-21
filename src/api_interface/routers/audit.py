"""
Audit log query router for OpenTextShield API.

Provides REST endpoints to retrieve and analyze audit logs.
"""

from fastapi import APIRouter, Depends, Query, HTTPException
from typing import Optional, List, Dict, Any

from ..models.audit_models import AuditEntryType
from ..services.audit_service import audit_service
from ..middleware.security import verify_ip_address
from ..config.settings import settings

router = APIRouter(tags=["Audit"], prefix="/audit")


@router.get(
    "/logs",
    summary="Query Audit Logs",
    description="Retrieve audit log entries with optional filtering",
    dependencies=[Depends(verify_ip_address)]
)
async def get_audit_logs(
    limit: int = Query(100, ge=1, le=10000, description="Maximum number of entries"),
    entry_type: Optional[str] = Query(None, description="Filter by entry type (prediction, feedback, access_denied, system_event)"),
    start_date: Optional[str] = Query(None, description="Start date (ISO 8601 format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO 8601 format)"),
    client_ip: Optional[str] = Query(None, description="Filter by client IP address"),
) -> List[Dict[str, Any]]:
    """
    Query audit logs with optional filtering.

    Returns audit entries in reverse chronological order (most recent first).

    Query Parameters:
    - `limit`: Maximum number of entries (1-10000, default 100)
    - `entry_type`: Filter by type (prediction, feedback, access_denied, system_event)
    - `start_date`: Include entries from this date onwards (ISO 8601)
    - `end_date`: Include entries up to this date (ISO 8601)
    - `client_ip`: Include only entries from this client IP

    Example:
    ```
    GET /audit/logs?limit=50&entry_type=prediction
    GET /audit/logs?limit=200&start_date=2025-10-20T00:00:00Z
    ```
    """
    if not settings.audit_enabled:
        raise HTTPException(
            status_code=503,
            detail="Audit logging is not enabled on this instance"
        )

    try:
        entries = audit_service.query_logs(
            limit=limit,
            entry_type=entry_type,
            start_date=start_date,
            end_date=end_date,
            client_ip=client_ip
        )
        return entries

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to query audit logs: {str(e)}"
        )


@router.get(
    "/stats",
    summary="Audit Statistics",
    description="Get summary statistics from audit logs",
    dependencies=[Depends(verify_ip_address)]
)
async def get_audit_stats() -> Dict[str, Any]:
    """
    Get audit log statistics and summary information.

    Returns:
    - `total_predictions`: Total number of classification predictions
    - `total_feedback`: Total number of feedback submissions
    - `total_access_denied`: Total number of access denied events
    - `predictions_by_label`: Count of predictions per classification label
    - `predictions_by_model`: Count of predictions per model
    - `avg_confidence`: Average confidence score across all predictions
    - `unique_client_count`: Number of unique client IPs

    Example:
    ```
    GET /audit/stats
    ```
    """
    if not settings.audit_enabled:
        raise HTTPException(
            status_code=503,
            detail="Audit logging is not enabled on this instance"
        )

    try:
        stats = audit_service.get_statistics()
        return stats

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate statistics: {str(e)}"
        )


@router.get(
    "/info",
    summary="Audit System Info",
    description="Get information about the audit logging system",
    dependencies=[Depends(verify_ip_address)]
)
async def get_audit_info() -> Dict[str, Any]:
    """
    Get information about the audit logging system configuration.

    Returns current audit settings and system status.

    Example:
    ```
    GET /audit/info
    ```
    """
    if not settings.audit_enabled:
        raise HTTPException(
            status_code=503,
            detail="Audit logging is not enabled on this instance"
        )

    archive_dir = str(settings.audit_archive_dir) if settings.audit_archive_dir else None

    return {
        # Enabled/disabled
        "audit_enabled": settings.audit_enabled,

        # Storage configuration
        "audit_directory": str(settings.audit_dir),
        "text_storage_mode": settings.audit_text_storage,
        "truncate_length": settings.audit_truncate_length,
        "redact_patterns": settings.audit_redact_patterns,

        # Rotation configuration
        "rotation": {
            "enabled": settings.audit_rotation_enabled,
            "strategy": settings.audit_rotation_strategy,
            "rotate_on_date_change": settings.audit_rotation_on_date_change,
            "max_file_size_mb": settings.audit_max_file_size_mb,
        },

        # Retention configuration
        "retention": {
            "enabled": settings.audit_retention_enabled,
            "retention_days": settings.audit_retention_days,
            "check_interval_hours": settings.audit_retention_check_interval_hours,
            "archive_enabled": settings.audit_archive_enabled,
            "archive_directory": archive_dir,
        },
    }
