"""
Health check router for OpenTextShield API.
"""

from datetime import datetime, timezone
from fastapi import APIRouter

from ..models.response_models import HealthResponse
from ..services.model_loader import model_manager
from ..config.settings import settings

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check the health status of the API and loaded models"
)
async def health_check() -> HealthResponse:
    """
    Perform health check.
    
    Returns:
        Health status including model loading status
    """
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        models_loaded=model_manager.get_model_status(),
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@router.get(
    "/",
    include_in_schema=False
)
async def root():
    """Root endpoint redirect."""
    return {"message": "OpenTextShield API", "version": settings.api_version}