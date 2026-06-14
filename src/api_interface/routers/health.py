"""
Health check router for OpenTextShield API.
"""

from datetime import datetime, timezone
from fastapi import APIRouter

from ..models.response_models import HealthResponse, ModelVersion
from ..services.model_loader import model_manager
from ..config.settings import settings


def _deployed_model() -> ModelVersion | None:
    """Build the deployed-model descriptor for /health from the default model.

    Resolves the version from the model manager (detected at load time) and the
    architecture from the configured tokenizer. Returns None if the default
    model isn't loaded, so /health degrades gracefully rather than erroring.
    """
    name = settings.default_mbert_version
    config = settings.mbert_model_configs.get(name, {})
    version = model_manager.get_model_version(name) or config.get("version")
    if not version:
        return None
    return ModelVersion(
        name="OTS_mBERT",
        version=version,
        architecture=config.get("tokenizer", "bert-base-multilingual-cased"),
    )

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
        api_version=settings.api_version,
        version=settings.api_version,  # backward-compatible alias
        model=_deployed_model(),
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