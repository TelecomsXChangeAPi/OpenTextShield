"""
Prediction router for OpenTextShield API.
"""

import asyncio
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime, timezone

from ..models.request_models import BatchPredictionRequest, PredictionRequest
from ..models.response_models import (
    BatchPredictionResponse,
    PredictionResponse,
    ErrorResponse,
)
from ..services.prediction_service import prediction_service
from ..services.audit_service import audit_service
from ..middleware.security import verify_ip_address
from ..utils.exceptions import OpenTextShieldException
from ..utils.logging import logger

router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict/",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Model Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    summary="Predict Text Classification",
    description="Classify text as ham, spam, or phishing using AI models",
    dependencies=[Depends(verify_ip_address)]
)
async def predict_text(request: PredictionRequest) -> PredictionResponse:
    """
    Classify text for spam/phishing detection.

    Args:
        request: Prediction request containing text and model preferences

    Returns:
        Classification result with confidence score and processing time

    Raises:
        HTTPException: For various error conditions
    """
    try:
        logger.info(f"Received prediction request: model={request.model}")
        result = await prediction_service.predict(request)

        # Log prediction to audit system (fire-and-forget, don't block response)
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(
                None,
                lambda: audit_service.log_prediction(
                    text=request.text,
                    label=result.label,
                    confidence=result.probability,
                    model=result.model_info.name,
                    model_version=result.model_info.version,
                    processing_time=result.processing_time,
                    client_ip="unknown",
                    text_length=len(request.text),
                ),
            )
        except Exception as audit_error:
            # Don't fail the request if audit logging fails
            logger.warning(f"Failed to log prediction to audit: {audit_error}")

        return result
        
    except OpenTextShieldException as e:
        logger.error(f"OpenTextShield error: {e.message}")
        
        # Map specific exceptions to HTTP status codes
        status_code = 400
        if e.error_code == "MODEL_NOT_FOUND":
            status_code = 404
        elif e.error_code == "VALIDATION_ERROR":
            status_code = 400
        elif e.error_code in ["MODEL_LOAD_ERROR", "PREDICTION_ERROR"]:
            status_code = 500
        
        raise HTTPException(
            status_code=status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "details": e.details,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in prediction endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": {"error": str(e)},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )


@router.post(
    "/predict-batch/",
    response_model=BatchPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Model Not Found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    summary="Classify a batch of texts in a single request",
    description=(
        "Classify up to 256 texts in one GPU forward pass. Intended for "
        "clients that already batch messages (SMPP bridges, queue workers) "
        "and want to trade a small amount of client-side latency for a "
        "large throughput uplift by eliminating the server-side batch-wait "
        "and the per-message HTTP round-trip. Index-aligned: items[i] is "
        "the classification for texts[i]."
    ),
    dependencies=[Depends(verify_ip_address)],
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Classify a batch of texts in a single forward pass.

    Args:
        request: Batch prediction request with a list of texts

    Returns:
        Batch prediction response, items index-aligned with request.texts

    Raises:
        HTTPException: For various error conditions
    """
    try:
        logger.info(
            f"Received batch prediction request: "
            f"model={request.model} batch_size={len(request.texts)}"
        )
        result = await prediction_service.predict_batch(request)

        # Fire-and-forget audit logging — one entry per item, dispatched
        # off the event loop so the response is not blocked.
        try:
            loop = asyncio.get_running_loop()
            for i, item in enumerate(result.items):
                loop.run_in_executor(
                    None,
                    lambda idx=i, it=item: audit_service.log_prediction(
                        text=request.texts[idx],
                        label=it.label,
                        confidence=it.probability,
                        model=result.model_info.name,
                        model_version=result.model_info.version,
                        processing_time=result.processing_time / result.batch_size,
                        client_ip="unknown",
                        text_length=len(request.texts[idx]),
                    ),
                )
        except Exception as audit_error:
            logger.warning(f"Failed to log batch predictions to audit: {audit_error}")

        return result

    except OpenTextShieldException as e:
        logger.error(f"OpenTextShield error (batch): {e.message}")
        status_code = 400
        if e.error_code == "MODEL_NOT_FOUND":
            status_code = 404
        elif e.error_code == "VALIDATION_ERROR":
            status_code = 400
        elif e.error_code in ["MODEL_LOAD_ERROR", "PREDICTION_ERROR"]:
            status_code = 500

        raise HTTPException(
            status_code=status_code,
            detail={
                "error": e.error_code,
                "message": e.message,
                "details": e.details,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as e:
        logger.error(f"Unexpected error in batch prediction endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "details": {"error": str(e)},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )