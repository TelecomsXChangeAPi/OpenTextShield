"""
Prediction router for OpenTextShield API.
"""

from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime

from ..models.request_models import PredictionRequest
from ..models.response_models import PredictionResponse, ErrorResponse
from ..services.prediction_service import prediction_service
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
                "timestamp": datetime.utcnow().isoformat() + "Z"
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
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )