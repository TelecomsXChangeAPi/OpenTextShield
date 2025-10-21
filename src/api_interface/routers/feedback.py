"""
Feedback router for OpenTextShield API.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from datetime import datetime

from ..models.request_models import FeedbackRequest
from ..models.response_models import FeedbackResponse, ErrorResponse
from ..services.feedback_service import feedback_service
from ..services.audit_service import audit_service
from ..middleware.security import verify_ip_address
from ..utils.logging import logger

router = APIRouter(tags=["Feedback"])


@router.post(
    "/feedback/",
    response_model=FeedbackResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
    summary="Submit Feedback",
    description="Submit feedback about classification results",
    dependencies=[Depends(verify_ip_address)]
)
async def submit_feedback(request: FeedbackRequest, client_ip: str = Depends(verify_ip_address)) -> FeedbackResponse:
    """
    Submit user feedback about classification results.
    
    Args:
        request: Feedback request
        
    Returns:
        Confirmation of feedback submission
        
    Raises:
        HTTPException: For various error conditions
    """
    try:
        logger.info(f"Received feedback request for model: {request.model}")
        result = await feedback_service.submit_feedback(request)

        # Log feedback to audit system
        try:
            audit_service.log_feedback(
                feedback_id=result.feedback_id,
                text=request.content,
                original_label="unknown",  # Not provided in feedback request
                user_feedback=request.feedback,
                thumbs_up=request.thumbs_up,
                thumbs_down=request.thumbs_down,
                model=request.model.value,
                client_ip=client_ip,
                user_id=request.user_id
            )
        except Exception as audit_error:
            # Don't fail the request if audit logging fails
            logger.warning(f"Failed to log feedback to audit: {audit_error}")

        return result
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "FEEDBACK_SUBMISSION_ERROR",
                "message": "Failed to submit feedback",
                "details": {"error": str(e)},
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


@router.get(
    "/feedback/download/{model_name}",
    responses={
        200: {"description": "Feedback file download"},
        403: {"model": ErrorResponse, "description": "Forbidden"},
        404: {"model": ErrorResponse, "description": "Feedback file not found"},
    },
    summary="Download Feedback File",
    description="Download feedback CSV file for a specific model",
    dependencies=[Depends(verify_ip_address)]
)
async def download_feedback(model_name: str):
    """
    Download feedback file for a specific model.
    
    Args:
        model_name: Name of the model (e.g., 'ots-mbert')
        
    Returns:
        CSV file containing feedback data
        
    Raises:
        HTTPException: If file not found or other errors
    """
    try:
        # Validate model name
        valid_models = ["ots-mbert", "mbert_multilingual"]
        if model_name not in valid_models:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_MODEL_NAME",
                    "message": f"Invalid model name: {model_name}",
                    "details": {"valid_models": valid_models},
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            )
        
        # Get feedback file path
        file_path = feedback_service.get_feedback_file_path(model_name)
        
        if not file_path:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "FEEDBACK_FILE_NOT_FOUND",
                    "message": f"No feedback file found for model: {model_name}",
                    "details": {"model_name": model_name},
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            )
        
        logger.info(f"Serving feedback file for model: {model_name}")
        
        return FileResponse(
            path=str(file_path),
            media_type='text/csv',
            filename=f"feedback_{model_name}.csv",
            headers={"Content-Disposition": f"attachment; filename=feedback_{model_name}.csv"}
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error downloading feedback file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "FEEDBACK_DOWNLOAD_ERROR",
                "message": "Failed to download feedback file",
                "details": {"error": str(e)},
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )