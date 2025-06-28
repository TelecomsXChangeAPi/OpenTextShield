"""
Feedback service for OpenTextShield API.
"""

import csv
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config.settings import settings
from ..utils.logging import logger
from ..models.request_models import FeedbackRequest
from ..models.response_models import FeedbackResponse


class FeedbackService:
    """Service for handling user feedback."""
    
    def __init__(self):
        # Ensure feedback directory exists
        settings.feedback_dir.mkdir(exist_ok=True)
    
    def _get_feedback_file(self, model_name: str) -> Path:
        """Get feedback file path for a specific model."""
        return settings.feedback_dir / f"feedback_{model_name}.csv"
    
    def _write_feedback_to_csv(
        self,
        feedback_data: list,
        model_name: str
    ) -> None:
        """
        Write feedback data to CSV file.
        
        Args:
            feedback_data: List of feedback data
            model_name: Name of the model
        """
        file_path = self._get_feedback_file(model_name)
        
        # Check if file exists and is empty (needs header)
        write_header = not file_path.exists() or file_path.stat().st_size == 0
        
        with open(file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            
            if write_header:
                writer.writerow([
                    "FeedbackID", "Timestamp", "UserID", "Content", 
                    "Feedback", "ThumbsUp", "ThumbsDown", "Model"
                ])
            
            writer.writerow(feedback_data)
    
    async def submit_feedback(self, request: FeedbackRequest) -> FeedbackResponse:
        """
        Submit user feedback.
        
        Args:
            request: Feedback request
            
        Returns:
            Feedback response
        """
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare feedback data
        feedback_data = [
            feedback_id,
            timestamp,
            request.user_id or "anonymous",
            request.content,
            request.feedback,
            "Yes" if request.thumbs_up else "No",
            "Yes" if request.thumbs_down else "No",
            request.model.value
        ]
        
        try:
            # Write to appropriate CSV file based on model type
            model_name = request.model.value
            
            self._write_feedback_to_csv(feedback_data, model_name)
            
            logger.info(f"Feedback submitted successfully: {feedback_id}")
            
            return FeedbackResponse(
                message="Feedback received successfully",
                feedback_id=feedback_id
            )
            
        except Exception as e:
            logger.error(f"Failed to save feedback: {str(e)}")
            return FeedbackResponse(
                message="Feedback received but could not be saved",
                feedback_id=feedback_id
            )
    
    def get_feedback_file_path(self, model_name: str) -> Optional[Path]:
        """
        Get feedback file path if it exists.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path to feedback file or None if not found
        """
        file_path = self._get_feedback_file(model_name)
        
        if file_path.exists() and file_path.stat().st_size > 0:
            return file_path
        
        return None


# Global feedback service instance
feedback_service = FeedbackService()