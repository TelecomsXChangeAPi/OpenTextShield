"""
Custom exceptions for OpenTextShield API.
"""

from typing import Any, Dict, Optional


class OpenTextShieldException(Exception):
    """Base exception for OpenTextShield API."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "GENERIC_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ModelLoadError(OpenTextShieldException):
    """Raised when model loading fails."""
    
    def __init__(self, model_name: str, details: Optional[Dict[str, Any]] = None):
        message = f"Failed to load model: {model_name}"
        super().__init__(
            message=message,
            error_code="MODEL_LOAD_ERROR",
            details=details
        )


class PredictionError(OpenTextShieldException):
    """Raised when prediction fails."""
    
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        message = "Prediction failed"
        super().__init__(
            message=message,
            error_code="PREDICTION_ERROR",
            details=details
        )


class ValidationError(OpenTextShieldException):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, message: str):
        super().__init__(
            message=f"Validation error for field '{field}': {message}",
            error_code="VALIDATION_ERROR",
            details={"field": field}
        )


class ModelNotFoundError(OpenTextShieldException):
    """Raised when requested model is not available."""
    
    def __init__(self, model_name: str):
        message = f"Model not found: {model_name}"
        super().__init__(
            message=message,
            error_code="MODEL_NOT_FOUND",
            details={"model_name": model_name}
        )