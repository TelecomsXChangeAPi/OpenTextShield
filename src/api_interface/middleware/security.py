"""
Security middleware for OpenTextShield API.
"""

from fastapi import HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from ..config.settings import settings
from ..utils.logging import logger


def verify_ip_address(request: Request) -> str:
    """
    Verify client IP address against allowed IPs.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Client IP address
        
    Raises:
        HTTPException: If IP is not allowed
    """
    client_host = request.client.host
    
    # Check if any IP is allowed
    if "ANY" in settings.allowed_ips:
        logger.debug(f"IP verification bypassed, allowing: {client_host}")
        return client_host
    
    # Check against allowed IPs
    if client_host in settings.allowed_ips:
        logger.debug(f"IP verification passed: {client_host}")
        return client_host
    
    logger.warning(f"IP verification failed: {client_host}")
    raise HTTPException(
        status_code=403,
        detail={
            "error": "ACCESS_DENIED",
            "message": "Your IP address is not authorized to access this service",
            "client_ip": client_host
        }
    )


def setup_cors_middleware(app) -> None:
    """
    Set up CORS middleware.
    
    Args:
        app: FastAPI application instance
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    logger.info(f"CORS middleware configured with origins: {settings.cors_origins}")