"""
Main FastAPI application for OpenTextShield API.

Professional SMS spam and phishing detection API with multiple ML model support.
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime

from .config.settings import settings
from .utils.logging import setup_logging, logger
from .utils.exceptions import OpenTextShieldException
from .services.model_loader import model_manager
from .middleware.security import setup_cors_middleware
from .routers import health, prediction, feedback


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting OpenTextShield API...")
    logger.info(f"Version: {settings.api_version}")
    logger.info(f"Environment: {settings.api_host}:{settings.api_port}")
    
    try:
        # Load all models
        await asyncio.get_event_loop().run_in_executor(
            None, model_manager.load_all_models
        )
        logger.info("All models loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load models during startup: {str(e)}")
        # Continue startup even if some models fail to load
        # Individual endpoints will handle model availability
    
    logger.info("OpenTextShield API startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down OpenTextShield API...")
    logger.info("OpenTextShield API shutdown completed")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Set up middleware
setup_cors_middleware(app)

# Include routers
app.include_router(health.router)
app.include_router(prediction.router)
app.include_router(feedback.router)


@app.exception_handler(OpenTextShieldException)
async def opentextshield_exception_handler(request, exc: OpenTextShieldException):
    """Handle OpenTextShield custom exceptions."""
    logger.error(f"OpenTextShield exception: {exc.message}")
    
    status_code = 500
    if exc.error_code == "VALIDATION_ERROR":
        status_code = 400
    elif exc.error_code == "MODEL_NOT_FOUND":
        status_code = 404
    
    return JSONResponse(
        status_code=status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "details": {"error": str(exc)},
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Set up logging
    setup_logging(log_level=settings.log_level)
    
    logger.info(f"Starting OpenTextShield API server on {settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=False
    )