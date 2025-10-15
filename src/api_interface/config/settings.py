"""
Configuration management for OpenTextShield API.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field
except ImportError:
    try:
        from pydantic import BaseSettings, Field
    except ImportError:
        # Fallback for very old versions
        class BaseSettings:
            pass
        def Field(*args, **kwargs):
            return None


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    api_title: str = "OpenTextShield API"
    api_description: str = "Professional SMS spam and phishing detection API"
    api_version: str = "2.5.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8002
    
    # Security
    allowed_ips: Set[str] = {"ANY", "127.0.0.1", "localhost"}
    cors_origins: List[str] = [
        "http://localhost:8080",
        "http://localhost:8081", 
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8081"
    ]
    
    # Model Paths
    project_root: Path = Path(__file__).parent.parent.parent.parent
    models_base_path: Path = project_root / "src"
    
    # mBERT Models
    mbert_model_configs: Dict[str, Dict[str, str]] = {
        "multilingual": {
            "path": "mBERT/training/model-training/mbert_ots_model_2.5.pth",
            "tokenizer": "bert-base-multilingual-cased",
            "num_labels": "3",
            "version": "2.5"
        }
    }
    
    
    # Processing
    max_text_length: int = 512
    default_model: str = "ots-mbert"
    default_mbert_version: str = "multilingual"
    
    # Device Configuration
    device: str = "cpu"  # Will be overridden by auto-detection
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Feedback
    feedback_dir: Path = Path("feedback")
    
    class Config:
        env_prefix = "OTS_"
        case_sensitive = False
        env_file = ".env"


# Global settings instance
settings = Settings()

# Ensure feedback directory exists
settings.feedback_dir.mkdir(exist_ok=True)