"""
Configuration management for OpenTextShield API.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
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
    api_version: str = "2.6.0"
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
    feedback_dir: Path = project_root / "feedback"

    # Audit Logging Configuration
    audit_enabled: bool = True
    audit_dir: Path = project_root / "audit_logs"
    audit_text_storage: str = "full"  # Options: full, truncated, hash_only, redacted
    audit_truncate_length: int = 100
    audit_redact_patterns: List[str] = []  # PII redaction patterns

    # Log Rotation Configuration
    audit_rotation_enabled: bool = True  # Enable log rotation
    audit_rotation_strategy: str = "size_or_date"  # Options: size_only, date_only, size_or_date
    audit_max_file_size_mb: int = 100  # Max file size before rotation (size_only or size_or_date)
    audit_rotation_on_date_change: bool = True  # Rotate at midnight (date_only or size_or_date)

    # Retention Configuration
    audit_retention_enabled: bool = True  # Enable retention policy
    audit_retention_days: int = 90  # Delete logs older than this many days
    audit_retention_check_interval_hours: int = 24  # Check retention policy every N hours
    audit_archive_enabled: bool = False  # Archive instead of delete old logs
    audit_archive_dir: Optional[Path] = None  # Directory for archived logs
    
    class Config:
        env_prefix = "OTS_"
        case_sensitive = False
        env_file = ".env"


# Global settings instance
settings = Settings()

# Note: Directory creation moved to main.py lifespan handler
# to avoid crashes on import if filesystem permissions are insufficient