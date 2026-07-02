"""
Configuration management for OpenTextShield API.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    api_title: str = "OpenTextShield API"
    api_description: str = "Professional SMS spam and phishing detection API"
    api_version: str = "2.10.0"
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
            "path": "mBERT/training/model-training/mbert_ots_model_2.7.pth",
            "tokenizer": "bert-base-multilingual-cased",
            "num_labels": "3",
            "version": "2.7"
        }
    }

    # Operational override for the deployed mBERT weights, relative to
    # models_base_path (or absolute). Set OTS_MBERT_MODEL_PATH to swap the model
    # without a code change — e.g. an instant rollback to the previous production
    # checkpoint: OTS_MBERT_MODEL_PATH=mBERT/training/model-training/mbert_ots_model_2.5.pth
    # The version is auto-detected from the filename, so /predict and /health
    # report the rolled-back version correctly.
    mbert_model_path: Optional[str] = None

    
    # Processing
    # SMS payloads are typically 20-60 tokens. The previous default of 512
    # padded every request to the full sequence length which wasted ~10x the
    # GPU FLOPs per forward pass. 96 tokens covers even long-form phishing
    # content with headroom; outliers are truncated safely.
    max_text_length: int = 96
    default_model: str = "ots-mbert"
    default_mbert_version: str = "multilingual"

    # Device Configuration
    device: str = "cpu"  # Will be overridden by auto-detection

    # FP16 inference on CUDA. Ignored on CPU/MPS. Tensor-core GPUs (T4, A10,
    # L4, A100) gain ~2x throughput with no measurable accuracy loss for
    # classification heads.
    use_fp16: bool = True

    # Dynamic batching configuration. Coalesces concurrent single-message
    # requests into padded batches to raise GPU utilization. Disable for
    # single-request debugging.
    batching_enabled: bool = True
    # Raised from 32 after GPU load testing on T4 (g4dn.4xlarge) showed batches
    # were filling to the cap under burst. 64 batch at 96 tokens fp16 stays
    # well under 500MB of activations — comfortable on a 16GB T4.
    max_batch_size: int = 64
    # Raised from 15ms after load-test observation that the 15ms window was
    # flushing mostly 5–8 sized micro-batches. 50ms lets the batcher pack
    # closer to max_batch_size under sustained load; adds a <=50ms latency
    # floor under low traffic.
    batch_wait_ms: int = 50
    
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
    
    model_config = SettingsConfigDict(
        env_prefix="OTS_",
        case_sensitive=False,
        env_file=".env",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _apply_model_path_override(self):
        """Apply OTS_MBERT_MODEL_PATH to the default model config, if set.

        Lets ops point the deployed model at a different checkpoint (e.g. roll
        back to v2.5) purely via environment, without editing mbert_model_configs.
        """
        # Guard on the key existing: OTS_DEFAULT_MBERT_VERSION could be set to a
        # value not present in mbert_model_configs, which would otherwise KeyError.
        if self.mbert_model_path and self.default_mbert_version in self.mbert_model_configs:
            self.mbert_model_configs[self.default_mbert_version]["path"] = self.mbert_model_path
        return self


# Global settings instance
settings = Settings()

# Note: Directory creation moved to main.py lifespan handler
# to avoid crashes on import if filesystem permissions are insufficient