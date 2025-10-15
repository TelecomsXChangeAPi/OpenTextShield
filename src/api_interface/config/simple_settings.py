"""
Simple configuration for OpenTextShield API (no pydantic-settings dependency).
"""

import os
from pathlib import Path
from typing import Dict, List, Set


class Settings:
    """Application settings with environment variable support."""
    
    def __init__(self):
        # API Configuration
        self.api_title = "OpenTextShield API"
        self.api_description = "Professional SMS spam and phishing detection API"
        self.api_version = "2.5.0"
        self.api_host = os.getenv("OTS_API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("OTS_API_PORT", "8002"))
        
        # Security
        allowed_ips_str = os.getenv("OTS_ALLOWED_IPS", "ANY,127.0.0.1,localhost")
        self.allowed_ips: Set[str] = set(allowed_ips_str.split(","))
        
        cors_origins_str = os.getenv("OTS_CORS_ORIGINS", "*")
        self.cors_origins: List[str] = cors_origins_str.split(",")
        
        # Model Paths
        self.project_root = Path(__file__).parent.parent.parent.parent
        self.models_base_path = self.project_root / "src"
        
        # BERT Models
        self.bert_model_configs: Dict[str, Dict[str, str]] = {
            "bert-base-multilingual-cased": {
                "path": "mBERT/training/model-training/mbert_ots_model_2.5.pth",
                "tokenizer": "bert-base-multilingual-cased",
                "num_labels": "3"
            }
        }
        
        # Note: FastText support removed in v2.1 - now using mBERT only
        
        # Processing
        self.max_text_length = int(os.getenv("OTS_MAX_TEXT_LENGTH", "512"))
        self.default_model = os.getenv("OTS_DEFAULT_MODEL", "bert")
        self.default_bert_version = os.getenv("OTS_DEFAULT_BERT_VERSION", "bert-base-multilingual-cased")
        
        # Device Configuration
        self.device = "cpu"  # Will be overridden by auto-detection
        
        # Logging
        self.log_level = os.getenv("OTS_LOG_LEVEL", "INFO")
        self.log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
        # Feedback
        self.feedback_dir = Path("feedback")
        self.feedback_dir.mkdir(exist_ok=True)


# Global settings instance
settings = Settings()