"""
Tests for the OTS_MBERT_MODEL_PATH model-path override in Settings.

The override mutates the (per-instance) mbert_model_configs dict, so these tests
also guard the KeyError edge case and per-instance isolation.

Run:
    pytest src/api_interface/tests/test_settings_override.py -v
"""

import sys
from pathlib import Path

# Make ``src`` importable when run from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api_interface.config.settings import Settings

DEFAULT_PATH = "mBERT/training/model-training/mbert_ots_model_2.7.pth"
ROLLBACK_PATH = "mBERT/training/model-training/mbert_ots_model_2.5.pth"


def test_override_applies_to_default_model():
    s = Settings(mbert_model_path=ROLLBACK_PATH)
    assert s.mbert_model_configs[s.default_mbert_version]["path"] == ROLLBACK_PATH


def test_no_override_keeps_configured_default():
    s = Settings()
    assert s.mbert_model_configs["multilingual"]["path"] == DEFAULT_PATH


def test_override_does_not_leak_across_instances():
    """Mutating one instance's config must not affect a fresh default instance."""
    Settings(mbert_model_path=ROLLBACK_PATH)  # mutate-and-discard
    fresh = Settings()
    assert fresh.mbert_model_configs["multilingual"]["path"] == DEFAULT_PATH


def test_override_with_unknown_default_version_does_not_raise():
    """KeyError guard: an unknown default_mbert_version must not crash startup."""
    s = Settings(mbert_model_path=ROLLBACK_PATH, default_mbert_version="does_not_exist")
    # No exception, and the real config is left untouched.
    assert "does_not_exist" not in s.mbert_model_configs
    assert s.mbert_model_configs["multilingual"]["path"] == DEFAULT_PATH
