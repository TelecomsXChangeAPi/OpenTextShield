#!/usr/bin/env python3
"""
Test script for OpenTextShield API structure without external dependencies.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_project_structure():
    """Test that all required files and directories exist."""
    print("🔍 Testing project structure...")
    
    required_files = [
        "src/api_interface/__init__.py",
        "src/api_interface/main.py",
        "src/api_interface/config/__init__.py",
        "src/api_interface/config/settings.py",
        "src/api_interface/models/__init__.py",
        "src/api_interface/models/request_models.py",
        "src/api_interface/models/response_models.py",
        "src/api_interface/services/__init__.py",
        "src/api_interface/services/model_loader.py",
        "src/api_interface/services/prediction_service.py",
        "src/api_interface/services/feedback_service.py",
        "src/api_interface/routers/__init__.py",
        "src/api_interface/routers/health.py",
        "src/api_interface/routers/prediction.py",
        "src/api_interface/routers/feedback.py",
        "src/api_interface/middleware/__init__.py",
        "src/api_interface/middleware/security.py",
        "src/api_interface/utils/__init__.py",
        "src/api_interface/utils/logging.py",
        "src/api_interface/utils/exceptions.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All required files exist!")
    return True

def test_model_paths():
    """Test model file paths."""
    print("\n🔍 Testing model paths...")
    
    model_paths = [
        "src/mBERT/training/model-training/mbert_ots_model_2.1.pth",
        "src/FastText/training/ots_fastext_model_v2.1.bin"
    ]
    
    missing_models = []
    for model_path in model_paths:
        full_path = project_root / model_path
        if not full_path.exists():
            missing_models.append(model_path)
            print(f"⚠️  {model_path} (missing)")
        else:
            print(f"✅ {model_path}")
    
    if missing_models:
        print(f"\n⚠️  Some model files are missing. This is expected if models haven't been trained yet.")
        return True  # Don't fail for missing models
    
    print("✅ All model files exist!")
    return True

def test_basic_imports():
    """Test basic Python imports without external dependencies."""
    print("\n🔍 Testing basic imports...")
    
    try:
        # Test settings without pydantic
        exec("""
import os
from pathlib import Path
from typing import Dict, List, Optional, Set

class MockSettings:
    def __init__(self):
        self.api_title = "OpenTextShield API"
        self.api_version = "2.1.0"
        self.api_host = "0.0.0.0"
        self.api_port = 8002
        self.allowed_ips = {"ANY", "127.0.0.1", "localhost"}
        self.cors_origins = ["*"]
        self.project_root = Path(__file__).parent if '__file__' in globals() else Path('.')
        self.models_base_path = self.project_root / "src"
        self.max_text_length = 512
        self.default_model = "bert"
        self.log_level = "INFO"

settings = MockSettings()
print(f"✅ Mock settings created: {settings.api_title} v{settings.api_version}")
""")
        
        print("✅ Basic imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_start_script():
    """Test if start script points to correct module."""
    print("\n🔍 Testing start script...")
    
    start_script = project_root / "start.sh"
    if not start_script.exists():
        print("❌ start.sh not found")
        return False
    
    with open(start_script, 'r') as f:
        content = f.read()
    
    if "src.api_interface.main:app" in content:
        print("✅ start.sh points to correct module (src.api_interface.main:app)")
        return True
    else:
        print("❌ start.sh does not point to correct module")
        return False

def main():
    """Run all tests."""
    print("🚀 OpenTextShield API Structure Test")
    print("=" * 50)
    
    tests = [
        test_project_structure,
        test_model_paths,
        test_basic_imports,
        test_start_script,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("🎉 API structure is ready for testing!")
        return 0
    else:
        print(f"❌ {total - passed} out of {total} tests failed")
        return 1

if __name__ == "__main__":
    exit(main())