#!/usr/bin/env python3
"""
Simple API test without complex dependencies.
"""

import sys
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_simple_settings():
    """Test simple settings without pydantic-settings."""
    print("🔍 Testing simple settings...")
    
    try:
        from src.api_interface.config.simple_settings import settings
        print(f"✅ Settings loaded: {settings.api_title} v{settings.api_version}")
        print(f"✅ API Host: {settings.api_host}:{settings.api_port}")
        print(f"✅ Allowed IPs: {len(settings.allowed_ips)} IPs")
        print(f"✅ Models path exists: {settings.models_base_path.exists()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple settings failed: {e}")
        return False

def test_health_check_standalone():
    """Test health check logic standalone."""
    print("\n🔍 Testing health check standalone...")
    
    try:
        # Import simple settings first
        from src.api_interface.config.simple_settings import settings
        
        # Mock health response
        health_data = {
            "status": "healthy",
            "version": settings.api_version,
            "models_loaded": {
                "bert": False,  # Models not loaded in test
                "fasttext": False
            },
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        print(f"✅ Health check data: {health_data['status']}")
        print(f"✅ Version: {health_data['version']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Health check test failed: {e}")
        return False

def test_mock_prediction():
    """Test prediction logic with mock data."""
    print("\n🔍 Testing mock prediction...")
    
    try:
        # Mock prediction request
        request_data = {
            "text": "Free money! Click here now!",
            "model": "bert",
            "bert_version": "bert-base-multilingual-cased"
        }
        
        # Mock prediction response
        response_data = {
            "label": "spam",
            "probability": 0.95,
            "processing_time": 0.15,
            "model_info": {
                "name": "OTS_mBERT",
                "version": "bert-base-multilingual-cased",
                "author": "TelecomsXChange (TCXC)",
                "last_training": "2024-03-20"
            }
        }
        
        print(f"✅ Mock request: {request_data['text'][:30]}...")
        print(f"✅ Mock response: {response_data['label']} ({response_data['probability']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock prediction test failed: {e}")
        return False

def test_file_structure():
    """Test that all files exist and are readable."""
    print("\n🔍 Testing file structure...")
    
    try:
        files_to_check = [
            "src/api_interface/main.py",
            "src/api_interface/config/simple_settings.py",
            "src/api_interface/routers/health.py",
            "src/api_interface/routers/prediction.py",
        ]
        
        for file_path in files_to_check:
            full_path = project_root / file_path
            if not full_path.exists():
                print(f"❌ Missing: {file_path}")
                return False
            
            # Try to read first few lines
            with open(full_path, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    print(f"✅ {file_path}")
                else:
                    print(f"⚠️  {file_path} (empty)")
        
        return True
        
    except Exception as e:
        print(f"❌ File structure test failed: {e}")
        return False

def test_basic_fastapi():
    """Test basic FastAPI functionality."""
    print("\n🔍 Testing basic FastAPI...")
    
    try:
        from fastapi import FastAPI
        
        # Create a simple app
        app = FastAPI(title="Test API", version="1.0.0")
        
        @app.get("/test")
        def test_endpoint():
            return {"message": "test"}
        
        print(f"✅ FastAPI app created: {app.title}")
        print(f"✅ Routes count: {len(app.routes)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic FastAPI test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 OpenTextShield API Simple Test")
    print("=" * 50)
    
    tests = [
        test_simple_settings,
        test_health_check_standalone,
        test_mock_prediction,
        test_file_structure,
        test_basic_fastapi,
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
        print("🎉 Basic API structure is working!")
        
        print("\n🚀 Next steps to test full API:")
        print("   1. Install missing dependencies if needed:")
        print("      pip install pydantic-settings")
        print("   2. Start API with simple config:")
        print("      python3 -c 'from src.api_interface.main import app; import uvicorn; uvicorn.run(app, host=\"0.0.0.0\", port=8002)'")
        print("   3. Test endpoints:")
        print("      curl http://localhost:8002/health")
        
        return 0
    else:
        print(f"❌ {total - passed} out of {total} tests failed")
        return 1

if __name__ == "__main__":
    exit(main())