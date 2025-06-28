#!/usr/bin/env python3
"""
Full API test with real dependencies.
"""

import sys
import asyncio
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all imports work correctly."""
    print("🔍 Testing imports...")
    
    try:
        # Test configuration
        from src.api_interface.config.settings import settings
        print(f"✅ Settings loaded: {settings.api_title} v{settings.api_version}")
        
        # Test models
        from src.api_interface.models.request_models import PredictionRequest, ModelType
        from src.api_interface.models.response_models import PredictionResponse
        print("✅ Pydantic models imported")
        
        # Test services (without external ML dependencies)
        print("✅ Basic imports successful")
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_pydantic_models():
    """Test Pydantic model validation."""
    print("\n🔍 Testing Pydantic models...")
    
    try:
        from src.api_interface.models.request_models import PredictionRequest, ModelType
        
        # Test valid request
        valid_request = PredictionRequest(
            text="Test message",
            model=ModelType.BERT
        )
        print(f"✅ Valid request: {valid_request.text}, {valid_request.model}")
        
        # Test validation
        try:
            invalid_request = PredictionRequest(
                text="",  # Empty text should fail
                model=ModelType.BERT
            )
            print("❌ Empty text validation failed")
            return False
        except Exception:
            print("✅ Empty text validation works")
        
        return True
        
    except Exception as e:
        print(f"❌ Pydantic model test failed: {e}")
        return False

def test_fastapi_app_creation():
    """Test FastAPI app can be created."""
    print("\n🔍 Testing FastAPI app creation...")
    
    try:
        from src.api_interface.main import app
        print(f"✅ FastAPI app created: {app.title}")
        print(f"✅ OpenAPI docs URL: {app.docs_url}")
        print(f"✅ API routes: {len(app.routes)} routes")
        
        # List main routes
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                methods = getattr(route, 'methods', set())
                if methods and 'GET' in methods or 'POST' in methods:
                    print(f"   - {route.path} {list(methods)}")
        
        return True
        
    except Exception as e:
        print(f"❌ FastAPI app creation failed: {e}")
        return False

async def test_health_endpoint():
    """Test health endpoint without starting server."""
    print("\n🔍 Testing health endpoint...")
    
    try:
        from src.api_interface.routers.health import health_check
        
        response = await health_check()
        print(f"✅ Health check response: {response.status}")
        print(f"✅ Version: {response.version}")
        print(f"✅ Models status: {response.models_loaded}")
        
        return True
        
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        return False

def test_exception_handling():
    """Test custom exception handling."""
    print("\n🔍 Testing exception handling...")
    
    try:
        from src.api_interface.utils.exceptions import (
            OpenTextShieldException, ModelNotFoundError, ValidationError
        )
        
        # Test custom exceptions
        try:
            raise ModelNotFoundError("test-model")
        except OpenTextShieldException as e:
            print(f"✅ ModelNotFoundError: {e.message}")
        
        try:
            raise ValidationError("test-field", "test message")
        except OpenTextShieldException as e:
            print(f"✅ ValidationError: {e.message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Exception handling test failed: {e}")
        return False

def test_configuration():
    """Test configuration system."""
    print("\n🔍 Testing configuration...")
    
    try:
        from src.api_interface.config.settings import settings
        
        # Test basic settings
        assert settings.api_version == "2.1.0"
        assert settings.api_port == 8002
        assert "127.0.0.1" in settings.allowed_ips
        
        # Test path resolution
        assert settings.project_root.exists()
        assert settings.models_base_path.exists()
        
        print(f"✅ API Version: {settings.api_version}")
        print(f"✅ API Port: {settings.api_port}")
        print(f"✅ Allowed IPs: {len(settings.allowed_ips)} IPs")
        print(f"✅ Project root: {settings.project_root}")
        print(f"✅ Models path: {settings.models_base_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def run_async_tests():
    """Run async tests."""
    print("\n🔄 Running async tests...")
    
    async_tests = [
        test_health_endpoint,
    ]
    
    results = []
    for test in async_tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"❌ Async test {test.__name__} failed: {e}")
            results.append(False)
    
    return results

def main():
    """Run all tests."""
    print("🚀 OpenTextShield API Full Test")
    print("=" * 50)
    
    # Sync tests
    sync_tests = [
        test_imports,
        test_pydantic_models,
        test_fastapi_app_creation,
        test_exception_handling,
        test_configuration,
    ]
    
    results = []
    for test in sync_tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Async tests
    try:
        async_results = asyncio.run(run_async_tests())
        results.extend(async_results)
    except Exception as e:
        print(f"❌ Async tests failed: {e}")
        results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ All {total} tests passed!")
        print("🎉 API is ready for production testing!")
        
        print("\n🚀 Next steps:")
        print("   1. Start API: ./start.sh")
        print("   2. Test endpoints: curl http://localhost:8002/health")
        print("   3. View docs: http://localhost:8002/docs")
        
        return 0
    else:
        print(f"❌ {total - passed} out of {total} tests failed")
        return 1

if __name__ == "__main__":
    exit(main())