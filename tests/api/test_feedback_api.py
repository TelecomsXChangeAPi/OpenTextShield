#!/usr/bin/env python3
"""
Test script for OpenTextShield feedback endpoints.
"""

import json
import requests
import time
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8002"
TIMEOUT = 30

def test_health_endpoint() -> bool:
    """Test if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=TIMEOUT)
        if response.status_code == 200:
            print("✅ API is running")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_prediction_endpoint() -> Dict[str, Any]:
    """Test prediction endpoint and return result for feedback testing."""
    print("\n🔍 Testing prediction endpoint...")
    
    test_data = {
        "text": "URGENT! Your bank account has been compromised. Click here immediately to verify your account.",
        "model": "ots-mbert"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/", 
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful: {result['label']} (confidence: {result['probability']:.2f})")
            return result
        else:
            print(f"❌ Prediction failed: {response.status_code} - {response.text}")
            return {}
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction request failed: {e}")
        return {}

def test_feedback_submission(prediction_result: Dict[str, Any]) -> str:
    """Test feedback submission endpoint."""
    print("\n📝 Testing feedback submission...")
    
    if not prediction_result:
        print("⚠️ Skipping feedback test - no prediction result available")
        return ""
    
    feedback_data = {
        "content": "URGENT! Your bank account has been compromised. Click here immediately to verify your account.",
        "feedback": "This was correctly identified as a phishing attempt. Good detection!",
        "thumbs_up": True,
        "thumbs_down": False,
        "user_id": "test_user_123",
        "model": "ots-mbert"
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/feedback/",
            json=feedback_data,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            feedback_id = result.get("feedback_id", "")
            print(f"✅ Feedback submitted successfully: {feedback_id}")
            print(f"   Message: {result.get('message', '')}")
            return feedback_id
        else:
            print(f"❌ Feedback submission failed: {response.status_code} - {response.text}")
            return ""
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Feedback submission request failed: {e}")
        return ""

def test_feedback_download() -> bool:
    """Test feedback file download."""
    print("\n📥 Testing feedback download...")
    
    model_name = "ots-mbert"
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/feedback/download/{model_name}",
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            print(f"✅ Feedback download successful for model: {model_name}")
            print(f"   Content-Type: {response.headers.get('content-type', 'unknown')}")
            print(f"   Content-Length: {len(response.content)} bytes")
            
            # Check if it's CSV content
            if 'text/csv' in response.headers.get('content-type', ''):
                content = response.text
                lines = content.split('\n')
                print(f"   CSV lines: {len(lines)}")
                if lines:
                    print(f"   Header: {lines[0]}")
                    if len(lines) > 1 and lines[1].strip():
                        print(f"   Sample data: {lines[1][:100]}...")
            
            return True
        elif response.status_code == 404:
            print(f"ℹ️ No feedback file found for model: {model_name} (this is normal if no feedback was submitted before)")
            return True
        else:
            print(f"❌ Feedback download failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Feedback download request failed: {e}")
        return False

def test_invalid_model_download() -> bool:
    """Test feedback download with invalid model name."""
    print("\n🚫 Testing invalid model download...")
    
    invalid_model = "invalid-model"
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/feedback/download/{invalid_model}",
            timeout=TIMEOUT
        )
        
        if response.status_code == 400:
            print(f"✅ Correctly rejected invalid model: {invalid_model}")
            return True
        else:
            print(f"❌ Should have rejected invalid model but got: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Invalid model test failed: {e}")
        return False

def main():
    """Run all feedback endpoint tests."""
    print("🧪 OpenTextShield Feedback API Tests")
    print("=" * 50)
    
    # Test API health
    if not test_health_endpoint():
        print("\n❌ API is not running. Please start the server first:")
        print("   ./start.sh")
        return
    
    time.sleep(1)
    
    # Test prediction (needed for feedback)
    prediction_result = test_prediction_endpoint()
    time.sleep(1)
    
    # Test feedback submission
    feedback_id = test_feedback_submission(prediction_result)
    time.sleep(1)
    
    # Test feedback download
    download_success = test_feedback_download()
    time.sleep(1)
    
    # Test invalid model download
    invalid_test_success = test_invalid_model_download()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"✅ API Health: PASS")
    print(f"✅ Prediction: {'PASS' if prediction_result else 'FAIL'}")
    print(f"✅ Feedback Submission: {'PASS' if feedback_id else 'FAIL'}")
    print(f"✅ Feedback Download: {'PASS' if download_success else 'FAIL'}")
    print(f"✅ Invalid Model Handling: {'PASS' if invalid_test_success else 'FAIL'}")
    
    if all([prediction_result, feedback_id, download_success, invalid_test_success]):
        print("\n🎉 All feedback endpoint tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the output above.")

if __name__ == "__main__":
    main()