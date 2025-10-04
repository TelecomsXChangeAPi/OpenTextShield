#!/usr/bin/env python3
"""
Test TMForum-compliant AI Inference Job Management API (TMF922).

Tests the new TMForum standard API alongside the existing legacy API.
"""

import sys
import time
import json
import requests
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

API_BASE_URL = "http://localhost:8002"


def test_tmforum_api_basic():
    """Test basic TMForum API functionality."""
    print("🧪 Testing TMForum API basic functionality...")

    # Test data
    test_text = "Congratulations! You've won $1000. Click here to claim your prize!"
    job_request = {
        "priority": "normal",
        "input": {
            "inputType": "text",
            "inputFormat": "plain",
            "inputData": {
                "text": test_text
            }
        },
        "model": {
            "id": "ots-mbert",
            "name": "OpenTextShield mBERT",
            "version": "2.1",
            "type": "bert",
            "capabilities": ["text-classification", "multilingual"]
        },
        "name": "Test SMS Classification",
        "description": "Testing TMForum API with spam message"
    }

    try:
        # Create inference job
        print("  📤 Creating TMForum inference job...")
        response = requests.post(
            f"{API_BASE_URL}/tmf-api/aiInferenceJob",
            json=job_request,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code != 201:
            print(f"  ❌ Failed to create job: {response.status_code} - {response.text}")
            return False

        job_data = response.json()
        job_id = job_data["id"]
        print(f"  ✅ Job created: {job_id}")

        # Wait for job completion (with timeout)
        print("  ⏳ Waiting for job completion...")
        max_attempts = 30  # 30 seconds timeout
        attempt = 0

        while attempt < max_attempts:
            response = requests.get(f"{API_BASE_URL}/tmf-api/aiInferenceJob/{job_id}")

            if response.status_code == 200:
                job_status = response.json()
                state = job_status["state"]

                if state == "completed":
                    print("  ✅ Job completed successfully")
                    break
                elif state == "failed":
                    print(f"  ❌ Job failed: {job_status.get('errorMessage', 'Unknown error')}")
                    return False
                elif state in ["acknowledged", "inProgress"]:
                    print(f"  ⏳ Job state: {state}")
                else:
                    print(f"  ⚠️  Unexpected job state: {state}")
            else:
                print(f"  ❌ Failed to get job status: {response.status_code}")
                return False

            time.sleep(1)
            attempt += 1

        if attempt >= max_attempts:
            print("  ❌ Job timeout - did not complete within 30 seconds")
            return False

        # Verify results
        final_job = response.json()
        output = final_job.get("output")

        if not output:
            print("  ❌ No output in completed job")
            return False

        label = output["outputData"].get("label")
        confidence = output.get("confidence")

        print(f"  📊 Classification result: {label} (confidence: {confidence:.3f})")

        # Verify expected result (should be spam)
        if label not in ["spam", "phishing"]:
            print(f"  ⚠️  Unexpected classification: {label}")
            return False

        print("  ✅ TMForum API test passed")
        return True

    except Exception as e:
        print(f"  ❌ TMForum API test failed: {str(e)}")
        return False


def test_legacy_api_compatibility():
    """Test that legacy API still works."""
    print("🧪 Testing legacy API compatibility...")

    test_text = "Free money! Click here now!"
    request_data = {
        "text": test_text,
        "model": "ots-mbert"
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code != 200:
            print(f"  ❌ Legacy API failed: {response.status_code} - {response.text}")
            return False

        result = response.json()
        label = result.get("label")
        probability = result.get("probability")

        print(f"  📊 Legacy API result: {label} (probability: {probability:.3f})")

        if label not in ["spam", "phishing"]:
            print(f"  ⚠️  Unexpected legacy classification: {label}")
            return False

        print("  ✅ Legacy API compatibility test passed")
        return True

    except Exception as e:
        print(f"  ❌ Legacy API test failed: {str(e)}")
        return False


def test_api_coexistence():
    """Test that both APIs can be used simultaneously."""
    print("🧪 Testing API coexistence...")

    # Test both APIs with the same input
    test_text = "Hello, this is a legitimate message from your bank."

    # Legacy API request
    legacy_request = {
        "text": test_text,
        "model": "ots-mbert"
    }

    # TMForum API request
    tmforum_request = {
        "priority": "high",
        "input": {
            "inputType": "text",
            "inputFormat": "plain",
            "inputData": {"text": test_text}
        },
        "model": {
            "id": "ots-mbert",
            "name": "OpenTextShield mBERT",
            "version": "2.1",
            "type": "bert",
            "capabilities": ["text-classification", "multilingual"]
        },
        "name": "Coexistence Test"
    }

    try:
        # Test legacy API
        legacy_response = requests.post(
            f"{API_BASE_URL}/predict/",
            json=legacy_request,
            headers={"Content-Type": "application/json"}
        )

        # Test TMForum API
        tmforum_response = requests.post(
            f"{API_BASE_URL}/tmf-api/aiInferenceJob",
            json=tmforum_request,
            headers={"Content-Type": "application/json"}
        )

        if legacy_response.status_code != 200:
            print(f"  ❌ Legacy API coexistence failed: {legacy_response.status_code}")
            return False

        if tmforum_response.status_code != 201:
            print(f"  ❌ TMForum API coexistence failed: {tmforum_response.status_code}")
            return False

        legacy_result = legacy_response.json()
        tmforum_job = tmforum_response.json()

        print(f"  📊 Legacy result: {legacy_result['label']}")
        print(f"  📊 TMForum job created: {tmforum_job['id']}")

        # Both should classify as ham (legitimate)
        if legacy_result['label'] != 'ham':
            print(f"  ⚠️  Legacy API unexpected result: {legacy_result['label']}")
            return False

        print("  ✅ API coexistence test passed")
        return True

    except Exception as e:
        print(f"  ❌ API coexistence test failed: {str(e)}")
        return False


def test_tmforum_list_jobs():
    """Test TMForum job listing functionality."""
    print("🧪 Testing TMForum job listing...")

    try:
        response = requests.get(f"{API_BASE_URL}/tmf-api/aiInferenceJob")

        if response.status_code != 200:
            print(f"  ❌ Job listing failed: {response.status_code}")
            return False

        result = response.json()
        jobs = result.get("jobs", [])

        print(f"  📊 Found {len(jobs)} jobs")

        # Should have at least the jobs we created in previous tests
        if len(jobs) < 2:  # At least 2 from previous tests
            print(f"  ⚠️  Expected at least 2 jobs, found {len(jobs)}")
            return False

        print("  ✅ Job listing test passed")
        return True

    except Exception as e:
        print(f"  ❌ Job listing test failed: {str(e)}")
        return False


def check_api_availability():
    """Check if the API is running and accessible."""
    print("🔍 Checking API availability...")

    try:
        # Check health endpoint
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)

        if response.status_code != 200:
            print(f"❌ API health check failed: {response.status_code}")
            print("💡 Make sure the API is running with: ./start.sh")
            return False

        health_data = response.json()
        print(f"✅ API is healthy: {health_data.get('status', 'unknown')}")

        # Check OpenAPI docs
        response = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API documentation is accessible")
        else:
            print("⚠️  API documentation may not be accessible")

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {str(e)}")
        print("💡 Make sure the API is running with: ./start.sh")
        return False


def main():
    """Run all TMForum API tests."""
    print("🚀 TMForum API Compatibility Test Suite")
    print("=" * 60)

    # Check prerequisites
    if not check_api_availability():
        print("\n❌ Prerequisites not met - exiting")
        return 1

    print("\n" + "=" * 60)
    print("🧪 RUNNING TESTS")
    print("=" * 60)

    tests = [
        test_legacy_api_compatibility,
        test_tmforum_api_basic,
        test_api_coexistence,
        test_tmforum_list_jobs,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {str(e)}")
            results.append(False)
            print()

    print("=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 ALL {total} TESTS PASSED!")
        print("\n✅ TMForum API successfully integrated")
        print("✅ Legacy API remains fully functional")
        print("✅ Both APIs work together seamlessly")
        print("\n🚀 Ready for production deployment!")
        return 0
    else:
        print(f"❌ {total - passed} out of {total} tests failed")
        print("\n🔧 Please check the API implementation and try again")
        return 1


if __name__ == "__main__":
    exit(main())