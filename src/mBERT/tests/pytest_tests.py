#!/usr/bin/env python3
"""
Pytest-compatible tests for OpenTextShield OTS-mBERT API.
"""

import pytest
import time
from test_utils import OTSAPIClient, TestDataGenerator


@pytest.fixture
def api_client():
    """Provide API client for tests."""
    client = OTSAPIClient()
    if not client.is_healthy():
        pytest.skip("OpenTextShield API is not running")
    return client


class TestBasicFunctionality:
    """Basic API functionality tests."""
    
    def test_api_health(self, api_client):
        """Test API health endpoint."""
        assert api_client.is_healthy()
    
    def test_spam_detection(self, api_client):
        """Test spam message detection."""
        spam_text = "FREE! Click here to win $1000 now! Limited time offer!"
        result = api_client.predict(spam_text)
        
        assert "error" not in result
        assert "label" in result
        assert "probability" in result
        assert result["label"] in ["ham", "spam", "phishing"]
        assert 0 <= result["probability"] <= 1
    
    def test_ham_detection(self, api_client):
        """Test legitimate message detection."""
        ham_text = "Hi mom, I'll be home for dinner at 7pm. Love you!"
        result = api_client.predict(ham_text)
        
        assert "error" not in result
        assert "label" in result
        assert "probability" in result
        assert result["label"] in ["ham", "spam", "phishing"]
    
    def test_phishing_detection(self, api_client):
        """Test phishing message detection."""
        phishing_text = "URGENT: Your bank account has been compromised. Click this link immediately to secure your account: http://fake-bank.com"
        result = api_client.predict(phishing_text)
        
        assert "error" not in result
        assert "label" in result
        assert "probability" in result
        assert result["label"] in ["ham", "spam", "phishing"]
    
    def test_empty_text(self, api_client):
        """Test API behavior with empty text."""
        result = api_client.predict("")
        # Should either handle gracefully or return appropriate error
        assert "label" in result or "error" in result
    
    def test_very_long_text(self, api_client):
        """Test API behavior with very long text."""
        long_text = "This is a test message. " * 100  # 2400 characters
        result = api_client.predict(long_text)
        
        assert "error" not in result
        assert "label" in result
        assert result["label"] in ["ham", "spam", "phishing"]


class TestPerformance:
    """API performance tests."""
    
    def test_response_time(self, api_client):
        """Test API response time is reasonable."""
        text = "Test message for performance testing"
        
        start_time = time.time()
        result = api_client.predict(text)
        response_time = time.time() - start_time
        
        assert "error" not in result
        assert response_time < 5.0  # Should respond within 5 seconds
    
    def test_multiple_requests(self, api_client):
        """Test multiple sequential requests."""
        texts = TestDataGenerator.generate_sample_messages("Test message", 10)
        
        for text in texts:
            result = api_client.predict(text)
            assert "error" not in result
            assert "label" in result


class TestErrorHandling:
    """Error handling tests."""
    
    def test_invalid_model(self):
        """Test API behavior with invalid model name."""
        client = OTSAPIClient()
        if not client.is_healthy():
            pytest.skip("API not available")
        
        result = client.predict("Test message", model="invalid-model")
        # Should return error or handle gracefully
        assert "error" in result or "label" in result
    
    def test_malformed_request(self):
        """Test API behavior with malformed requests."""
        # This would require direct requests to test malformed JSON, etc.
        # For now, just test that our client handles basic cases
        client = OTSAPIClient()
        if client.is_healthy():
            result = client.predict("Test message")
            assert "label" in result or "error" in result


@pytest.mark.slow
class TestStressScenarios:
    """Stress test scenarios (marked as slow)."""
    
    def test_burst_requests(self, api_client):
        """Test handling of burst requests."""
        texts = TestDataGenerator.generate_sample_messages("Burst test message", 50)
        
        results = []
        for text in texts:
            result = api_client.predict(text)
            results.append(result)
        
        # Check that most requests succeeded
        successful = sum(1 for r in results if "error" not in r)
        assert successful >= 45  # At least 90% success rate
    
    @pytest.mark.timeout(300)  # 5 minute timeout
    def test_sustained_load(self, api_client):
        """Test sustained load over time."""
        texts = TestDataGenerator.generate_sample_messages("Sustained test", 100)
        
        successful = 0
        for text in texts:
            result = api_client.predict(text)
            if "error" not in result:
                successful += 1
        
        # Should maintain good success rate
        success_rate = successful / len(texts)
        assert success_rate >= 0.9  # 90% success rate


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])