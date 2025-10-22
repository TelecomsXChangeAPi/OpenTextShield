"""
Shared utilities for OpenTextShield mBERT API testing.
"""

import requests
import time
import random
import logging
from typing import Dict, Any, Optional


class OTSAPIClient:
    """OpenTextShield API client for testing."""
    
    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url
        self.predict_url = f"{base_url}/predict/"
        self.health_url = f"{base_url}/health"
    
    def is_healthy(self) -> bool:
        """Check if API is running and healthy."""
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def predict(self, text: str, model: str = "ots-mbert") -> Dict[str, Any]:
        """Make prediction via API."""
        data = {"text": text, "model": model}
        
        try:
            response = requests.post(self.predict_url, json=data, timeout=30)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API request failed with status {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}


class TestDataGenerator:
    """Generate test data for stress testing."""
    
    @staticmethod
    def generate_random_text(base_text: str, index: int) -> str:
        """Generate unique test message."""
        return f"{base_text} - Message {index} - Random {random.randint(1, 10000)}"
    
    @staticmethod
    def generate_sample_messages(base_text: str, count: int) -> list:
        """Generate list of sample messages."""
        return [TestDataGenerator.generate_random_text(base_text, i) for i in range(count)]


class TestLogger:
    """Centralized test logging."""
    
    def __init__(self, log_file: str = "ots_test_results.log"):
        self.logger = logging.getLogger("OTSTests")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def error(self, message: str):
        self.logger.error(message)


class TestRunner:
    """Base class for running tests with common functionality."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.client = OTSAPIClient()
        self.logger = TestLogger()
        self.results = {
            "successful_requests": 0,
            "failed_requests": 0,
            "start_time": None,
            "end_time": None
        }
    
    def check_prerequisites(self) -> bool:
        """Check if API is available before running tests."""
        if not self.client.is_healthy():
            print("❌ OpenTextShield API is not running!")
            print("Please start the API first with: ./start.sh")
            return False
        print("✅ OpenTextShield API is running")
        return True
    
    def start_test(self):
        """Initialize test run."""
        print(f"\n{'='*60}")
        print(f"Starting {self.test_name}")
        print(f"{'='*60}")
        self.results["start_time"] = time.time()
        self.logger.info(f"Started {self.test_name}")
    
    def process_prediction(self, result: Dict[str, Any], text: str = "") -> bool:
        """Process a single prediction result."""
        if "error" in result:
            self.results["failed_requests"] += 1
            self.logger.error(f"Failed prediction: {result['error']}")
            return False
        else:
            self.results["successful_requests"] += 1
            self.logger.info(f"Success - Label: {result.get('label')}, Confidence: {result.get('probability', 0):.4f}")
            return True
    
    def finish_test(self, total_messages: int):
        """Complete test and show results."""
        self.results["end_time"] = time.time()
        total_time = self.results["end_time"] - self.results["start_time"]
        throughput = total_messages / total_time if total_time > 0 else 0
        success_rate = (self.results["successful_requests"] / total_messages) * 100
        
        print(f"\n=== {self.test_name} Results ===")
        print(f"Total messages: {total_messages:,}")
        print(f"Successful requests: {self.results['successful_requests']:,}")
        print(f"Failed requests: {self.results['failed_requests']:,}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Throughput: {throughput:.2f} messages/second")
        print(f"Success rate: {success_rate:.1f}%")
        
        self.logger.info(f"Completed {self.test_name} - {self.results['successful_requests']}/{total_messages} successful")
        
        return self.results["failed_requests"] == 0