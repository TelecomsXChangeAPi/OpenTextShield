#!/usr/bin/env python3
"""
Stress tests for OpenTextShield OTS-mBERT API.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from test_utils import TestRunner, TestDataGenerator


class StressTestRunner(TestRunner):
    """Specialized test runner for stress tests."""
    
    def run_sequential_test(self, message_count: int, base_text: str = "Sample SMS text for testing"):
        """Run sequential stress test."""
        if not self.check_prerequisites():
            return False
        
        self.start_test()
        
        # Generate test messages
        messages = TestDataGenerator.generate_sample_messages(base_text, message_count)
        
        # Sequential processing
        for i, text in enumerate(messages):
            result = self.client.predict(text)
            self.process_prediction(result, text)
            
            if (i + 1) % 50 == 0:
                success_rate = (self.results["successful_requests"] / (i + 1)) * 100
                print(f"Processed {i + 1}/{message_count} messages... (Success rate: {success_rate:.1f}%)")
        
        return self.finish_test(message_count)
    
    def run_concurrent_test(self, message_count: int, max_workers: int = 10, base_text: str = "Sample SMS text for concurrent testing"):
        """Run concurrent stress test."""
        if not self.check_prerequisites():
            return False
        
        self.start_test()
        print(f"Using ThreadPoolExecutor with {max_workers} workers")
        print()
        
        # Generate test messages
        messages = TestDataGenerator.generate_sample_messages(base_text, message_count)
        
        # Concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(self.client.predict, text) for text in messages]
            
            # Process completed futures
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                self.process_prediction(result)
                
                if i % 100 == 0:
                    success_rate = (self.results["successful_requests"] / i) * 100
                    print(f"Processed {i}/{message_count} messages... (Success rate: {success_rate:.1f}%)")
        
        return self.finish_test(message_count)
    
    def run_massive_test(self, message_count: int, base_text: str = "Sample SMS text for massive testing"):
        """Run massive sequential stress test with ETA."""
        if not self.check_prerequisites():
            return False
        
        self.start_test()
        print(f"This may take several minutes to complete.")
        print()
        
        # Generate test messages
        messages = TestDataGenerator.generate_sample_messages(base_text, message_count)
        
        # Sequential processing with ETA
        for i, text in enumerate(messages):
            result = self.client.predict(text)
            self.process_prediction(result, text)
            
            # Progress reporting every 500 messages
            if (i + 1) % 500 == 0:
                elapsed = time.time() - self.results["start_time"]
                current_rate = (i + 1) / elapsed
                eta = (message_count - (i + 1)) / current_rate if current_rate > 0 else 0
                success_rate = (self.results["successful_requests"] / (i + 1)) * 100
                print(f"Processed {i + 1}/{message_count:,} messages... (Success: {success_rate:.1f}%) - ETA: {eta:.0f}s")
        
        return self.finish_test(message_count)


def test_500_sequential():
    """500 message sequential stress test."""
    runner = StressTestRunner("Sequential Stress Test (500 messages)")
    return runner.run_sequential_test(500)


def test_1000_concurrent():
    """1000 message concurrent stress test."""
    runner = StressTestRunner("Concurrent Stress Test (1000 messages)")
    return runner.run_concurrent_test(1000, max_workers=10)


def test_20k_massive():
    """20,000 message massive stress test."""
    runner = StressTestRunner("Massive Stress Test (20,000 messages)")
    return runner.run_massive_test(20000)


def main():
    """Run stress test based on command line argument or prompt user."""
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == "500":
            return test_500_sequential()
        elif test_type == "1000":
            return test_1000_concurrent()
        elif test_type == "20k":
            return test_20k_massive()
        else:
            print("Invalid test type. Options: 500, 1000, 20k")
            return False
    else:
        print("Available stress tests:")
        print("1. 500 message sequential test")
        print("2. 1000 message concurrent test")
        print("3. 20,000 message massive test")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            return test_500_sequential()
        elif choice == "2":
            return test_1000_concurrent()
        elif choice == "3":
            return test_20k_massive()
        else:
            print("Invalid choice")
            return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)