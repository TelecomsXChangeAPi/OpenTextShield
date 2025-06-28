#!/usr/bin/env python3
"""
Comprehensive test runner for OpenTextShield mBERT API.
Modern, maintainable test orchestration.
"""

import os
import sys
import subprocess
import time
import argparse
from test_utils import OTSAPIClient, TestLogger


class ModernTestRunner:
    """Modern test runner with better organization and reporting."""
    
    def __init__(self):
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.client = OTSAPIClient()
        self.logger = TestLogger("test_runner.log")
        
        # Available tests
        self.tests = {
            "basic": {
                "file": "test_basic.py",
                "description": "Basic SMS classification test",
                "estimated_time": "< 5 seconds"
            },
            "stress": {
                "file": "test_stress.py",
                "description": "Stress tests (500/1000/20k messages)",
                "estimated_time": "Varies (1 min - 30 min)"
            },
            "pytest": {
                "file": "pytest_tests.py",
                "description": "Pytest suite with unit tests",
                "estimated_time": "< 30 seconds"
            }
        }
    
    def check_prerequisites(self) -> bool:
        """Check if environment is ready for testing."""
        print("🔍 Checking prerequisites...")
        
        if not self.client.is_healthy():
            print("❌ OpenTextShield API is not running!")
            print("   Please start the API first: ./start.sh")
            return False
        
        print("✅ OpenTextShield API is running")
        
        # Check if pytest is available for pytest tests
        try:
            import pytest
            print("✅ Pytest is available")
        except ImportError:
            print("⚠️  Pytest not available (pytest tests will be skipped)")
        
        return True
    
    def run_single_test(self, test_name: str, *args) -> bool:
        """Run a single test with arguments."""
        if test_name not in self.tests:
            print(f"❌ Unknown test: {test_name}")
            return False
        
        test_info = self.tests[test_name]
        test_file = os.path.join(self.test_dir, test_info["file"])
        
        if not os.path.exists(test_file):
            print(f"❌ Test file not found: {test_info['file']}")
            return False
        
        print(f"\n{'='*60}")
        print(f"Running: {test_info['description']}")
        print(f"File: {test_info['file']}")
        print(f"Estimated time: {test_info['estimated_time']}")
        print(f"{'='*60}")
        
        try:
            start_time = time.time()
            
            # Build command
            cmd = [sys.executable, test_file] + list(args)
            
            # Special handling for pytest
            if test_name == "pytest":
                cmd = [sys.executable, "-m", "pytest", test_file, "-v"] + list(args)
            
            result = subprocess.run(cmd, timeout=1800)  # 30 minute timeout
            
            end_time = time.time()
            duration = end_time - start_time
            
            success = result.returncode == 0
            status = "✅ PASSED" if success else "❌ FAILED"
            
            print(f"\n{status} - Completed in {duration:.2f} seconds")
            
            self.logger.info(f"Test {test_name}: {status} in {duration:.2f}s")
            
            return success
            
        except subprocess.TimeoutExpired:
            print("❌ Test timed out (30 minutes)")
            self.logger.error(f"Test {test_name}: TIMEOUT")
            return False
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            self.logger.error(f"Test {test_name}: EXCEPTION - {e}")
            return False
    
    def run_all_tests(self) -> bool:
        """Run all available tests."""
        if not self.check_prerequisites():
            return False
        
        print(f"\n🚀 Running all OpenTextShield mBERT tests...")
        
        results = {}
        total_start = time.time()
        
        # Run each test
        for test_name in self.tests:
            # Skip pytest if not available
            if test_name == "pytest":
                try:
                    import pytest
                except ImportError:
                    print(f"\n⏭️  Skipping {test_name} (pytest not installed)")
                    continue
            
            success = self.run_single_test(test_name)
            results[test_name] = success
        
        total_time = time.time() - total_start
        
        # Summary
        self._print_summary(results, total_time)
        
        # Return True if all tests passed
        return all(results.values())
    
    def run_stress_tests(self) -> bool:
        """Run all stress test variants."""
        if not self.check_prerequisites():
            return False
        
        stress_tests = ["500", "1000", "20k"]
        results = {}
        
        print(f"\n🔥 Running all stress tests...")
        
        for test_type in stress_tests:
            print(f"\n--- Stress Test: {test_type} messages ---")
            success = self.run_single_test("stress", test_type)
            results[f"stress_{test_type}"] = success
            
            if not success:
                print(f"⚠️  Stress test {test_type} failed, stopping here.")
                break
            
            # Brief pause between stress tests
            if test_type != stress_tests[-1]:
                print("😴 Brief pause before next stress test...")
                time.sleep(5)
        
        return all(results.values())
    
    def _print_summary(self, results: dict, total_time: float):
        """Print test summary."""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        
        passed = sum(1 for success in results.values() if success)
        failed = len(results) - passed
        
        for test_name, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            description = self.tests.get(test_name, {}).get("description", test_name)
            print(f"{status} {description}")
        
        print(f"\nResults: {passed} passed, {failed} failed")
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        
        if failed == 0:
            print("🎉 All tests passed!")
        else:
            print("💥 Some tests failed - check logs for details")
    
    def list_tests(self):
        """List available tests."""
        print("Available tests:")
        for name, info in self.tests.items():
            print(f"  {name:10} - {info['description']} ({info['estimated_time']})")


def main():
    parser = argparse.ArgumentParser(description="OpenTextShield mBERT Test Runner")
    parser.add_argument("command", nargs="?", choices=["all", "basic", "stress", "pytest", "list"], 
                       default="list", help="Test command to run")
    parser.add_argument("args", nargs="*", help="Additional arguments for specific tests")
    
    args = parser.parse_args()
    runner = ModernTestRunner()
    
    if args.command == "list":
        runner.list_tests()
        return
    elif args.command == "all":
        success = runner.run_all_tests()
    elif args.command == "stress" and args.args and args.args[0] == "all":
        success = runner.run_stress_tests()
    elif args.command in runner.tests:
        if not runner.check_prerequisites():
            sys.exit(1)
        success = runner.run_single_test(args.command, *args.args)
    else:
        print(f"Unknown command: {args.command}")
        runner.list_tests()
        sys.exit(1)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()