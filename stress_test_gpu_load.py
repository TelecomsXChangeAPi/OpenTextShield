#!/usr/bin/env python3
"""
GPU Load Stress Test - 100 Concurrent Inferences for 5 Minutes
This script will send continuous inference requests to stress test the GPU
and allow you to monitor GPU usage in Activity Monitor/GPU gauge
"""

import asyncio
import time
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from api_interface.services.model_loader import model_manager
from api_interface.services.prediction_service import prediction_service
from api_interface.models.request_models import PredictionRequest, ModelType


# Test messages to vary input
TEST_MESSAGES = [
    "This is a test message for GPU stress testing",
    "Click here to claim your free prize now",
    "Congratulations you have won a million dollars",
    "Hello how are you doing today",
    "Please verify your account immediately",
    "Your account has been compromised please reset password",
    "I love this product it is amazing",
    "This is spam message please ignore",
    "Urgent: Update your information now",
    "Free shipping on all orders today",
    "You have inherited money from relative",
    "Call this number for free consultation",
    "Verify your identity by clicking link",
    "Your package is ready for pickup",
    "Limited time offer expires today",
    "Win a new iPhone absolutely free",
    "Confirm your bank details now",
    "Your payment method has expired",
    "Click to see who viewed your profile",
    "You have a new message from friend",
]

# Performance metrics
class PerformanceMetrics:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_time = 0
        self.start_time = None
        self.request_times: List[float] = []
        self.lock = asyncio.Lock()

    async def record_request(self, processing_time: float, success: bool = True):
        """Record a request result"""
        async with self.lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            self.request_times.append(processing_time)

    async def get_stats(self):
        """Get current statistics"""
        async with self.lock:
            if not self.request_times:
                return {}

            avg_time = sum(self.request_times) / len(self.request_times)
            min_time = min(self.request_times)
            max_time = max(self.request_times)

            return {
                "total_requests": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "requests_per_sec": self.total_requests / max(1, time.time() - self.start_time) if self.start_time else 0
            }


async def run_single_inference(message_index: int, metrics: PerformanceMetrics):
    """Run a single inference request"""
    try:
        # Select random test message
        text = random.choice(TEST_MESSAGES)

        # Create request
        request = PredictionRequest(
            text=text,
            model=ModelType.OTS_MBERT
        )

        # Time the inference
        start = time.time()
        result = await prediction_service.predict(request)
        processing_time = time.time() - start

        # Record metrics
        await metrics.record_request(processing_time, success=True)

        return {
            "index": message_index,
            "label": result.label,
            "probability": result.probability,
            "time": processing_time,
            "success": True
        }

    except Exception as e:
        await metrics.record_request(0, success=False)
        return {
            "index": message_index,
            "error": str(e),
            "success": False
        }


async def run_concurrent_batch(batch_size: int, metrics: PerformanceMetrics, batch_number: int):
    """Run a batch of concurrent inferences"""
    print(f"\n[Batch {batch_number}] Starting {batch_size} concurrent requests...")

    tasks = [
        run_single_inference(i + (batch_number * batch_size), metrics)
        for i in range(batch_size)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Count successes
    successes = sum(1 for r in results if r.get("success", False))

    print(f"[Batch {batch_number}] ✅ {successes}/{batch_size} successful")

    return results


async def print_stats_periodically(metrics: PerformanceMetrics, interval: int = 10):
    """Print statistics every interval seconds"""
    while True:
        await asyncio.sleep(interval)

        stats = await metrics.get_stats()
        if stats:
            elapsed = time.time() - metrics.start_time
            print(f"\n{'='*80}")
            print(f"[{elapsed:.1f}s] PERFORMANCE STATS")
            print(f"{'='*80}")
            print(f"Total Requests:     {stats['total_requests']}")
            print(f"Successful:         {stats['successful']} ✅")
            print(f"Failed:             {stats['failed']} ❌")
            print(f"Avg Time:           {stats['avg_time']:.4f}s")
            print(f"Min Time:           {stats['min_time']:.4f}s")
            print(f"Max Time:           {stats['max_time']:.4f}s")
            print(f"Requests/sec:       {stats['requests_per_sec']:.2f}")
            print(f"{'='*80}\n")


async def main():
    """Run the stress test"""
    print("\n")
    print("█" * 80)
    print("█ OpenTextShield GPU STRESS TEST - 100 Concurrent Requests for 5 Minutes")
    print("█" * 80)
    print("\n⚠️  INSTRUCTIONS:")
    print("   1. Open Activity Monitor on your Mac")
    print("   2. Go to Window → GPU (or Window → Processes → click GPU column header)")
    print("   3. Watch the GPU metrics while this script runs")
    print("   4. You should see high GPU usage for the next 5 minutes")
    print("\n📊 Starting GPU stress test in 3 seconds...\n")

    time.sleep(3)

    # Load model
    print("Loading mBERT model onto GPU...")
    model_manager.load_all_models()
    print("✅ Model loaded on: " + str(model_manager.device).upper())
    print(f"✅ Device type: {model_manager.device.type}")

    # Initialize metrics
    metrics = PerformanceMetrics()
    metrics.start_time = time.time()

    # Start stats printer task
    stats_task = asyncio.create_task(print_stats_periodically(metrics, interval=15))

    # Run stress test for 5 minutes (300 seconds)
    test_duration = 300  # 5 minutes
    batch_size = 20  # 20 concurrent requests per batch
    batch_delay = 2   # 2 second delay between batches (to allow pipeline)

    start_time = time.time()
    batch_number = 0

    print(f"\n🚀 STARTING GPU LOAD TEST")
    print(f"   Duration: {test_duration} seconds (5 minutes)")
    print(f"   Batch size: {batch_size} concurrent requests")
    print(f"   Batch delay: {batch_delay} seconds")
    print(f"   Total expected: ~{(test_duration // batch_delay) * batch_size} requests\n")

    try:
        while time.time() - start_time < test_duration:
            remaining = test_duration - (time.time() - start_time)
            print(f"\n⏱️  Time remaining: {remaining:.0f}s | Batch #{batch_number}")

            # Run batch of concurrent requests
            await run_concurrent_batch(batch_size, metrics, batch_number)

            # Wait before next batch
            await asyncio.sleep(batch_delay)

            batch_number += 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")

    finally:
        # Cancel stats printer
        stats_task.cancel()

        # Print final summary
        await asyncio.sleep(1)

        final_stats = await metrics.get_stats()
        total_elapsed = time.time() - start_time

        print("\n\n")
        print("█" * 80)
        print("█ FINAL STRESS TEST SUMMARY")
        print("█" * 80)
        print(f"\nTotal Duration:     {total_elapsed:.1f} seconds")
        print(f"Total Requests:     {final_stats.get('total_requests', 0)}")
        print(f"Successful:         {final_stats.get('successful', 0)} ✅")
        print(f"Failed:             {final_stats.get('failed', 0)} ❌")
        print(f"\nAverage Time/Request: {final_stats.get('avg_time', 0):.4f}s")
        print(f"Min Time:             {final_stats.get('min_time', 0):.4f}s")
        print(f"Max Time:             {final_stats.get('max_time', 0):.4f}s")
        print(f"Throughput:           {final_stats.get('requests_per_sec', 0):.2f} requests/sec")

        if final_stats.get('total_requests', 0) > 0:
            success_rate = (final_stats.get('successful', 0) / final_stats.get('total_requests', 1)) * 100
            print(f"Success Rate:         {success_rate:.1f}%")

        print("\n✅ GPU STRESS TEST COMPLETE")
        print("█" * 80)
        print("\nYou can now check your Activity Monitor GPU graph to see the load that was applied.\n")


if __name__ == "__main__":
    asyncio.run(main())
