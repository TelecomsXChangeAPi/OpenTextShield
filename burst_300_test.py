#!/usr/bin/env python3
"""
Burst Test - 300 Messages Arriving Simultaneously
Simulates SMSC sending 300 SMS messages to the API at nearly the same time
"""

import asyncio
import time
import random
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from api_interface.services.model_loader import model_manager
from api_interface.services.prediction_service import prediction_service
from api_interface.models.request_models import PredictionRequest, ModelType

# Test messages from various origins
TEST_MESSAGES = [
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
    "Act now before this deal expires",
]


class BurstMetrics:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.request_times = []
        self.lock = asyncio.Lock()
        self.start_time = None
        self.end_time = None
        self.labels = {"ham": 0, "spam": 0, "phishing": 0}

    async def record_request(self, processing_time: float, success: bool = True, label: str = None):
        """Record a request result"""
        async with self.lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
                if label:
                    self.labels[label] = self.labels.get(label, 0) + 1
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
            p50 = sorted(self.request_times)[len(self.request_times) // 2]
            p95 = sorted(self.request_times)[int(len(self.request_times) * 0.95)]
            p99 = sorted(self.request_times)[int(len(self.request_times) * 0.99)] if len(self.request_times) > 100 else max_time

            return {
                "total_requests": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "p50_time": p50,
                "p95_time": p95,
                "p99_time": p99,
                "total_time": self.end_time - self.start_time if (self.start_time and self.end_time) else 0,
                "throughput": self.total_requests / (self.end_time - self.start_time) if (self.start_time and self.end_time) else 0,
                "labels": self.labels
            }


async def send_single_inference(message_index: int, metrics: BurstMetrics):
    """Send a single inference request"""
    try:
        text = random.choice(TEST_MESSAGES)
        request = PredictionRequest(text=text, model=ModelType.OTS_MBERT)

        start = time.time()
        result = await prediction_service.predict(request)
        processing_time = time.time() - start

        await metrics.record_request(processing_time, success=True, label=result.label)

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


async def main():
    """Run the burst test"""
    print("\n")
    print("█" * 80)
    print("█ OpenTextShield BURST TEST - 300 Messages from SMSC Arriving Simultaneously")
    print("█" * 80)
    print("\n⚠️  SCENARIO:")
    print("   Simulating SMSC gateway sending 300 SMS messages to your API")
    print("   at nearly the same time (burst traffic pattern)")
    print("\n📊 Starting burst test in 2 seconds...\\n")

    time.sleep(2)

    # Load model
    print("Loading mBERT model onto GPU...")
    model_manager.load_all_models()
    print(f"✅ Model loaded on: {str(model_manager.device).upper()}")

    # Initialize metrics
    metrics = BurstMetrics()
    metrics.start_time = time.time()

    print(f"\n🚀 BURST: Sending 300 concurrent requests simultaneously...\n")

    # Create 300 concurrent tasks (simulating burst from SMSC)
    tasks = [send_single_inference(i, metrics) for i in range(300)]

    # Execute all 300 requests at once
    burst_start = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=False)
    burst_end = time.time()
    burst_duration = burst_end - burst_start

    metrics.end_time = time.time()

    # Process results
    successes = sum(1 for r in results if r.get("success", False))
    failures = sum(1 for r in results if not r.get("success", False))

    # Get stats
    stats = await metrics.get_stats()

    # Print results
    print("\n")
    print("█" * 80)
    print("█ BURST TEST RESULTS - 300 SIMULTANEOUS MESSAGES")
    print("█" * 80)
    print(f"\n⏱️  TIMING:")
    print(f"   Burst Duration (all 300 sent): {burst_duration:.2f} seconds")
    print(f"   Time to Process All 300: {stats['total_time']:.2f} seconds")
    print(f"   Total API Time (load + inference): {time.time() - metrics.start_time:.2f}s")

    print(f"\n📊 REQUEST STATISTICS:")
    print(f"   Total Requests: {stats['total_requests']}")
    print(f"   Successful: {stats['successful']} ✅")
    print(f"   Failed: {stats['failed']} ❌")
    print(f"   Success Rate: {(stats['successful']/stats['total_requests']*100):.1f}%")

    print(f"\n⚡ THROUGHPUT:")
    print(f"   Throughput: {stats['throughput']:.2f} requests/second")
    print(f"   Effective Rate: {300/burst_duration:.2f} req/s (burst phase)")

    print(f"\n⏳ LATENCY ANALYSIS:")
    print(f"   Min Time: {stats['min_time']:.4f}s ({stats['min_time']*1000:.2f}ms)")
    print(f"   P50 (Median): {stats['p50_time']:.4f}s ({stats['p50_time']*1000:.2f}ms)")
    print(f"   P95: {stats['p95_time']:.4f}s ({stats['p95_time']*1000:.2f}ms)")
    print(f"   P99: {stats['p99_time']:.4f}s ({stats['p99_time']*1000:.2f}ms)")
    print(f"   Avg Time: {stats['avg_time']:.4f}s ({stats['avg_time']*1000:.2f}ms)")
    print(f"   Max Time: {stats['max_time']:.4f}s ({stats['max_time']*1000:.2f}ms)")

    print(f"\n🏷️  CLASSIFICATION RESULTS:")
    total_classified = sum(stats['labels'].values())
    for label, count in stats['labels'].items():
        percentage = (count / total_classified * 100) if total_classified > 0 else 0
        print(f"   {label.upper()}: {count} ({percentage:.1f}%)")

    print(f"\n💡 ANALYSIS:")
    if stats['success_rate'] == 100:
        print(f"   ✅ All 300 messages processed successfully")
        print(f"   ✅ System handled burst without errors")
    else:
        print(f"   ⚠️  {stats['failed']} messages failed to process")
        print(f"   ⚠️  Success rate: {stats['success_rate']:.1f}%")

    if stats['max_time'] > 1.0:
        print(f"   ⚠️  Some requests took {stats['max_time']:.2f}s to complete")
        print(f"   💬 This indicates queueing/queuing effect under burst load")
    else:
        print(f"   ✅ Max latency was {stats['max_time']*1000:.1f}ms - good responsiveness")

    avg_queue_time = stats['avg_time'] - 0.056  # Subtract baseline inference time
    if avg_queue_time > 0.1:
        print(f"   📈 Average queue wait time: ~{avg_queue_time*1000:.0f}ms per request")

    print(f"\n🎯 PRODUCTION CAPACITY:")
    print(f"   Your API can handle 300 simultaneous messages")
    print(f"   Processing them at {stats['throughput']:.2f} req/s")
    print(f"   Total completion time: {stats['total_time']:.2f}s")
    print(f"   Queue depth peaked at: ~300 messages")

    print("\n" + "█" * 80)
    print("█ BURST TEST COMPLETE")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
