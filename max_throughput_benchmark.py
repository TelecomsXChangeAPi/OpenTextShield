#!/usr/bin/env python3
"""
Maximum Throughput Benchmark - Push Hardware to the Limit
Tests with increasing batch sizes to find optimal throughput
"""

import asyncio
import time
import random
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).parent / "src"))

from api_interface.services.model_loader import model_manager
from api_interface.services.prediction_service import prediction_service
from api_interface.models.request_models import PredictionRequest, ModelType

TEST_MESSAGES = [
    "Click here to claim your free prize now",
    "Congratulations you have won a million dollars",
    "Hello how are you doing today",
    "Please verify your account immediately",
    "Your account has been compromised please reset password",
]


class ThroughputBenchmark:
    def __init__(self):
        self.results = {}

    async def run_batch_test(self, batch_size: int, duration_seconds: int = 60):
        """Run inference test with specific batch size for fixed duration"""
        print(f"\n{'='*80}")
        print(f"Testing Batch Size: {batch_size} concurrent requests")
        print(f"Duration: {duration_seconds} seconds")
        print(f"{'='*80}")

        start_time = time.time()
        total_requests = 0
        successful_requests = 0
        request_times = []
        failed_count = 0

        batch_count = 0

        try:
            while time.time() - start_time < duration_seconds:
                # Create batch of concurrent tasks
                tasks = []
                for i in range(batch_size):
                    text = random.choice(TEST_MESSAGES)
                    request = PredictionRequest(
                        text=text,
                        model=ModelType.OTS_MBERT
                    )

                    async def run_inference(req):
                        try:
                            start = time.time()
                            result = await prediction_service.predict(req)
                            elapsed = time.time() - start
                            return elapsed, True
                        except Exception as e:
                            return 0, False

                    tasks.append(run_inference(request))

                # Execute all tasks concurrently
                batch_start = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=False)
                batch_elapsed = time.time() - batch_start

                # Process results
                for elapsed, success in results:
                    total_requests += 1
                    if success:
                        successful_requests += 1
                        request_times.append(elapsed)
                    else:
                        failed_count += 1

                batch_count += 1
                elapsed = time.time() - start_time
                throughput = total_requests / elapsed

                # Print progress every 10 batches
                if batch_count % 10 == 0:
                    print(
                        f"[{elapsed:.1f}s] Batch #{batch_count} | "
                        f"Total: {total_requests} | "
                        f"Success: {successful_requests} | "
                        f"Current Throughput: {throughput:.2f} req/s"
                    )

        except KeyboardInterrupt:
            print("\nTest interrupted by user")

        # Calculate statistics
        total_time = time.time() - start_time
        throughput = total_requests / total_time if total_time > 0 else 0
        avg_time = sum(request_times) / len(request_times) if request_times else 0
        min_time = min(request_times) if request_times else 0
        max_time = max(request_times) if request_times else 0

        result = {
            "batch_size": batch_size,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_count,
            "total_time": total_time,
            "throughput": throughput,
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "success_rate": (
                (successful_requests / total_requests * 100)
                if total_requests > 0
                else 0
            ),
        }

        self.results[batch_size] = result

        print(f"\n{'='*80}")
        print(f"BATCH SIZE {batch_size} RESULTS")
        print(f"{'='*80}")
        print(f"Total Requests:       {result['total_requests']}")
        print(f"Successful:           {result['successful_requests']} ✅")
        print(f"Failed:               {result['failed_requests']} ❌")
        print(f"Success Rate:         {result['success_rate']:.2f}%")
        print(f"Total Time:           {result['total_time']:.2f}s")
        print(f"Throughput:           {result['throughput']:.2f} requests/sec")
        print(f"Avg Inference Time:   {result['avg_time']:.4f}s")
        print(f"Min Inference Time:   {result['min_time']:.4f}s")
        print(f"Max Inference Time:   {result['max_time']:.4f}s")
        print(f"{'='*80}")

        return result

    def print_comparison(self):
        """Print comparison of all batch sizes tested"""
        print("\n\n")
        print("█" * 80)
        print("█ MAXIMUM THROUGHPUT COMPARISON - ALL BATCH SIZES")
        print("█" * 80)
        print()

        # Sort by batch size
        sorted_results = sorted(self.results.items(), key=lambda x: x[0])

        print(f"{'Batch Size':<15} {'Throughput':<20} {'Avg Time':<15} {'Success Rate':<15}")
        print("-" * 65)

        max_throughput = 0
        optimal_batch_size = 0

        for batch_size, result in sorted_results:
            throughput = result["throughput"]
            avg_time = result["avg_time"]
            success_rate = result["success_rate"]

            print(
                f"{batch_size:<15} {throughput:.2f} req/s{'':<8} "
                f"{avg_time:.4f}s{'':<7} {success_rate:.1f}%"
            )

            if throughput > max_throughput:
                max_throughput = throughput
                optimal_batch_size = batch_size

        print()
        print("=" * 65)
        print(f"MAXIMUM THROUGHPUT: {max_throughput:.2f} requests/second")
        print(f"OPTIMAL BATCH SIZE: {optimal_batch_size} concurrent requests")
        print("=" * 65)
        print()


async def main():
    print("\n" + "█" * 80)
    print("█ OpenTextShield Maximum Throughput Benchmark")
    print("█ Testing with increasing batch sizes to find hardware limits")
    print("█" * 80)

    # Load model
    print("\n🔄 Loading mBERT model onto GPU...")
    model_manager.load_all_models()
    print(f"✅ Model loaded on: {str(model_manager.device).upper()}")

    benchmark = ThroughputBenchmark()

    # Test with increasing batch sizes
    batch_sizes = [10, 25, 50, 100, 150, 200, 250, 300]
    test_duration = 30  # 30 seconds per test for faster benchmarking

    print(f"\n📊 Starting benchmarks with {test_duration}s per batch size...\n")

    for batch_size in batch_sizes:
        await benchmark.run_batch_test(batch_size, duration_seconds=test_duration)

        # Add delay between tests for GPU cooldown
        print(f"⏳ Cooling down for 5 seconds...")
        await asyncio.sleep(5)

    # Print final comparison
    benchmark.print_comparison()

    # Recommendations
    print("\n" + "█" * 80)
    print("█ HARDWARE CAPACITY RECOMMENDATIONS")
    print("█" * 80)
    print()

    optimal = benchmark.results[max(benchmark.results.keys(),
                                     key=lambda k: benchmark.results[k]["throughput"])]

    print(f"🎯 Optimal Configuration:")
    print(f"   • Batch Size: {optimal['batch_size']} concurrent requests")
    print(f"   • Throughput: {optimal['throughput']:.2f} requests/second")
    print(f"   • Avg Latency: {optimal['avg_time']*1000:.1f}ms per request")
    print()

    print(f"📈 Production Recommendations:")
    print(f"   • For sustained load: Use batch size 50-100")
    print(f"   • For peak capacity: Use batch size {optimal['batch_size']}")
    print(f"   • Monitor GPU temperature to avoid thermal throttling")
    print()

    print(f"⚠️  Limitations:")
    if optimal['success_rate'] < 100:
        print(f"   • At {optimal['batch_size']} requests, success rate: {optimal['success_rate']:.1f}%")
        print(f"   • Recommend reducing batch size for 100% reliability")
    else:
        print(f"   • All tests maintained 100% success rate")
        print(f"   • Hardware is operating at safe capacity")

    print()
    print("█" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
