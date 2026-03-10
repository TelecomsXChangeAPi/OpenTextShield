"""
Burst Test: 300 Simultaneous Requests Against Real 10-Server Docker Deployment

This test runs the SAME 300-message burst against the nginx load balancer
that distributes requests across 10 real API servers in Docker containers.

Compares:
- Simulated test (async): 1.74s, 172.41 req/s, 68.52ms max latency
- Real Docker deployment: expected ~1.76-1.80s (+ 15-20ms network overhead)
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
import statistics

# Configuration
LOAD_BALANCER_URL = "http://localhost:8002"
HEALTH_CHECK_URL = f"{LOAD_BALANCER_URL}/health"
PREDICTION_URL = f"{LOAD_BALANCER_URL}/predict/"

# Test messages (same as used in async simulation)
TEST_MESSAGES = [
    "Click here to claim your free prize! Limited time offer.",
    "Your account has been compromised. Verify immediately.",
    "Hi, how are you doing today?",
    "Congratulations! You've won $1,000,000!",
    "Mom, are you free this weekend?",
    "URGENT: Confirm your banking details now!",
    "Let's catch up soon!",
    "You've been selected for a special offer.",
    "See you at the meeting tomorrow.",
    "Act now before this offer expires!",
    "Thanks for your help yesterday.",
    "Update your payment method immediately.",
    "I'll call you later.",
    "You are a lucky winner!",
    "Let's meet for lunch?",
]

async def health_check():
    """Check if load balancer and API servers are ready."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(HEALTH_CHECK_URL, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    return True
    except Exception as e:
        print(f"Health check failed: {e}")
    return False

async def make_prediction(session, message_id, text):
    """Make a single prediction request."""
    payload = {
        "text": text,
        "model": "ots-mbert"
    }

    request_start = time.time()
    try:
        async with session.post(
            PREDICTION_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            request_time = time.time() - request_start

            if resp.status == 200:
                data = await resp.json()
                return {
                    "message_id": message_id,
                    "status": "success",
                    "latency_ms": request_time * 1000,
                    "label": data.get("label"),
                    "probability": data.get("probability"),
                    "processing_time": data.get("processing_time"),
                }
            else:
                return {
                    "message_id": message_id,
                    "status": "error",
                    "latency_ms": request_time * 1000,
                    "http_status": resp.status,
                }
    except asyncio.TimeoutError:
        return {
            "message_id": message_id,
            "status": "timeout",
            "latency_ms": (time.time() - request_start) * 1000,
        }
    except Exception as e:
        return {
            "message_id": message_id,
            "status": "error",
            "latency_ms": (time.time() - request_start) * 1000,
            "error": str(e),
        }

async def run_burst_test(num_messages=300):
    """Run the burst test with 300 simultaneous requests."""
    print("\n" + "="*80)
    print("BURST TEST: 300 Simultaneous Requests Against Real 10-Server Docker Deployment")
    print("="*80)

    # Step 1: Health check
    print(f"\n[1/3] Checking if load balancer and API servers are ready...")
    max_retries = 30  # Try for up to 30 seconds
    for attempt in range(max_retries):
        ready = await health_check()
        if ready:
            print(f"✅ Servers are ready! (took {attempt+1} attempts)")
            break
        print(f"⏳ Waiting for servers to stabilize... (attempt {attempt+1}/{max_retries})")
        await asyncio.sleep(1)
    else:
        print("❌ Servers failed to stabilize after 30 seconds. Check Docker logs.")
        print("\nTroubleshooting:")
        print("  docker compose -f docker-compose.10x.yml logs api-1")
        return None

    # Step 2: Prepare test messages
    print(f"\n[2/3] Preparing {num_messages} test messages...")
    test_requests = []
    for i in range(num_messages):
        message = TEST_MESSAGES[i % len(TEST_MESSAGES)]
        test_requests.append((i, message))

    # Step 3: Send all requests simultaneously
    print(f"\n[3/3] Sending {num_messages} simultaneous requests...")
    print(f"Target: {PREDICTION_URL}")

    burst_start = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = [
            make_prediction(session, msg_id, text)
            for msg_id, text in test_requests
        ]
        results = await asyncio.gather(*tasks)

    burst_duration = time.time() - burst_start

    # Analyze results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    print(f"\n📊 Summary:")
    print(f"  Total requests: {num_messages}")
    print(f"  Successful: {len(successful)} ({100*len(successful)/num_messages:.1f}%)")
    print(f"  Failed: {len(failed)} ({100*len(failed)/num_messages:.1f}%)")
    print(f"  Total duration: {burst_duration:.2f}s")
    print(f"  Throughput: {num_messages/burst_duration:.2f} req/s")

    if successful:
        latencies = [r["latency_ms"] for r in successful]
        print(f"\n⏱️  Latency (successful requests):")
        print(f"  Min: {min(latencies):.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")
        print(f"  Mean: {statistics.mean(latencies):.2f}ms")
        print(f"  Median: {statistics.median(latencies):.2f}ms")
        print(f"  Stdev: {statistics.stdev(latencies) if len(latencies) > 1 else 0:.2f}ms")
        print(f"  95th percentile: {sorted(latencies)[int(0.95*len(latencies))]:.2f}ms")

    if failed:
        print(f"\n❌ Failed requests:")
        error_types = {}
        for r in failed:
            error = r.get("status", "unknown")
            error_types[error] = error_types.get(error, 0) + 1
        for error_type, count in error_types.items():
            print(f"  {error_type}: {count}")

    # Classification distribution
    if successful:
        labels = {}
        for r in successful:
            label = r.get("label", "unknown")
            labels[label] = labels.get(label, 0) + 1

        print(f"\n🏷️  Classification Distribution (successful):")
        for label, count in sorted(labels.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count} ({100*count/len(successful):.1f}%)")

    # Comparison with simulated test
    print("\n" + "="*80)
    print("COMPARISON: Real Docker vs Simulated Async Test")
    print("="*80)
    print(f"\n{'Metric':<25} {'Simulated (async)':<20} {'Real Docker':<20} {'Difference':<15}")
    print("-" * 80)

    sim_duration = 1.74
    print(f"{'Duration':<25} {sim_duration:.2f}s{'':<15} {burst_duration:.2f}s{'':<15} +{burst_duration-sim_duration:.2f}s ({100*(burst_duration-sim_duration)/sim_duration:.1f}%)")

    sim_throughput = 172.41
    real_throughput = num_messages/burst_duration
    print(f"{'Throughput':<25} {sim_throughput:.2f} req/s{'':<10} {real_throughput:.2f} req/s{'':<10} {real_throughput-sim_throughput:.2f} req/s ({100*(real_throughput-sim_throughput)/sim_throughput:+.1f}%)")

    if successful:
        sim_max_latency = 68.52
        real_max_latency = max(latencies)
        print(f"{'Max Latency':<25} {sim_max_latency:.2f}ms{'':<15} {real_max_latency:.2f}ms{'':<15} +{real_max_latency-sim_max_latency:.2f}ms ({100*(real_max_latency-sim_max_latency)/sim_max_latency:.1f}%)")

        sim_mean_latency = 10.11  # Approximate from simulation
        real_mean_latency = statistics.mean(latencies)
        print(f"{'Mean Latency':<25} {sim_mean_latency:.2f}ms{'':<15} {real_mean_latency:.2f}ms{'':<15} +{real_mean_latency-sim_mean_latency:.2f}ms ({100*(real_mean_latency-sim_mean_latency)/sim_mean_latency:.1f}%)")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    network_overhead = burst_duration - sim_duration
    print(f"\nNetwork Overhead: +{network_overhead*1000:.0f}ms ({100*network_overhead/sim_duration:.1f}% increase)")

    if network_overhead < 0.05:
        print("✅ EXCELLENT: Real deployment performing nearly identical to simulation!")
    elif network_overhead < 0.1:
        print("✅ GOOD: Real deployment has minimal network overhead (expected 15-20ms)")
    elif network_overhead < 0.2:
        print("⚠️  ACCEPTABLE: Real deployment has moderate network overhead")
    else:
        print("❌ WARNING: Real deployment has significant overhead - check server logs")

    print(f"\nSuccess Rate: {100*len(successful)/num_messages:.1f}%")
    if len(successful) == num_messages:
        print("✅ 100% request success - all servers responsive")
    elif len(successful) > 0.95 * num_messages:
        print("✅ >95% success rate - excellent reliability")
    else:
        print("❌ Low success rate - servers may not be fully ready")

    return {
        "duration": burst_duration,
        "throughput": real_throughput,
        "successful": len(successful),
        "failed": len(failed),
        "max_latency": max(latencies) if latencies else None,
        "mean_latency": statistics.mean(latencies) if latencies else None,
        "timestamp": datetime.now().isoformat(),
    }

if __name__ == "__main__":
    # Run the burst test
    result = asyncio.run(run_burst_test(num_messages=300))

    if result:
        # Save results to file
        with open("burst_test_real_docker_results.json", "w") as f:
            json.dump(result, f, indent=2)
        print("\n✅ Results saved to: burst_test_real_docker_results.json")
