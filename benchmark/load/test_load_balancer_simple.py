"""
Simple Load Balancing Test - Tests with just 20-50 requests to verify
that nginx is distributing traffic across multiple API servers.
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime

LOAD_BALANCER_URL = "http://localhost:8002"
PREDICTION_URL = f"{LOAD_BALANCER_URL}/predict/"

TEST_MESSAGES = [
    "Click here to claim your free prize!",
    "Your account has been compromised.",
    "Hi, how are you doing today?",
    "Congratulations! You've won $1,000,000!",
    "Mom, are you free this weekend?",
    "URGENT: Confirm your banking details now!",
    "Let's catch up soon!",
    "You've been selected for a special offer.",
    "See you at the meeting tomorrow.",
    "Act now before this offer expires!",
]

async def test_load_balancer(num_requests=30):
    """Test load balancer with concurrent requests."""

    print("\n" + "="*80)
    print(f"LOAD BALANCER TEST - {num_requests} Simultaneous Requests")
    print("="*80)

    # Prepare requests
    requests_list = []
    for i in range(num_requests):
        message = TEST_MESSAGES[i % len(TEST_MESSAGES)]
        requests_list.append((i, message))

    print(f"\n📤 Sending {num_requests} simultaneous requests to {LOAD_BALANCER_URL}")

    start_time = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = []
        for msg_id, text in requests_list:
            payload = {"text": text, "model": "ots-mbert"}
            task = session.post(
                PREDICTION_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            )
            tasks.append(task)

        # Send all requests
        responses = []
        for task in asyncio.as_completed(tasks):
            try:
                async with await task as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        responses.append({
                            "status": "success",
                            "label": data.get("label"),
                            "latency": (time.time() - start_time) * 1000
                        })
                    else:
                        responses.append({
                            "status": f"error_{resp.status}",
                            "latency": (time.time() - start_time) * 1000
                        })
            except Exception as e:
                responses.append({
                    "status": f"exception: {str(e)}",
                    "latency": (time.time() - start_time) * 1000
                })

    total_time = time.time() - start_time

    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    successful = [r for r in responses if r["status"] == "success"]
    failed = [r for r in responses if r["status"] != "success"]

    print(f"\n✅ Successful: {len(successful)}/{num_requests}")
    print(f"❌ Failed: {len(failed)}/{num_requests}")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📊 Throughput: {num_requests/total_time:.2f} req/s")

    if successful:
        labels = {}
        for r in successful:
            label = r["label"]
            labels[label] = labels.get(label, 0) + 1

        print(f"\n🏷️  Classification Results:")
        for label, count in sorted(labels.items()):
            print(f"  {label}: {count}")

    if failed:
        print(f"\n⚠️  Failed requests:")
        for r in failed:
            print(f"  {r['status']}")

    return {
        "total_requests": num_requests,
        "successful": len(successful),
        "failed": len(failed),
        "duration": total_time,
        "throughput": num_requests / total_time,
    }

if __name__ == "__main__":
    result = asyncio.run(test_load_balancer(num_requests=30))

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Status: {'✅ PASS' if result['successful'] == result['total_requests'] else '⚠️  PARTIAL'}")
    print(f"Success Rate: {100*result['successful']/result['total_requests']:.1f}%")
    print(f"Duration: {result['duration']:.2f}s")
    print(f"Throughput: {result['throughput']:.2f} req/s")

    with open("load_balancer_test_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n✅ Results saved to: load_balancer_test_results.json")
