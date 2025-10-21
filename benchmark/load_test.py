#!/usr/bin/env python3
"""
Load testing script for SMS classification systems
Tests concurrent request handling and throughput
"""

import json
import time
import asyncio
import aiohttp
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import statistics

# Configuration
TEST_DATASET_FILE = Path(__file__).parent / "test_dataset.json"
LOAD_TEST_RESULTS_FILE_PREFIX = Path(__file__).parent / "results_load_test"

# Test configurations
LOAD_TEST_CONFIGS = [
    {"name": "Light Load", "concurrent_requests": 10, "total_requests": 100},
    {"name": "Medium Load", "concurrent_requests": 50, "total_requests": 500},
    {"name": "Heavy Load", "concurrent_requests": 100, "total_requests": 1000},
    {"name": "SMSC Realistic (100 MPS)", "concurrent_requests": 100, "duration_seconds": 10},
    {"name": "SMSC Peak (400 MPS)", "concurrent_requests": 400, "duration_seconds": 5},
]

def load_test_dataset() -> Dict:
    """Load the test dataset."""
    with open(TEST_DATASET_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

async def classify_text_ots(session: aiohttp.ClientSession, text: str) -> Dict:
    """Classify text using OpenTextShield API."""
    start_time = time.time()
    try:
        async with session.post(
            "http://localhost:8002/predict/",
            json={"text": text, "model": "ots-mbert"},
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            elapsed = time.time() - start_time
            if response.status == 200:
                result = await response.json()
                return {
                    "success": True,
                    "response_time": elapsed,
                    "label": result.get("label"),
                    "probability": result.get("probability"),
                    "model_processing_time": result.get("processing_time", 0)
                }
            else:
                return {
                    "success": False,
                    "response_time": elapsed,
                    "error": f"HTTP {response.status}"
                }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "response_time": elapsed,
            "error": str(e)
        }

async def classify_text_gpt(session: aiohttp.ClientSession, text: str, system_prompt: str) -> Dict:
    """Classify text using GPT-OSS-20B API."""
    start_time = time.time()
    try:
        payload = {
            "model": "openai/gpt-oss-20b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.3,
            "max_tokens": 500,
            "stream": False
        }

        async with session.post(
            "http://0.0.0.0:1234/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            elapsed = time.time() - start_time
            if response.status == 200:
                result = await response.json()
                content = result['choices'][0]['message']['content']

                # Parse JSON from content
                content_clean = content.strip()
                if content_clean.startswith('```json'):
                    content_clean = content_clean[7:]
                if content_clean.startswith('```'):
                    content_clean = content_clean[3:]
                if content_clean.endswith('```'):
                    content_clean = content_clean[:-3]
                content_clean = content_clean.strip()

                parsed = json.loads(content_clean)

                return {
                    "success": True,
                    "response_time": elapsed,
                    "label": parsed.get("label"),
                    "probability": parsed.get("probability")
                }
            else:
                return {
                    "success": False,
                    "response_time": elapsed,
                    "error": f"HTTP {response.status}"
                }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "response_time": elapsed,
            "error": str(e)
        }

async def run_load_test_batch(
    api_type: str,
    test_messages: List[str],
    concurrent_requests: int,
    total_requests: int = None,
    duration_seconds: int = None,
    system_prompt: str = None
) -> Dict:
    """Run a single load test configuration."""

    results = {
        "successful_requests": 0,
        "failed_requests": 0,
        "response_times": [],
        "model_processing_times": [],
        "errors": []
    }

    connector = aiohttp.TCPConnector(limit=concurrent_requests)
    async with aiohttp.ClientSession(connector=connector) as session:

        if total_requests:
            # Fixed number of requests
            tasks = []
            for i in range(total_requests):
                text = test_messages[i % len(test_messages)]
                if api_type == "opentextshield":
                    task = classify_text_ots(session, text)
                else:
                    task = classify_text_gpt(session, text, system_prompt)
                tasks.append(task)

            # Execute with concurrency limit
            start_time = time.time()
            for i in range(0, len(tasks), concurrent_requests):
                batch = tasks[i:i + concurrent_requests]
                batch_results = await asyncio.gather(*batch)

                for result in batch_results:
                    if result["success"]:
                        results["successful_requests"] += 1
                        results["response_times"].append(result["response_time"])
                        if "model_processing_time" in result:
                            results["model_processing_times"].append(result["model_processing_time"])
                    else:
                        results["failed_requests"] += 1
                        results["errors"].append(result.get("error", "Unknown"))

            total_time = time.time() - start_time

        else:
            # Duration-based testing
            start_time = time.time()
            request_count = 0

            while time.time() - start_time < duration_seconds:
                tasks = []
                for _ in range(concurrent_requests):
                    text = test_messages[request_count % len(test_messages)]
                    if api_type == "opentextshield":
                        task = classify_text_ots(session, text)
                    else:
                        task = classify_text_gpt(session, text, system_prompt)
                    tasks.append(task)
                    request_count += 1

                batch_results = await asyncio.gather(*tasks)

                for result in batch_results:
                    if result["success"]:
                        results["successful_requests"] += 1
                        results["response_times"].append(result["response_time"])
                        if "model_processing_time" in result:
                            results["model_processing_times"].append(result["model_processing_time"])
                    else:
                        results["failed_requests"] += 1
                        results["errors"].append(result.get("error", "Unknown"))

            total_time = time.time() - start_time

    # Calculate statistics
    total_requests_made = results["successful_requests"] + results["failed_requests"]

    return {
        "total_time": total_time,
        "total_requests": total_requests_made,
        "successful_requests": results["successful_requests"],
        "failed_requests": results["failed_requests"],
        "success_rate": results["successful_requests"] / total_requests_made * 100 if total_requests_made > 0 else 0,
        "throughput": total_requests_made / total_time if total_time > 0 else 0,
        "avg_response_time": statistics.mean(results["response_times"]) if results["response_times"] else 0,
        "median_response_time": statistics.median(results["response_times"]) if results["response_times"] else 0,
        "p95_response_time": statistics.quantiles(results["response_times"], n=20)[18] if len(results["response_times"]) > 20 else 0,
        "p99_response_time": statistics.quantiles(results["response_times"], n=100)[98] if len(results["response_times"]) > 100 else 0,
        "min_response_time": min(results["response_times"]) if results["response_times"] else 0,
        "max_response_time": max(results["response_times"]) if results["response_times"] else 0,
        "avg_model_processing_time": statistics.mean(results["model_processing_times"]) if results["model_processing_times"] else None,
        "error_types": dict(zip(*[results["errors"], [results["errors"].count(e) for e in set(results["errors"])]])) if results["errors"] else {}
    }

async def run_all_load_tests(api_type: str, system_prompt: str = None):
    """Run all load test configurations for a given API."""

    print("=" * 80)
    print(f"Load Testing: {api_type.upper()}")
    print("=" * 80)
    print()

    # Load test messages
    dataset = load_test_dataset()
    test_messages = [sample["text"] for sample in dataset["samples"]]

    all_results = {
        "metadata": {
            "api_type": api_type,
            "test_date": datetime.utcnow().isoformat() + "Z",
            "platform": "M4 Mac Mini",
            "test_dataset_size": len(test_messages)
        },
        "tests": []
    }

    for config in LOAD_TEST_CONFIGS:
        print(f"\nRunning: {config['name']}")
        print(f"  Concurrent Requests: {config['concurrent_requests']}")

        if "total_requests" in config:
            print(f"  Total Requests: {config['total_requests']}")
            result = await run_load_test_batch(
                api_type,
                test_messages,
                config['concurrent_requests'],
                total_requests=config['total_requests'],
                system_prompt=system_prompt
            )
        else:
            print(f"  Duration: {config['duration_seconds']}s")
            result = await run_load_test_batch(
                api_type,
                test_messages,
                config['concurrent_requests'],
                duration_seconds=config['duration_seconds'],
                system_prompt=system_prompt
            )

        test_result = {
            "test_name": config['name'],
            "config": config,
            "results": result
        }

        all_results["tests"].append(test_result)

        print(f"\n  Results:")
        print(f"    Total Time: {result['total_time']:.2f}s")
        print(f"    Total Requests: {result['total_requests']}")
        print(f"    Successful: {result['successful_requests']}")
        print(f"    Failed: {result['failed_requests']}")
        print(f"    Success Rate: {result['success_rate']:.2f}%")
        print(f"    Throughput: {result['throughput']:.2f} requests/second")
        print(f"    Avg Response Time: {result['avg_response_time']:.3f}s")
        print(f"    Median Response Time: {result['median_response_time']:.3f}s")
        print(f"    P95 Response Time: {result['p95_response_time']:.3f}s")
        print(f"    P99 Response Time: {result['p99_response_time']:.3f}s")
        if result['avg_model_processing_time']:
            print(f"    Avg Model Processing: {result['avg_model_processing_time']:.3f}s")

    # Save results
    filename = f"{LOAD_TEST_RESULTS_FILE_PREFIX}_{api_type}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {filename}")
    print("=" * 80)

    return all_results

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python load_test.py [opentextshield|gpt-oss]")
        sys.exit(1)

    api_type = sys.argv[1].lower()

    if api_type not in ["opentextshield", "gpt-oss"]:
        print("Error: API type must be 'opentextshield' or 'gpt-oss'")
        sys.exit(1)

    # Load system prompt for GPT-OSS
    system_prompt = None
    if api_type == "gpt-oss":
        prompt_file = Path(__file__).parent / "opentextshield_prompt.txt"
        with open(prompt_file, 'r') as f:
            system_prompt = f.read()

    asyncio.run(run_all_load_tests(api_type, system_prompt))
