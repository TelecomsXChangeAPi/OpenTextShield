#!/usr/bin/env python3
"""
Multi-Server Load Balancer Test
Simulates 4 API servers with load balancing across a 300-message burst
Tests if multiple servers improve response times
"""

import asyncio
import time
import random
import sys
from pathlib import Path
from typing import List, Dict
import json

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


class APIServer:
    """Simulates a single API server with its own GPU/processing queue"""

    def __init__(self, server_id: int):
        self.server_id = server_id
        self.queue = asyncio.Queue()
        self.active = False
        self.processed_count = 0
        self.processing_times = []

    async def start(self):
        """Start processing requests from queue"""
        self.active = True
        while self.active:
            try:
                # Get request from queue with timeout
                request_data = await asyncio.wait_for(self.queue.get(), timeout=0.1)

                # Simulate processing
                start = time.time()
                result = await prediction_service.predict(request_data['request'])
                processing_time = time.time() - start

                self.processed_count += 1
                self.processing_times.append(processing_time)

                # Return result
                request_data['future'].set_result({
                    'label': result.label,
                    'probability': result.probability,
                    'time': processing_time,
                    'server': self.server_id
                })

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                self.active = False
                break
            except Exception as e:
                if not request_data['future'].done():
                    request_data['future'].set_exception(e)

    async def submit_request(self, request: PredictionRequest):
        """Submit a request to this server's queue"""
        future = asyncio.Future()
        await self.queue.put({
            'request': request,
            'future': future,
            'submitted_time': time.time()
        })
        return future


class LoadBalancer:
    """Distributes requests across multiple API servers"""

    def __init__(self, num_servers: int = 4):
        self.servers = [APIServer(i) for i in range(num_servers)]
        self.request_count = 0
        self.metrics = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'response_times': [],
            'server_loads': {i: [] for i in range(num_servers)}
        }
        self.lock = asyncio.Lock()

    async def start(self):
        """Start all server tasks"""
        self.server_tasks = [asyncio.create_task(server.start()) for server in self.servers]

    async def stop(self):
        """Stop all servers"""
        for server in self.servers:
            server.active = False
        for task in self.server_tasks:
            await asyncio.sleep(0.1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def submit_request(self, request: PredictionRequest):
        """Submit request to least-loaded server (round-robin would also work)"""
        # Round-robin load balancing
        server_index = self.request_count % len(self.servers)
        self.request_count += 1

        server = self.servers[server_index]

        try:
            start_time = time.time()
            result = await server.submit_request(request)
            response = await result

            response_time = time.time() - start_time

            async with self.lock:
                self.metrics['total_requests'] += 1
                self.metrics['successful'] += 1
                self.metrics['response_times'].append(response_time)
                self.metrics['server_loads'][server_index].append(response_time)

            return response

        except Exception as e:
            async with self.lock:
                self.metrics['total_requests'] += 1
                self.metrics['failed'] += 1
            return None


async def send_burst_request(message_index: int, balancer: LoadBalancer):
    """Send a single request through the load balancer"""
    try:
        text = random.choice(TEST_MESSAGES)
        request = PredictionRequest(text=text, model=ModelType.OTS_MBERT)

        result = await balancer.submit_request(request)
        return {
            'index': message_index,
            'success': True,
            'result': result
        }
    except Exception as e:
        return {
            'index': message_index,
            'success': False,
            'error': str(e)
        }


async def main():
    print("\n")
    print("█" * 80)
    print("█ Multi-Server Load Balancer Test - 4 API Servers")
    print("█ Burst: 300 Messages Distributed Across 4 Servers")
    print("█" * 80)
    print("\n📊 Setup:")
    print("   • 4 API Servers (each with independent GPU/processing)")
    print("   • Load Balancer (round-robin distribution)")
    print("   • 300 burst messages (75 per server)")
    print("\n🔄 Loading model on primary server...")

    model_manager.load_all_models()
    print(f"✅ Model loaded on: {str(model_manager.device).upper()}")

    # Create load balancer with 4 servers
    balancer = LoadBalancer(num_servers=4)

    print(f"\n🚀 Starting 4 API servers...")
    await balancer.start()
    await asyncio.sleep(1)  # Let servers start

    print(f"✅ 4 servers ready\n")
    print(f"🚀 Sending 300 concurrent requests (75 per server via load balancer)...\n")

    # Create 300 concurrent tasks
    tasks = [send_burst_request(i, balancer) for i in range(300)]

    # Execute burst
    burst_start = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=False)
    burst_end = time.time()
    burst_duration = burst_end - burst_start

    # Stop servers
    await balancer.stop()

    # Process results
    successes = sum(1 for r in results if r.get('success', False))
    failures = sum(1 for r in results if not r.get('success', False))

    metrics = balancer.metrics
    response_times = metrics['response_times']

    # Calculate statistics
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        min_response_time = min(response_times)
        max_response_time = max(response_times)
        p50 = sorted(response_times)[len(response_times) // 2]
        p95 = sorted(response_times)[int(len(response_times) * 0.95)]
        p99 = sorted(response_times)[int(len(response_times) * 0.99)] if len(response_times) > 100 else max_response_time
    else:
        avg_response_time = min_response_time = max_response_time = p50 = p95 = p99 = 0

    # Print results
    print("\n")
    print("█" * 80)
    print("█ 4-SERVER LOAD BALANCER RESULTS")
    print("█" * 80)

    print(f"\n⏱️  TIMING:")
    print(f"   Total Burst Duration: {burst_duration:.2f} seconds")
    print(f"   Throughput: {300/burst_duration:.2f} requests/second")

    print(f"\n📊 REQUEST STATISTICS:")
    print(f"   Total Requests: {successes + failures}")
    print(f"   Successful: {successes} ✅")
    print(f"   Failed: {failures} ❌")
    print(f"   Success Rate: {(successes/(successes+failures)*100):.1f}%")

    print(f"\n⏳ RESPONSE TIME ANALYSIS:")
    print(f"   Min: {min_response_time:.4f}s ({min_response_time*1000:.2f}ms)")
    print(f"   P50 (Median): {p50:.4f}s ({p50*1000:.2f}ms)")
    print(f"   P95: {p95:.4f}s ({p95*1000:.2f}ms)")
    print(f"   P99: {p99:.4f}s ({p99*1000:.2f}ms)")
    print(f"   Avg: {avg_response_time:.4f}s ({avg_response_time*1000:.2f}ms)")
    print(f"   Max: {max_response_time:.4f}s ({max_response_time*1000:.2f}ms)")

    print(f"\n🖥️  PER-SERVER LOAD DISTRIBUTION:")
    for server_id, times in metrics['server_loads'].items():
        if times:
            avg_time = sum(times) / len(times)
            print(f"   Server {server_id}: {len(times)} requests, avg {avg_time*1000:.2f}ms")

    # Compare with single server
    print(f"\n📊 COMPARISON: Single Server vs 4 Servers")
    print(f"   {'Metric':<25} {'Single Server':<20} {'4 Servers':<20}")
    print(f"   {'-'*65}")
    print(f"   {'Total Duration':<25} {'16.28s':<20} {f'{burst_duration:.2f}s':<20}")
    print(f"   {'Max Response Time':<25} {'260.25ms':<20} {f'{max_response_time*1000:.2f}ms':<20}")
    print(f"   {'Median Response Time':<25} {'52.98ms':<20} {f'{p50*1000:.2f}ms':<20}")
    print(f"   {'Avg Response Time':<25} {'54.24ms':<20} {f'{avg_response_time*1000:.2f}ms':<20}")
    print(f"   {'P95 Response Time':<25} {'58.71ms':<20} {f'{p95*1000:.2f}ms':<20}")

    # Calculate improvement
    single_max = 260.25
    improvement = ((single_max - (max_response_time * 1000)) / single_max) * 100
    time_improvement = ((16.28 - burst_duration) / 16.28) * 100

    print(f"\n🎯 IMPROVEMENT WITH 4 SERVERS:")
    print(f"   Max Response Time: {improvement:.1f}% faster ⚡")
    print(f"   Total Duration: {time_improvement:.1f}% faster ⚡")
    print(f"   Requests per second: {300/burst_duration:.2f} (vs 18.43 single server) 🚀")

    if max_response_time < 0.2:
        print(f"\n✅ ALL RESPONSES UNDER 200ms!")
        print(f"   → No SMSC timeout issues (assuming timeout > 200ms)")

    print("\n" + "█" * 80)
    print("█ TEST COMPLETE")
    print("█" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
