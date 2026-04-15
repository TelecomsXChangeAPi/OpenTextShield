"""
Tests for the /metrics Prometheus endpoint.

These tests spin up the FastAPI app in-memory and verify that the rendered
exposition format contains the expected counters and gauges.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_app(batcher=None):
    """Construct a minimal FastAPI app with just the metrics router mounted."""
    from src.api_interface.routers import metrics as metrics_router
    from src.api_interface.services import batching_service

    batching_service._batcher = batcher  # inject stub batcher (or None)
    app = FastAPI()
    app.include_router(metrics_router.router)
    return app


def test_metrics_endpoint_disabled_batcher():
    client = TestClient(_build_app(batcher=None))
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "ots_api_info" in body
    assert "ots_batching_enabled 0" in body
    # Must advertise text/plain for Prometheus to scrape it.
    assert resp.headers["content-type"].startswith("text/plain")


def test_metrics_endpoint_active_batcher():
    from src.api_interface.services.batching_service import DynamicBatcher

    b = DynamicBatcher(max_batch_size=16, batch_wait_ms=20, max_text_length=64)
    # Simulate that two batches of sizes 4 and 8 have been processed.
    b.metrics.record_batch(size=4, wait_seconds_sum=0.04, inference_seconds=0.02)
    b.metrics.record_batch(size=8, wait_seconds_sum=0.12, inference_seconds=0.03)
    b.metrics.current_queue_depth = 2

    client = TestClient(_build_app(batcher=b))
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text

    # Required series present.
    for name in [
        "ots_batching_enabled 1",
        "ots_batch_max_size 16",
        "ots_requests_total 12",  # 4 + 8
        "ots_batches_total 2",
        "ots_queue_depth 2",
        "ots_last_batch_size 8",
        "ots_batch_size_bucket",
    ]:
        assert name in body, f"expected '{name}' in metrics output:\n{body}"

    # Histogram lines must be well-formed Prometheus counters.
    hist_lines = [ln for ln in body.splitlines() if ln.startswith("ots_batch_size_bucket{")]
    assert hist_lines, "no histogram buckets emitted"
    for line in hist_lines:
        assert 'le="' in line
