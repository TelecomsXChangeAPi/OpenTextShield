"""
Prometheus-compatible metrics endpoint.

Exposes batcher and model throughput counters in the text-based exposition
format so operators (including the ots-bridge) can scrape per-instance GPU
utilisation, batch efficiency, and queue pressure without pulling in an
extra client dependency.
"""

from fastapi import APIRouter, Response

from ..services.batching_service import get_batcher
from ..services.model_loader import model_manager
from ..config.settings import settings

router = APIRouter(tags=["Metrics"])

_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _render() -> str:
    lines = []

    def emit(name: str, help_text: str, type_: str, value: float, labels: str = "") -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {type_}")
        if labels:
            lines.append(f"{name}{{{labels}}} {value}")
        else:
            lines.append(f"{name} {value}")

    emit(
        "ots_api_info",
        "Static build info.",
        "gauge",
        1,
        labels=(
            f'version="{settings.api_version}",'
            f'device="{model_manager.device.type}",'
            f'fp16="{str(settings.use_fp16 and model_manager.device.type == "cuda").lower()}",'
            f'max_text_length="{settings.max_text_length}"'
        ),
    )

    batcher = get_batcher()
    if batcher is None:
        emit(
            "ots_batching_enabled",
            "1 if dynamic batching is active.",
            "gauge",
            0,
        )
        return "\n".join(lines) + "\n"

    m = batcher.metrics
    emit("ots_batching_enabled", "1 if dynamic batching is active.", "gauge", 1)
    emit(
        "ots_batch_max_size",
        "Configured maximum batch size.",
        "gauge",
        batcher.max_batch_size,
    )
    emit(
        "ots_batch_wait_seconds",
        "Configured max wait window before flushing a partial batch.",
        "gauge",
        batcher.batch_wait_seconds,
    )
    emit(
        "ots_requests_total",
        "Total prediction requests processed through the batcher.",
        "counter",
        m.total_requests,
    )
    emit(
        "ots_batches_total",
        "Total batches executed.",
        "counter",
        m.total_batches,
    )
    emit(
        "ots_inference_seconds_total",
        "Total wall-clock seconds spent in model forward passes.",
        "counter",
        round(m.total_inference_seconds, 6),
    )
    emit(
        "ots_request_wait_seconds_total",
        "Sum of end-to-end wait time (queue + inference) across all requests.",
        "counter",
        round(m.total_wait_seconds, 6),
    )
    emit(
        "ots_queue_depth",
        "Current number of requests waiting in the batcher queue.",
        "gauge",
        m.current_queue_depth,
    )
    emit(
        "ots_effective_arrival_rate_msgs_per_second",
        "Observed arrival rate (msgs/sec) averaged over the last 60s. "
        "Separates bridge-limited from GPU-limited scenarios.",
        "gauge",
        round(m.rolling_arrival_rate(), 3),
    )
    emit(
        "ots_arrival_rate_lifetime_msgs_per_second",
        "Observed arrival rate (msgs/sec) averaged since process start.",
        "gauge",
        round(m.lifetime_arrival_rate(), 3),
    )
    emit(
        "ots_last_batch_size",
        "Size of the most recently executed batch.",
        "gauge",
        m.last_batch_size,
    )
    emit(
        "ots_batch_errors_total",
        "Total batches that failed with an uncaught exception.",
        "counter",
        m.errors,
    )

    # Histogram-style bucket counters for batch sizes (power-of-two buckets).
    lines.append("# HELP ots_batch_size_bucket Number of batches per power-of-two size bucket.")
    lines.append("# TYPE ots_batch_size_bucket counter")
    for bucket, count in sorted(m.batch_size_histogram.items()):
        lines.append(f'ots_batch_size_bucket{{le="{bucket}"}} {count}')

    return "\n".join(lines) + "\n"


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    description="Prometheus-compatible metrics scrape endpoint.",
    response_class=Response,
)
async def metrics_endpoint() -> Response:
    return Response(content=_render(), media_type=_CONTENT_TYPE)
