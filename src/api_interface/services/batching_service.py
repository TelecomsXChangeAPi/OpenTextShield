"""
Dynamic batching service for OpenTextShield API.

Coalesces concurrent single-message prediction requests into padded batches
that are fed to the mBERT model in a single forward pass. This raises GPU
utilization dramatically (a T4 processing 32 SMS in one batch costs roughly
the same wall time as a single message, so per-message throughput scales
near-linearly with batch size up to saturation).

Design:
  - A single asyncio background worker owns the model and drains a queue.
  - Callers submit (text, Future) tuples via ``submit()`` and await the Future.
  - The worker collects up to ``max_batch_size`` items within a ``batch_wait_ms``
    window (whichever limit hits first), runs one forward pass, and resolves
    each Future with its own (label, probability).
  - Tokenization uses ``padding='longest'`` so a batch of short SMS is not
    padded out to the full max length.

This module is import-safe even when torch is unavailable; the batcher is a
no-op until ``start()`` is called from the FastAPI lifespan handler.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..config.settings import settings
from ..utils.logging import logger
from ..utils.exceptions import PredictionError


@dataclass
class _PendingRequest:
    """A single in-flight prediction request waiting for its batch."""
    text: str
    future: "asyncio.Future[Tuple[str, float, float]]"
    enqueued_at: float


@dataclass
class BatchingMetrics:
    """Runtime metrics exposed via /metrics."""
    total_requests: int = 0
    total_batches: int = 0
    total_inference_seconds: float = 0.0
    total_wait_seconds: float = 0.0
    batch_size_histogram: Dict[int, int] = field(default_factory=dict)
    last_batch_size: int = 0
    current_queue_depth: int = 0
    errors: int = 0

    def record_batch(self, size: int, wait_seconds_sum: float, inference_seconds: float) -> None:
        self.total_requests += size
        self.total_batches += 1
        self.total_inference_seconds += inference_seconds
        self.total_wait_seconds += wait_seconds_sum
        self.last_batch_size = size
        bucket = self._bucket_for(size)
        self.batch_size_histogram[bucket] = self.batch_size_histogram.get(bucket, 0) + 1

    @staticmethod
    def _bucket_for(size: int) -> int:
        # Power-of-two buckets (1, 2, 4, 8, 16, 32, 64...)
        if size <= 1:
            return 1
        bucket = 1
        while bucket < size:
            bucket *= 2
        return bucket


class DynamicBatcher:
    """Coalesces concurrent prediction requests into padded GPU batches."""

    def __init__(
        self,
        max_batch_size: int,
        batch_wait_ms: int,
        max_text_length: int,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.batch_wait_seconds = batch_wait_ms / 1000.0
        self.max_text_length = max_text_length

        self._queue: "asyncio.Queue[_PendingRequest]" = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        self._stopping = asyncio.Event()
        self._label_map = {0: "ham", 1: "spam", 2: "phishing"}
        self.metrics = BatchingMetrics()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Launch the background worker. Idempotent."""
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._stopping.clear()
        self._worker_task = asyncio.create_task(self._worker_loop(), name="ots-batcher")
        logger.info(
            "DynamicBatcher started (max_batch_size=%d, batch_wait_ms=%.0f, max_text_length=%d)",
            self.max_batch_size,
            self.batch_wait_seconds * 1000,
            self.max_text_length,
        )

    async def stop(self) -> None:
        """Signal the worker to drain and exit. Pending requests are failed."""
        self._stopping.set()
        if self._worker_task is not None:
            try:
                await asyncio.wait_for(self._worker_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._worker_task.cancel()
        # Fail any still-queued requests so callers do not hang.
        while not self._queue.empty():
            try:
                pending = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not pending.future.done():
                pending.future.set_exception(
                    PredictionError({"error": "batcher shutting down"})
                )
        logger.info("DynamicBatcher stopped")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def submit(self, text: str) -> Tuple[str, float, float]:
        """Submit a text and await the (label, probability, processing_time) result."""
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[Tuple[str, float, float]]" = loop.create_future()
        pending = _PendingRequest(text=text, future=future, enqueued_at=time.monotonic())
        await self._queue.put(pending)
        self.metrics.current_queue_depth = self._queue.qsize()
        return await future

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------
    async def _worker_loop(self) -> None:
        """Pull items off the queue, form batches, run inference."""
        while not self._stopping.is_set():
            try:
                batch = await self._collect_batch()
                if not batch:
                    continue
                await asyncio.get_running_loop().run_in_executor(
                    None, self._run_batch_sync, batch
                )
            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Batcher worker crashed: %s", exc, exc_info=True)
                self.metrics.errors += 1

    async def _collect_batch(self) -> List[_PendingRequest]:
        """Collect up to max_batch_size items within the wait window."""
        try:
            first = await asyncio.wait_for(self._queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return []

        batch: List[_PendingRequest] = [first]
        deadline = time.monotonic() + self.batch_wait_seconds

        while len(batch) < self.max_batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            batch.append(item)

        self.metrics.current_queue_depth = self._queue.qsize()
        return batch

    def _run_batch_sync(self, batch: List[_PendingRequest]) -> None:
        """Tokenize and run a single forward pass for the whole batch."""
        # Imported lazily to avoid a circular import at module load time.
        from .model_loader import model_manager

        start = time.monotonic()
        try:
            model, tokenizer, model_version = model_manager.get_mbert_model(
                settings.default_mbert_version
            )
        except Exception as exc:
            self._fail_all(batch, exc)
            return

        texts = [item.text for item in batch]

        try:
            inputs = tokenizer(
                texts,
                add_special_tokens=True,
                max_length=self.max_text_length,
                padding="longest",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(model_manager.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)
                logits = outputs.logits.float()  # cast back from fp16 for softmax stability
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

            predictions_list = predictions.tolist()
            probabilities_list = probabilities.tolist()

        except Exception as exc:
            logger.error("Batch inference failed (size=%d): %s", len(batch), exc)
            self._fail_all(batch, exc)
            return

        inference_seconds = time.monotonic() - start
        finish_time = time.monotonic()
        wait_seconds_sum = sum(finish_time - item.enqueued_at for item in batch)

        for idx, item in enumerate(batch):
            pred = predictions_list[idx]
            probs = probabilities_list[idx]
            label = self._label_map.get(pred, "ham")
            probability = float(probs[pred])
            per_request_time = finish_time - item.enqueued_at
            if not item.future.done():
                item.future.get_loop().call_soon_threadsafe(
                    item.future.set_result, (label, probability, per_request_time)
                )

        self.metrics.record_batch(
            size=len(batch),
            wait_seconds_sum=wait_seconds_sum,
            inference_seconds=inference_seconds,
        )

        logger.info(
            "batch inference: size=%d inference_time=%.3fs avg_wait=%.3fs version=%s",
            len(batch),
            inference_seconds,
            wait_seconds_sum / len(batch),
            model_version,
        )

    @staticmethod
    def _fail_all(batch: List[_PendingRequest], exc: Exception) -> None:
        for item in batch:
            if not item.future.done():
                item.future.get_loop().call_soon_threadsafe(
                    item.future.set_exception,
                    PredictionError({"error": str(exc)}),
                )


# Global batcher instance, constructed lazily once settings are available.
_batcher: Optional[DynamicBatcher] = None


def get_batcher() -> Optional[DynamicBatcher]:
    """Return the active batcher, or None if batching is disabled."""
    return _batcher


def init_batcher() -> Optional[DynamicBatcher]:
    """Construct (but do not start) the batcher based on settings."""
    global _batcher
    if not settings.batching_enabled:
        logger.info("Dynamic batching disabled via settings")
        _batcher = None
        return None
    _batcher = DynamicBatcher(
        max_batch_size=settings.max_batch_size,
        batch_wait_ms=settings.batch_wait_ms,
        max_text_length=settings.max_text_length,
    )
    return _batcher
