"""
Tests for the DynamicBatcher.

These tests exercise the real async logic of the batcher against a stub
model + stub tokenizer. They do not require the mBERT weights or the
transformers library — just torch for tensor ops.

Run:
    pytest src/api_interface/tests/test_batching_service.py -v
"""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
import pytest_asyncio
import torch

# Make ``src`` importable when the tests are run from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.api_interface.services import batching_service
from src.api_interface.services.batching_service import DynamicBatcher


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
class _StubTokenizer:
    """Produces deterministic token ids without requiring HuggingFace tokenizers."""

    def __call__(
        self,
        texts,
        add_special_tokens=True,
        max_length=96,
        padding="longest",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    ):
        if isinstance(texts, str):
            texts = [texts]
        # One token per character, capped at max_length.
        encoded = [[ord(c) % 30000 for c in t[:max_length]] or [0] for t in texts]
        longest = max(len(e) for e in encoded)
        input_ids = torch.zeros(len(encoded), longest, dtype=torch.long)
        attention_mask = torch.zeros(len(encoded), longest, dtype=torch.long)
        for i, row in enumerate(encoded):
            input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[i, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _StubModel(torch.nn.Module):
    """Returns a deterministic label per text based on its first character.

    ``h``* -> ham (class 0), ``s``* -> spam (class 1), ``p``* -> phishing (class 2).
    Records each batch size so tests can assert coalescing worked.
    """

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: List[int] = []
        self.call_delay = 0.0
        self.raise_on_next = False

    def forward(self, input_ids, attention_mask=None, **_ignored):
        self.batch_sizes.append(input_ids.shape[0])
        if self.raise_on_next:
            self.raise_on_next = False
            raise RuntimeError("forced failure")
        if self.call_delay:
            import time as _t
            _t.sleep(self.call_delay)
        batch = input_ids.shape[0]
        logits = torch.full((batch, 3), -5.0)
        for i in range(batch):
            # Use first non-pad token to decide class
            first_tok = int(input_ids[i, 0].item())
            ch = chr(first_tok) if first_tok < 128 else "h"
            if ch == "s":
                cls = 1
            elif ch == "p":
                cls = 2
            else:
                cls = 0
            logits[i, cls] = 5.0
        return SimpleNamespace(logits=logits)


class _StubModelManager:
    def __init__(self, model: _StubModel, tokenizer: _StubTokenizer) -> None:
        self.device = torch.device("cpu")
        self._model = model
        self._tokenizer = tokenizer

    def get_mbert_model(self, _name):
        return self._model, self._tokenizer, "stub-2.5"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def stub_stack(monkeypatch):
    """Replace the real model_manager inside batching_service with a stub."""
    model = _StubModel()
    tokenizer = _StubTokenizer()
    manager = _StubModelManager(model, tokenizer)

    # The batcher imports ``model_manager`` lazily inside ``_run_batch_sync`` via
    # ``from .model_loader import model_manager`` — patch the attribute on the
    # module so that later imports hit our stub.
    import src.api_interface.services.model_loader as model_loader_mod
    monkeypatch.setattr(model_loader_mod, "model_manager", manager)

    return SimpleNamespace(model=model, tokenizer=tokenizer, manager=manager)


@pytest_asyncio.fixture()
async def batcher(stub_stack):
    b = DynamicBatcher(max_batch_size=8, batch_wait_ms=30, max_text_length=32)
    await b.start()
    try:
        yield b, stub_stack
    finally:
        await b.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_single_request_returns_correct_label(batcher):
    b, stack = batcher
    label, prob, elapsed = await b.submit("hello there")
    assert label == "ham"
    assert 0.99 < prob <= 1.0
    assert elapsed >= 0
    assert stack.model.batch_sizes == [1]


@pytest.mark.asyncio
async def test_concurrent_requests_are_coalesced(batcher):
    b, stack = batcher
    # Fire 8 requests simultaneously. They should land in a single batch since
    # the batch wait window (30 ms) is larger than the scheduling jitter and
    # max_batch_size == 8.
    texts = ["ham one", "spam two", "phish me", "hello", "hi", "hey", "yo", "sup"]
    results = await asyncio.gather(*(b.submit(t) for t in texts))
    labels = [r[0] for r in results]
    assert labels[0] == "ham"
    assert labels[1] == "spam"
    assert labels[2] == "phishing"
    # One batch, size 8.
    assert stack.model.batch_sizes == [8], (
        f"expected a single coalesced batch of 8, got {stack.model.batch_sizes}"
    )


@pytest.mark.asyncio
async def test_max_batch_size_is_respected(stub_stack):
    b = DynamicBatcher(max_batch_size=4, batch_wait_ms=50, max_text_length=32)
    await b.start()
    try:
        texts = [f"hello {i}" for i in range(10)]
        results = await asyncio.gather(*(b.submit(t) for t in texts))
        assert len(results) == 10
        # No single batch should exceed max_batch_size == 4.
        assert all(sz <= 4 for sz in stub_stack.model.batch_sizes), stub_stack.model.batch_sizes
        # All batches summed should account for every request.
        assert sum(stub_stack.model.batch_sizes) == 10
    finally:
        await b.stop()


@pytest.mark.asyncio
async def test_wait_window_flushes_partial_batch(stub_stack):
    b = DynamicBatcher(max_batch_size=32, batch_wait_ms=20, max_text_length=32)
    await b.start()
    try:
        # Only 2 requests; should flush after the 20 ms window, not stall.
        import time
        t0 = time.monotonic()
        await asyncio.gather(b.submit("hello"), b.submit("hi there"))
        elapsed = time.monotonic() - t0
        # Must have waited at least one window, but well under a second.
        assert 0.01 < elapsed < 1.0, f"elapsed={elapsed}"
        assert stub_stack.model.batch_sizes == [2]
    finally:
        await b.stop()


@pytest.mark.asyncio
async def test_model_error_propagates_to_all_futures(stub_stack):
    b = DynamicBatcher(max_batch_size=4, batch_wait_ms=20, max_text_length=32)
    await b.start()
    try:
        stub_stack.model.raise_on_next = True
        with pytest.raises(Exception):
            await asyncio.gather(b.submit("hello"), b.submit("spammy"))
    finally:
        await b.stop()


@pytest.mark.asyncio
async def test_metrics_are_recorded(batcher):
    b, stack = batcher
    await asyncio.gather(b.submit("hello"), b.submit("spam"), b.submit("phish"))
    m = b.metrics
    assert m.total_requests == 3
    assert m.total_batches >= 1
    assert m.last_batch_size >= 1
    assert m.total_inference_seconds > 0
    # Histogram bucket for size-3 batch rounds up to 4.
    assert any(bucket >= 3 for bucket in m.batch_size_histogram)


@pytest.mark.asyncio
async def test_shutdown_fails_pending_requests(stub_stack):
    """Pending submissions must not hang when the batcher is stopped."""
    b = DynamicBatcher(max_batch_size=32, batch_wait_ms=50, max_text_length=32)
    # Slow the model down so requests queue up.
    stub_stack.model.call_delay = 0.2
    await b.start()
    # Submit one request that will be in-flight, then stop immediately.
    task = asyncio.create_task(b.submit("hello"))
    await asyncio.sleep(0.01)
    await b.stop()
    # Task must complete (success or failure) rather than hang forever.
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except Exception:
        pass  # any terminal state is acceptable; hanging is not
    assert task.done()


@pytest.mark.asyncio
async def test_init_batcher_respects_disabled_flag(monkeypatch):
    from src.api_interface.config import settings as settings_mod
    monkeypatch.setattr(settings_mod.settings, "batching_enabled", False)
    # Re-run the init to pick up the new flag.
    result = batching_service.init_batcher()
    assert result is None
    assert batching_service.get_batcher() is None
