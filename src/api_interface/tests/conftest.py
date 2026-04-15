"""Pytest configuration for the API test suite."""

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-mark all coroutine tests so pytest-asyncio picks them up."""
    for item in items:
        if item.get_closest_marker("asyncio") is None:
            test_fn = getattr(item, "function", None)
            if test_fn is not None and _is_coroutine(test_fn):
                item.add_marker(pytest.mark.asyncio)


def _is_coroutine(fn):
    import inspect
    return inspect.iscoroutinefunction(fn)
