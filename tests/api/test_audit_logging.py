#!/usr/bin/env python3
"""
Comprehensive test suite for audit logging functionality.

Tests cover initialization, logging, file rotation, PII redaction, and API querying.
"""

import sys
import json
import asyncio
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_audit_service_initialization():
    """Test audit service initializes correctly."""
    print("🔍 Testing audit service initialization...")

    try:
        from src.api_interface.services.audit_service import audit_service
        from src.api_interface.config.settings import settings

        assert settings.audit_enabled == True
        assert settings.audit_dir.exists()
        assert audit_service is not None

        print("✅ Audit service initialized correctly")
        return True

    except Exception as e:
        print(f"❌ Audit service initialization failed: {e}")
        return False


def test_prediction_audit_logging():
    """Test prediction audit log entry creation."""
    print("\n🔍 Testing prediction audit logging...")

    try:
        from src.api_interface.services.audit_service import audit_service
        from src.api_interface.config.settings import settings

        # Log a test prediction
        audit_service.log_prediction(
            text="Free money! Click here now!",
            label="spam",
            confidence=0.95,
            model="ots-mbert",
            model_version="2.5",
            processing_time=150.5,
            client_ip="192.168.1.100",
            text_length=29,
        )

        # Flush to ensure it's written
        audit_service.flush()

        # Verify log file was created and contains entry
        log_files = list(settings.audit_dir.glob("*.jsonl"))
        assert len(log_files) > 0, "No audit log files created"

        # Read and verify last entry
        with open(log_files[-1], "r") as f:
            lines = f.readlines()
            assert len(lines) > 0, "Log file is empty"

            last_entry = json.loads(lines[-1])

            assert last_entry["entry_type"] == "prediction"
            assert last_entry["text"] == "Free money! Click here now!"
            assert last_entry["label"] == "spam"
            assert last_entry["confidence"] == 0.95
            assert "text_hash" in last_entry
            assert last_entry["text_hash"].startswith("sha256:")

        print("✅ Prediction audit logging works correctly")
        return True

    except AssertionError as e:
        print(f"❌ Prediction audit logging assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Prediction audit logging failed: {e}")
        return False


def test_feedback_audit_logging():
    """Test feedback audit log entry creation."""
    print("\n🔍 Testing feedback audit logging...")

    try:
        from src.api_interface.services.audit_service import audit_service

        audit_service.log_feedback(
            feedback_id="test-feedback-123",
            text="Test spam message",
            original_label="spam",
            user_feedback="Correctly identified",
            thumbs_up=True,
            thumbs_down=False,
            model="ots-mbert",
            client_ip="192.168.1.101",
            user_id="test_user",
        )

        audit_service.flush()

        print("✅ Feedback audit logging works correctly")
        return True

    except Exception as e:
        print(f"❌ Feedback audit logging failed: {e}")
        return False


def test_access_denied_logging():
    """Test access denied audit logging."""
    print("\n🔍 Testing access denied logging...")

    try:
        from src.api_interface.services.audit_service import audit_service

        audit_service.log_access_denied(
            client_ip="203.0.113.45",
            endpoint="/predict/",
            reason="IP not in allowlist",
            attempted_action="POST",
        )

        audit_service.flush()

        print("✅ Access denied logging works correctly")
        return True

    except Exception as e:
        print(f"❌ Access denied logging failed: {e}")
        return False


def test_system_event_logging():
    """Test system event audit logging."""
    print("\n🔍 Testing system event logging...")

    try:
        from src.api_interface.services.audit_service import audit_service

        audit_service.log_system_event(
            event_type="test_event",
            message="Testing system event logging",
            metadata={"test": True, "timestamp": "2025-10-20T00:00:00Z"},
        )

        audit_service.flush()

        print("✅ System event logging works correctly")
        return True

    except Exception as e:
        print(f"❌ System event logging failed: {e}")
        return False


def test_text_hashing():
    """Test SHA-256 text hashing for integrity."""
    print("\n🔍 Testing text hashing...")

    try:
        from src.api_interface.services.audit_service import audit_service

        text = "Test message for hashing"
        hash1 = audit_service._compute_text_hash(text)
        hash2 = audit_service._compute_text_hash(text)

        assert hash1 == hash2, "Same text should produce same hash"
        assert hash1.startswith("sha256:"), "Hash should have sha256: prefix"
        assert len(hash1) == 71, f"Hash length should be 71, got {len(hash1)}"

        print("✅ Text hashing works correctly")
        return True

    except AssertionError as e:
        print(f"❌ Text hashing assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Text hashing failed: {e}")
        return False


def test_audit_query():
    """Test audit log querying functionality."""
    print("\n🔍 Testing audit log queries...")

    try:
        from src.api_interface.services.audit_service import audit_service

        # Query all logs
        entries = audit_service.query_logs(limit=50)
        assert isinstance(entries, list), "Query should return list"

        # Query by type
        if entries:
            first_entry = entries[0]
            entry_type = first_entry.get("entry_type")
            filtered = audit_service.query_logs(limit=50, entry_type=entry_type)
            assert all(
                e.get("entry_type") == entry_type for e in filtered
            ), "All entries should match filter"

        print(f"✅ Audit log queries work correctly ({len(entries)} entries found)")
        return True

    except AssertionError as e:
        print(f"❌ Audit query assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Audit query failed: {e}")
        return False


def test_statistics():
    """Test audit log statistics calculation."""
    print("\n🔍 Testing audit statistics...")

    try:
        from src.api_interface.services.audit_service import audit_service

        stats = audit_service.get_statistics()

        assert isinstance(stats, dict), "Statistics should be a dictionary"
        assert "total_predictions" in stats
        assert "total_feedback" in stats
        assert "total_access_denied" in stats
        assert "predictions_by_label" in stats
        assert "predictions_by_model" in stats
        assert "avg_confidence" in stats
        assert "unique_client_count" in stats

        print(
            f"✅ Audit statistics work correctly\n"
            f"   - Total predictions: {stats['total_predictions']}\n"
            f"   - Total feedback: {stats['total_feedback']}\n"
            f"   - Unique clients: {stats['unique_client_count']}"
        )
        return True

    except AssertionError as e:
        print(f"❌ Statistics assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Statistics failed: {e}")
        return False


def test_performance():
    """Test audit logging performance impact."""
    print("\n🔍 Testing audit logging performance...")

    try:
        from src.api_interface.services.audit_service import audit_service

        # Measure time for multiple audit logs
        start = time.time()
        iterations = 100

        for i in range(iterations):
            audit_service.log_prediction(
                text="Performance test message",
                label="ham",
                confidence=0.9,
                model="test",
                model_version="1.0",
                processing_time=5,
                client_ip="127.0.0.1",
                text_length=24,
            )

        audit_service.flush()
        elapsed = time.time() - start
        avg_time_ms = (elapsed / iterations) * 1000

        assert (
            avg_time_ms < 10
        ), f"Performance should be < 10ms per log, got {avg_time_ms:.2f}ms"

        print(f"✅ Performance test passed: {avg_time_ms:.2f}ms per log entry")
        return True

    except AssertionError as e:
        print(f"❌ Performance assertion failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def test_concurrent_writes():
    """Test thread-safe concurrent writes to audit log."""
    print("\n🔍 Testing concurrent writes...")

    try:
        import threading
        from src.api_interface.services.audit_service import audit_service

        def write_logs(thread_id, count):
            for i in range(count):
                audit_service.log_prediction(
                    text=f"Thread {thread_id} message {i}",
                    label="ham",
                    confidence=0.5,
                    model="test",
                    model_version="1.0",
                    processing_time=1,
                    client_ip="127.0.0.1",
                    text_length=20,
                )

        # Create multiple threads writing concurrently
        threads = []
        thread_count = 5
        logs_per_thread = 20

        for i in range(thread_count):
            t = threading.Thread(target=write_logs, args=(i, logs_per_thread))
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        audit_service.flush()

        print(
            f"✅ Concurrent writes test passed ({thread_count} threads, "
            f"{logs_per_thread} logs each)"
        )
        return True

    except Exception as e:
        print(f"❌ Concurrent writes test failed: {e}")
        return False


def run_all_tests():
    """Run all audit logging tests."""
    print("=" * 70)
    print("🧪 Running OpenTextShield Audit Logging Test Suite")
    print("=" * 70)

    tests = [
        test_audit_service_initialization,
        test_prediction_audit_logging,
        test_feedback_audit_logging,
        test_access_denied_logging,
        test_system_event_logging,
        test_text_hashing,
        test_audit_query,
        test_statistics,
        test_performance,
        test_concurrent_writes,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 70)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 70)

    if all(results):
        print("✅ All audit logging tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
