"""
Audit logging service for OpenTextShield API.

Provides comprehensive audit logging for classification decisions, feedback,
access denials, and system events. Designed for compliance and ML model improvement.
"""

import json
import hashlib
import threading
import re
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

from ..config.settings import settings
from ..utils.logging import logger as app_logger


class AuditLogger:
    """Comprehensive audit logging service with tamper detection and rotation."""

    def __init__(self):
        """Initialize audit logger."""
        self.audit_dir = settings.audit_dir
        self.max_file_size_bytes = settings.audit_max_file_size_mb * 1024 * 1024
        self.current_log_file = None
        self.lock = threading.RLock()
        self.entry_buffer = []
        self.buffer_size = 10  # Flush after N entries

        # Rotation configuration
        self.rotation_enabled = settings.audit_rotation_enabled
        self.rotation_strategy = settings.audit_rotation_strategy
        self.rotate_on_date_change = settings.audit_rotation_on_date_change
        self.last_rotation_date = None

        # Retention configuration
        self.retention_enabled = settings.audit_retention_enabled
        self.retention_days = settings.audit_retention_days
        self.retention_check_interval = settings.audit_retention_check_interval_hours * 3600
        self.last_retention_check = time.time()
        self.archive_enabled = settings.audit_archive_enabled
        self.archive_dir = settings.audit_archive_dir or (settings.audit_dir / "archive")

        self._initialize_log_file()
        self._cleanup_old_logs()

    def _initialize_log_file(self) -> None:
        """Initialize or get current log file."""
        # Ensure audit directory exists
        self.audit_dir.mkdir(parents=True, exist_ok=True)

        # Get or create today's log file
        today = datetime.now().strftime("%Y-%m-%d")
        self.current_log_file = self.audit_dir / f"audit_{today}.jsonl"

    def log_prediction(
        self,
        text: str,
        label: str,
        confidence: float,
        model: str,
        model_version: str,
        processing_time: float,
        client_ip: str,
        text_length: int,
    ) -> None:
        """
        Log a classification prediction.

        Args:
            text: Full text that was classified
            label: Classification label (ham, spam, phishing)
            confidence: Confidence score (0.0-1.0)
            model: Model name used for prediction
            model_version: Model version
            processing_time: Processing time in milliseconds
            client_ip: Client IP address
            text_length: Length of original text
        """
        if not settings.audit_enabled:
            return

        try:
            text_preview, redaction_applied, redacted_entities = self._apply_text_storage(text)
            text_hash = self._compute_text_hash(text)

            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "entry_type": "prediction",
                "client_ip": client_ip,
                "text": text_preview,
                "text_length": text_length,
                "text_hash": text_hash,
                "label": label,
                "confidence": round(confidence, 4),
                "model": model,
                "model_version": model_version,
                "processing_time_ms": round(processing_time, 2),
                "redaction_applied": redaction_applied,
            }

            if redacted_entities:
                entry["redacted_entities"] = redacted_entities

            self._write_audit_entry(entry)

        except Exception as e:
            app_logger.error(f"Failed to log prediction to audit: {str(e)}")

    def log_feedback(
        self,
        feedback_id: str,
        text: str,
        original_label: str,
        user_feedback: str,
        thumbs_up: bool,
        thumbs_down: bool,
        model: str,
        client_ip: str,
        user_id: Optional[str] = None,
    ) -> None:
        """
        Log user feedback submission.

        Args:
            feedback_id: Unique feedback identifier
            text: Original text that was classified
            original_label: Original classification label
            user_feedback: User's feedback text
            thumbs_up: Whether user agreed with classification
            thumbs_down: Whether user disagreed with classification
            model: Model that was used
            client_ip: Client IP address
            user_id: Optional user identifier
        """
        if not settings.audit_enabled:
            return

        try:
            text_preview, redaction_applied, redacted_entities = self._apply_text_storage(text)
            text_hash = self._compute_text_hash(text)

            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "entry_type": "feedback",
                "client_ip": client_ip,
                "feedback_id": feedback_id,
                "text": text_preview,
                "text_hash": text_hash,
                "original_label": original_label,
                "user_feedback": user_feedback,
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "model": model,
                "redaction_applied": redaction_applied,
            }

            if user_id:
                entry["user_id"] = user_id

            if redacted_entities:
                entry["redacted_entities"] = redacted_entities

            self._write_audit_entry(entry)

        except Exception as e:
            app_logger.error(f"Failed to log feedback to audit: {str(e)}")

    def log_access_denied(
        self, client_ip: str, endpoint: str, reason: str, attempted_action: Optional[str] = None
    ) -> None:
        """
        Log access denied event (security).

        Args:
            client_ip: Client IP address
            endpoint: Endpoint that was accessed
            reason: Reason for denial
            attempted_action: HTTP method or action attempted
        """
        if not settings.audit_enabled:
            return

        try:
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "entry_type": "access_denied",
                "client_ip": client_ip,
                "endpoint": endpoint,
                "reason": reason,
            }

            if attempted_action:
                entry["attempted_action"] = attempted_action

            self._write_audit_entry(entry)

        except Exception as e:
            app_logger.error(f"Failed to log access denied to audit: {str(e)}")

    def log_system_event(
        self,
        event_type: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log system event (startup, shutdown, errors, etc.).

        Args:
            event_type: Type of system event (startup, shutdown, model_load, etc.)
            message: Event message
            metadata: Optional metadata dictionary
        """
        if not settings.audit_enabled:
            return

        try:
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "entry_type": "system_event",
                "client_ip": "system",
                "event_type": event_type,
                "message": message,
            }

            if metadata:
                entry["metadata"] = metadata

            self._write_audit_entry(entry)

        except Exception as e:
            app_logger.error(f"Failed to log system event to audit: {str(e)}")

    def _apply_text_storage(self, text: str) -> Tuple[str, bool, Optional[List[str]]]:
        """
        Apply text storage policy based on configuration.

        Returns:
            Tuple of (processed_text, redaction_applied, redacted_entities)
        """
        storage_mode = settings.audit_text_storage

        if storage_mode == "full":
            return text, False, None

        elif storage_mode == "truncated":
            truncated = text[: settings.audit_truncate_length]
            if len(text) > settings.audit_truncate_length:
                truncated += "..."
            return truncated, False, None

        elif storage_mode == "hash_only":
            return self._compute_text_hash(text), False, None

        elif storage_mode == "redacted":
            if not settings.audit_redact_patterns:
                return text, False, None

            redacted_text = text
            redacted_entities = []

            for pattern_name in settings.audit_redact_patterns:
                pattern = self._get_redaction_pattern(pattern_name)
                if pattern:
                    if re.search(pattern, redacted_text):
                        redacted_text = re.sub(pattern, "[REDACTED]", redacted_text)
                        redacted_entities.append(pattern_name)

            return redacted_text, len(redacted_entities) > 0, redacted_entities if redacted_entities else None

        else:
            return text, False, None

    def _get_redaction_pattern(self, pattern_name: str) -> Optional[str]:
        """Get regex pattern for common PII types."""
        patterns = {
            "phone_numbers": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\+\d{1,3}\s?\d{1,14}",
            "email_addresses": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "credit_cards": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "urls": r"https?://[^\s]+",
        }
        return patterns.get(pattern_name)

    def _compute_text_hash(self, text: str) -> str:
        """
        Compute SHA-256 hash of text for integrity verification.

        Returns:
            Hash in format "sha256:hex_hash"
        """
        hash_obj = hashlib.sha256(text.encode("utf-8"))
        return f"sha256:{hash_obj.hexdigest()}"

    def _write_audit_entry(self, entry: Dict[str, Any]) -> None:
        """
        Write audit entry to JSON Lines file with thread safety.

        Args:
            entry: Audit entry dictionary to write
        """
        with self.lock:
            # Check if we need to rotate the log file
            self._check_log_rotation()

            # Buffer the entry
            self.entry_buffer.append(entry)

            # Flush buffer if it reaches threshold
            if len(self.entry_buffer) >= self.buffer_size:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Write buffered entries to file."""
        if not self.entry_buffer or not self.current_log_file:
            return

        try:
            # Open in append mode and write entries
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                for entry in self.entry_buffer:
                    f.write(json.dumps(entry, default=str) + "\n")

            self.entry_buffer = []

        except Exception as e:
            app_logger.error(f"Failed to flush audit buffer: {str(e)}")

    def _check_log_rotation(self) -> None:
        """Check if log file needs rotation based on configured strategy."""
        if not self.rotation_enabled or not self.current_log_file:
            return

        today = datetime.now().strftime("%Y-%m-%d")
        should_rotate = False
        reason = ""

        # Check date-based rotation
        if self.rotation_strategy in ("date_only", "size_or_date"):
            if self.rotate_on_date_change:
                if not self.current_log_file.name.endswith(f"_{today}.jsonl"):
                    should_rotate = True
                    reason = "date changed"

        # Check size-based rotation
        if self.rotation_strategy in ("size_only", "size_or_date"):
            if self.current_log_file.exists():
                file_size = self.current_log_file.stat().st_size
                if file_size > self.max_file_size_bytes:
                    should_rotate = True
                    reason = f"size exceeded ({file_size / 1024 / 1024:.1f}MB)"

        if should_rotate:
            self._rotate_log_file(reason)

        # Periodically check retention policy
        if self.retention_enabled and (time.time() - self.last_retention_check) > self.retention_check_interval:
            self._cleanup_old_logs()
            self.last_retention_check = time.time()

    def _rotate_log_file(self, reason: str = "") -> None:
        """
        Rotate the current log file.

        Args:
            reason: Reason for rotation (for logging)
        """
        if not self.current_log_file or not self.current_log_file.exists():
            self._initialize_log_file()
            return

        try:
            today = datetime.now().strftime("%Y-%m-%d")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Rename current file
            backup_file = self.current_log_file.parent / f"audit_{today}_{timestamp}.jsonl"
            self.current_log_file.rename(backup_file)

            app_logger.info(f"Rotated audit log: {self.current_log_file.name} → {backup_file.name} ({reason})")

            # Create new log file
            self._initialize_log_file()

        except Exception as e:
            app_logger.error(f"Failed to rotate log file: {str(e)}")

    def _cleanup_old_logs(self) -> None:
        """
        Clean up old audit logs based on retention policy.

        Supports:
        - Deletion of logs older than retention_days
        - Archiving of old logs instead of deletion
        """
        if not self.retention_enabled or self.retention_days <= 0:
            return

        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            cutoff_timestamp = cutoff_date.timestamp()

            # Get all audit log files
            log_files = list(self.audit_dir.glob("audit_*.jsonl"))

            deleted_count = 0
            archived_count = 0

            for log_file in log_files:
                # Skip current log file
                if log_file == self.current_log_file:
                    continue

                # Check file modification time
                file_mtime = log_file.stat().st_mtime
                if file_mtime < cutoff_timestamp:
                    if self.archive_enabled:
                        # Archive the file
                        self.archive_dir.mkdir(parents=True, exist_ok=True)
                        archive_path = self.archive_dir / log_file.name
                        shutil.move(str(log_file), str(archive_path))
                        archived_count += 1
                        app_logger.info(f"Archived old audit log: {log_file.name} → {archive_path.name}")
                    else:
                        # Delete the file
                        log_file.unlink()
                        deleted_count += 1
                        app_logger.info(f"Deleted old audit log: {log_file.name}")

            if deleted_count > 0 or archived_count > 0:
                app_logger.info(
                    f"Retention cleanup: deleted {deleted_count} logs, archived {archived_count} logs "
                    f"(older than {self.retention_days} days)"
                )

        except Exception as e:
            app_logger.error(f"Failed to cleanup old audit logs: {str(e)}")

    def flush(self) -> None:
        """Flush all buffered entries to file (call on shutdown)."""
        with self.lock:
            self._flush_buffer()

    def query_logs(
        self,
        limit: int = 100,
        entry_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query audit logs with optional filters.

        Args:
            limit: Maximum number of entries to return
            entry_type: Filter by entry type (prediction, feedback, etc.)
            start_date: Filter by start date (ISO 8601)
            end_date: Filter by end date (ISO 8601)
            client_ip: Filter by client IP

        Returns:
            List of matching audit entries
        """
        entries = []

        # Get all audit log files
        log_files = sorted(self.audit_dir.glob("audit_*.jsonl"), reverse=True)

        for log_file in log_files:
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)

                            # Apply filters
                            if entry_type and entry.get("entry_type") != entry_type:
                                continue
                            if client_ip and entry.get("client_ip") != client_ip:
                                continue
                            if start_date:
                                entry_date = entry.get("timestamp", "")
                                if entry_date < start_date:
                                    continue
                            if end_date:
                                entry_date = entry.get("timestamp", "")
                                if entry_date > end_date:
                                    continue

                            entries.append(entry)

                            # Check if we've reached limit
                            if len(entries) >= limit:
                                return entries

                        except json.JSONDecodeError:
                            app_logger.warning(f"Failed to parse JSON line in {log_file}")
                            continue

            except Exception as e:
                app_logger.error(f"Failed to query log file {log_file}: {str(e)}")
                continue

        return entries

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics from audit logs.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "total_predictions": 0,
            "total_feedback": 0,
            "total_access_denied": 0,
            "predictions_by_label": {},
            "predictions_by_model": {},
            "avg_confidence": 0.0,
            "unique_clients": set(),
        }

        try:
            entries = self.query_logs(limit=100000)

            confidences = []

            for entry in entries:
                entry_type = entry.get("entry_type")

                if entry_type == "prediction":
                    stats["total_predictions"] += 1
                    label = entry.get("label", "unknown")
                    stats["predictions_by_label"][label] = (
                        stats["predictions_by_label"].get(label, 0) + 1
                    )
                    model = entry.get("model", "unknown")
                    stats["predictions_by_model"][model] = (
                        stats["predictions_by_model"].get(model, 0) + 1
                    )
                    confidence = entry.get("confidence")
                    if confidence is not None:
                        confidences.append(confidence)

                elif entry_type == "feedback":
                    stats["total_feedback"] += 1

                elif entry_type == "access_denied":
                    stats["total_access_denied"] += 1

                # Track unique clients
                client_ip = entry.get("client_ip")
                if client_ip and client_ip != "system":
                    stats["unique_clients"].add(client_ip)

            # Calculate average confidence
            if confidences:
                stats["avg_confidence"] = round(sum(confidences) / len(confidences), 4)

            # Convert set to list for JSON serialization
            stats["unique_client_count"] = len(stats["unique_clients"])
            del stats["unique_clients"]

            return stats

        except Exception as e:
            app_logger.error(f"Failed to calculate statistics: {str(e)}")
            return stats


# Global audit service instance
audit_service = AuditLogger()
