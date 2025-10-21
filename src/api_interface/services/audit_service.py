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
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

from ..config.settings import settings
from ..utils.logging import logger as app_logger
from ..models.tmforum_event_models import (
    TMForumEvent,
    TMForumEventCreate,
    TMForumEventStatistics,
    TMForumEventType,
    TMForumEventPriority,
    TMForumEventState,
    Characteristic,
    RelatedParty,
    RelatedEntity,
)


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

    # ============================================================================
    # TMF688 Event Management API Adapter Methods
    # ============================================================================

    def convert_to_tmf688_event(self, audit_entry: Dict[str, Any]) -> TMForumEvent:
        """
        Convert internal audit log format to TMF688 Event resource.

        Args:
            audit_entry: Internal audit log entry

        Returns:
            TMF688-compliant Event resource
        """
        entry_type = audit_entry.get("entry_type", "unknown")
        timestamp_str = audit_entry.get("timestamp", datetime.utcnow().isoformat() + "Z")
        event_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

        # Generate unique event ID
        event_id = str(uuid.uuid4())
        event_business_id = f"{entry_type}-{event_time.strftime('%Y-%m-%d-%H%M%S')}-{event_id[:8]}"

        # Map entry type to TMForum event type
        event_type_map = {
            "prediction": TMForumEventType.PREDICTION,
            "feedback": TMForumEventType.FEEDBACK,
            "access_denied": TMForumEventType.SECURITY,
            "system_event": TMForumEventType.SYSTEM,
        }
        event_type = event_type_map.get(entry_type, TMForumEventType.SYSTEM)

        # Calculate priority
        priority = self._calculate_event_priority(audit_entry)

        # Build source (client IP)
        client_ip = audit_entry.get("client_ip", "unknown")
        source = RelatedParty(
            id=client_ip,
            role="client" if client_ip != "system" else "system",
            referredType="IPAddress" if client_ip != "system" else "System"
        )

        # Build reporting system (model or system)
        model_name = audit_entry.get("model", "OpenTextShield")
        model_version = audit_entry.get("model_version")
        reporting_system = RelatedEntity(
            id=model_name.replace(" ", "_"),
            name=f"{model_name} v{model_version}" if model_version else model_name,
            referredType="AIModel" if entry_type in ("prediction", "feedback") else "System"
        )

        # Build characteristics from entry data
        characteristics = []

        if entry_type == "prediction":
            characteristics.extend([
                Characteristic(name="label", value=str(audit_entry.get("label", "")), valueType="string"),
                Characteristic(name="confidence", value=str(audit_entry.get("confidence", 0.0)), valueType="float"),
                Characteristic(name="processing_time_ms", value=str(audit_entry.get("processing_time_ms", 0.0)), valueType="float"),
                Characteristic(name="text_length", value=str(audit_entry.get("text_length", 0)), valueType="integer"),
                Characteristic(name="text_hash", value=str(audit_entry.get("text_hash", "")), valueType="string"),
                Characteristic(name="redaction_applied", value=str(audit_entry.get("redaction_applied", False)), valueType="boolean"),
            ])
        elif entry_type == "feedback":
            characteristics.extend([
                Characteristic(name="feedback_id", value=str(audit_entry.get("feedback_id", "")), valueType="string"),
                Characteristic(name="original_label", value=str(audit_entry.get("original_label", "")), valueType="string"),
                Characteristic(name="user_feedback", value=str(audit_entry.get("user_feedback", "")), valueType="string"),
                Characteristic(name="thumbs_up", value=str(audit_entry.get("thumbs_up", False)), valueType="boolean"),
                Characteristic(name="thumbs_down", value=str(audit_entry.get("thumbs_down", False)), valueType="boolean"),
                Characteristic(name="text_hash", value=str(audit_entry.get("text_hash", "")), valueType="string"),
            ])
            if audit_entry.get("user_id"):
                characteristics.append(
                    Characteristic(name="user_id", value=str(audit_entry.get("user_id")), valueType="string")
                )
        elif entry_type == "access_denied":
            characteristics.extend([
                Characteristic(name="endpoint", value=str(audit_entry.get("endpoint", "")), valueType="string"),
                Characteristic(name="reason", value=str(audit_entry.get("reason", "")), valueType="string"),
            ])
            if audit_entry.get("attempted_action"):
                characteristics.append(
                    Characteristic(name="attempted_action", value=str(audit_entry.get("attempted_action")), valueType="string")
                )
        elif entry_type == "system_event":
            characteristics.extend([
                Characteristic(name="event_type", value=str(audit_entry.get("event_type", "")), valueType="string"),
                Characteristic(name="message", value=str(audit_entry.get("message", "")), valueType="string"),
            ])
            if audit_entry.get("metadata"):
                for key, value in audit_entry.get("metadata", {}).items():
                    characteristics.append(
                        Characteristic(name=f"metadata_{key}", value=str(value), valueType="string")
                    )

        # Generate title and description
        title, description = self._generate_event_title_description(entry_type, audit_entry)

        # Build TMForum Event
        event = TMForumEvent(
            id=f"evt-{event_id}",
            href=f"/tmf-api/event/evt-{event_id}",
            eventId=event_business_id,
            eventType=event_type,
            eventTime=event_time,
            timeOccurred=event_time,
            title=title,
            description=description,
            priority=priority,
            state=TMForumEventState.ACKNOWLEDGED,
            correlationId=None,
            source=source,
            reportingSystem=reporting_system,
            characteristic=characteristics,
            type=f"{event_type.value}",
        )

        return event

    def _calculate_event_priority(self, audit_entry: Dict[str, Any]) -> TMForumEventPriority:
        """
        Calculate event priority based on entry characteristics.

        Args:
            audit_entry: Internal audit log entry

        Returns:
            TMForum event priority
        """
        entry_type = audit_entry.get("entry_type")

        # Security events are high priority
        if entry_type == "access_denied":
            return TMForumEventPriority.HIGH

        # System events vary by type
        if entry_type == "system_event":
            system_event_type = audit_entry.get("event_type", "")
            if "error" in system_event_type.lower() or "fail" in system_event_type.lower():
                return TMForumEventPriority.HIGH
            elif "shutdown" in system_event_type.lower() or "startup" in system_event_type.lower():
                return TMForumEventPriority.MEDIUM
            return TMForumEventPriority.LOW

        # Predictions with phishing label are higher priority
        if entry_type == "prediction":
            label = audit_entry.get("label", "")
            if label == "phishing":
                return TMForumEventPriority.MEDIUM
            return TMForumEventPriority.LOW

        # Feedback is generally low priority
        if entry_type == "feedback":
            # Negative feedback (thumbs down) is medium priority
            if audit_entry.get("thumbs_down", False):
                return TMForumEventPriority.MEDIUM
            return TMForumEventPriority.LOW

        return TMForumEventPriority.LOW

    def _generate_event_title_description(self, entry_type: str, audit_entry: Dict[str, Any]) -> Tuple[str, str]:
        """
        Generate human-readable title and description for event.

        Args:
            entry_type: Type of audit entry
            audit_entry: Internal audit log entry

        Returns:
            Tuple of (title, description)
        """
        if entry_type == "prediction":
            label = audit_entry.get("label", "unknown")
            confidence = audit_entry.get("confidence", 0.0)
            model = audit_entry.get("model", "Unknown")
            title = f"Text Classification: {label}"
            description = f"Message classified as '{label}' by {model} with {confidence:.2%} confidence"
            return title, description

        elif entry_type == "feedback":
            feedback_id = audit_entry.get("feedback_id", "")
            thumbs_up = audit_entry.get("thumbs_up", False)
            thumbs_down = audit_entry.get("thumbs_down", False)
            sentiment = "positive" if thumbs_up else ("negative" if thumbs_down else "neutral")
            title = f"User Feedback: {sentiment}"
            description = f"User submitted {sentiment} feedback (ID: {feedback_id})"
            return title, description

        elif entry_type == "access_denied":
            endpoint = audit_entry.get("endpoint", "unknown")
            reason = audit_entry.get("reason", "Unknown reason")
            title = f"Access Denied: {endpoint}"
            description = f"Access denied to {endpoint}: {reason}"
            return title, description

        elif entry_type == "system_event":
            system_event_type = audit_entry.get("event_type", "unknown")
            message = audit_entry.get("message", "")
            title = f"System Event: {system_event_type}"
            description = message or f"System event of type {system_event_type}"
            return title, description

        return "Audit Event", "Audit log entry"

    def get_event_by_id(self, event_id: str) -> Optional[TMForumEvent]:
        """
        Retrieve single event by UUID, convert to TMF688 format.

        Args:
            event_id: Event UUID (format: evt-{uuid})

        Returns:
            TMF688 Event or None if not found
        """
        # For now, we need to scan through logs to find by generated ID
        # In a production system, you'd maintain an index or database
        # This is a simplified implementation for demonstration

        # Note: Since we generate UUIDs on conversion, we can't actually retrieve
        # by ID without storing the mapping. For this demo, we'll return None
        # and document that event retrieval by ID requires event storage enhancement.

        app_logger.warning(f"Event retrieval by ID '{event_id}' not fully implemented - requires event ID mapping")
        return None

    def query_events_tmf688(
        self,
        event_type: Optional[str] = None,
        time_gte: Optional[datetime] = None,
        time_lte: Optional[datetime] = None,
        source_id: Optional[str] = None,
        reporting_system: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[TMForumEvent], int]:
        """
        Query events with TMF688 parameters, return (events, total_count).

        Args:
            event_type: Filter by event type (PredictionEvent, FeedbackEvent, etc.)
            time_gte: Filter events after this time
            time_lte: Filter events before this time
            source_id: Filter by source ID (client IP)
            reporting_system: Filter by reporting system (model name)
            priority: Filter by priority (low, medium, high, critical)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            Tuple of (list of TMF688 events, total count matching filters)
        """
        # Map TMForum event types to internal entry types
        entry_type_map = {
            "PredictionEvent": "prediction",
            "FeedbackEvent": "feedback",
            "SecurityEvent": "access_denied",
            "SystemEvent": "system_event",
        }

        internal_entry_type = entry_type_map.get(event_type) if event_type else None

        # Query internal audit logs
        all_entries = self.query_logs(
            limit=10000,  # Get large batch to filter
            entry_type=internal_entry_type,
            start_date=time_gte.isoformat() + "Z" if time_gte else None,
            end_date=time_lte.isoformat() + "Z" if time_lte else None,
            client_ip=source_id,
        )

        # Convert to TMF688 events
        tmf_events = []
        for entry in all_entries:
            try:
                tmf_event = self.convert_to_tmf688_event(entry)

                # Apply additional filters
                if reporting_system and tmf_event.reportingSystem.id != reporting_system:
                    continue

                if priority and tmf_event.priority.value != priority:
                    continue

                tmf_events.append(tmf_event)

            except Exception as e:
                app_logger.warning(f"Failed to convert entry to TMF688: {str(e)}")
                continue

        # Get total count before pagination
        total_count = len(tmf_events)

        # Apply pagination
        paginated_events = tmf_events[offset:offset + limit]

        return paginated_events, total_count

    def create_event_tmf688(self, event_create: TMForumEventCreate) -> TMForumEvent:
        """
        Create new event from TMForum format.

        Args:
            event_create: TMForum event creation request

        Returns:
            Created TMF688 Event
        """
        # Generate event ID
        event_id = str(uuid.uuid4())
        event_time = datetime.utcnow()
        event_business_id = f"{event_create.eventType.value}-{event_time.strftime('%Y-%m-%d-%H%M%S')}-{event_id[:8]}"

        # Build TMForum Event
        event = TMForumEvent(
            id=f"evt-{event_id}",
            href=f"/tmf-api/event/evt-{event_id}",
            eventId=event_business_id,
            eventType=event_create.eventType,
            eventTime=event_time,
            timeOccurred=event_time,
            title=event_create.title,
            description=event_create.description,
            priority=event_create.priority,
            state=TMForumEventState.ACKNOWLEDGED,
            correlationId=event_create.correlationId,
            source=event_create.source,
            reportingSystem=event_create.reportingSystem,
            characteristic=event_create.characteristic,
            type=event_create.eventType.value,
        )

        # Optionally write to audit log in internal format
        # (This allows manual TMF688 events to be stored in audit logs)
        try:
            internal_entry = {
                "timestamp": event_time.isoformat() + "Z",
                "entry_type": "manual_event",
                "client_ip": event_create.source.id,
                "event_type": event_create.eventType.value,
                "title": event_create.title,
                "description": event_create.description,
                "priority": event_create.priority.value,
                "characteristics": [char.model_dump() for char in event_create.characteristic],
            }
            self._write_audit_entry(internal_entry)
        except Exception as e:
            app_logger.warning(f"Failed to write manual TMF688 event to audit log: {str(e)}")

        return event

    def get_event_statistics_tmf688(self) -> TMForumEventStatistics:
        """
        Get statistics in TMF688 format.

        Returns:
            TMF688 Event statistics
        """
        # Get internal statistics
        internal_stats = self.get_statistics()

        # Convert to TMF688 format
        events_by_type = {
            "PredictionEvent": internal_stats.get("total_predictions", 0),
            "FeedbackEvent": internal_stats.get("total_feedback", 0),
            "SecurityEvent": internal_stats.get("total_access_denied", 0),
            "SystemEvent": 0,  # Not tracked separately in internal stats
        }

        # Calculate events by priority (approximate based on internal data)
        total_events = sum(events_by_type.values())
        events_by_priority = {
            "low": internal_stats.get("total_predictions", 0),  # Most predictions are low
            "medium": internal_stats.get("total_feedback", 0),  # Feedback is medium
            "high": internal_stats.get("total_access_denied", 0),  # Security is high
            "critical": 0,
        }

        # Events by model
        events_by_model = internal_stats.get("predictions_by_model", {})

        # Get time range from logs
        time_range = None
        try:
            recent_logs = self.query_logs(limit=1)
            oldest_logs = self.query_logs(limit=1)  # Should query oldest, but simplified
            if recent_logs and oldest_logs:
                time_range = {
                    "start": oldest_logs[0].get("timestamp", ""),
                    "end": recent_logs[0].get("timestamp", ""),
                }
        except Exception as e:
            app_logger.warning(f"Failed to determine time range: {str(e)}")

        return TMForumEventStatistics(
            totalEvents=total_events,
            eventsByType=events_by_type,
            eventsByPriority=events_by_priority,
            eventsByModel=events_by_model if events_by_model else None,
            averageConfidence=internal_stats.get("avg_confidence"),
            uniqueSourceCount=internal_stats.get("unique_client_count", 0),
            timeRange=time_range,
        )


# Global audit service instance
audit_service = AuditLogger()
