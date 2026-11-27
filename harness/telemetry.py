"""Privacy-safe JSONL loggers separating operational runtime metrics from security audit events."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets
import sys
import threading
from typing import Any
from harness.config import (
    AUDIT_LOG_PATH,
    RUNTIME_LOG_PATH,
    TELEMETRY_ENABLED,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_log_path(path_value: str | Path) -> Path:
    """Resolve relative log paths against the project root; preserve absolute paths."""
    path = Path(path_value)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def query_fingerprint(raw_query: str) -> str:
    """Return a short deterministic SHA-256 prefix for correlating logs without storing raw text."""
    normalized = raw_query.strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:12]


class TelemetryLogger:
    """Thread-safe, non-fatal JSONL writer for runtime.jsonl and audit.jsonl."""

    def __init__(
        self,
        runtime_path: str | Path = RUNTIME_LOG_PATH,
        audit_path: str | Path = AUDIT_LOG_PATH,
        enabled: bool = TELEMETRY_ENABLED,
    ) -> None:
        self.enabled = enabled
        self.runtime_path = _resolve_log_path(runtime_path)
        self.audit_path = _resolve_log_path(audit_path)
        self.session_id = secrets.token_hex(4)
        self._turn_counter = 0
        self._lock = threading.Lock()

    def next_turn_id(self) -> int:
        """Allocate a monotonically increasing turn ID scoped to this session."""
        with self._lock:
            self._turn_counter += 1
            return self._turn_counter

    def _append_jsonl(self, path: Path, record: dict[str, Any]) -> None:
        """Append a single JSON object line; never crash the agent on filesystem errors."""
        if not self.enabled:
            return
        try:
            line = json.dumps(record, ensure_ascii=False)
            with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError as exc:
            sys.stderr.write(f"[Telemetry Warning] Failed writing {path.name}: {exc}\n")

    def log_runtime(
        self,
        *,
        turn_id: int,
        query_sha256: str,
        query_chars: int,
        result: dict[str, Any],
    ) -> None:
        """Record one privacy-safe operational metrics entry in logs/runtime.jsonl."""
        record = {
            "ts": _utc_now_iso(),
            "session_id": self.session_id,
            "turn_id": turn_id,
            "query_sha256": query_sha256,
            "query_chars": query_chars,
            "rewritten": bool(result.get("rewritten_query")),
            "status": result.get("status"),
            "latency_ms": result.get("latency_ms"),
            "top_score": result.get("top_score"),
            "tokens": result.get("usage"),
            "llm": result.get("llm_meta"),
            "web": result.get("web_meta"),
            "memory_turns": result.get("memory_turns"),
            "memory_max": result.get("memory_max"),
        }
        self._append_jsonl(self.runtime_path, record)

    def log_audit(
        self,
        event: str,
        *,
        turn_id: int | None = None,
        query_sha256: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record a security, governance, or failure event in logs/audit.jsonl."""
        record: dict[str, Any] = {
            "ts": _utc_now_iso(),
            "session_id": self.session_id,
            "turn_id": turn_id,
            "event": event,
            "query_sha256": query_sha256,
        }
        if details:
            record["details"] = details
        self._append_jsonl(self.audit_path, record)