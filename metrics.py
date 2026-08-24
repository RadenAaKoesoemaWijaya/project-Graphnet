"""Lightweight observability helpers for ASTINA.

Provides:
* ``record_event(name, **fields)``  — emit a structured log line that
  Cloud Logging can turn into a log-based metric.
* ``metrics_snapshot()``            — return a JSON string with the
  most recent values of common counters/timers. Suitable for exposing
  through a Streamlit endpoint or a sidecar.

The module is intentionally framework-agnostic and safe to import from
any page or background thread.
"""
import json
import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict

logger = logging.getLogger("graphnet.metrics")

# Lightweight in-process metrics. No third-party dependency on
# prometheus_client because Cloud Run's metric model is log-based.
_lock = threading.Lock()
_state: Dict[str, Any] = {
    "started_at": time.time(),
    "counters": {},
    "last_values": {},
    "events": deque(maxlen=200),  # ring buffer of the most recent events
}


def _now() -> float:
    return time.time()


def record_event(name: str, **fields: Any) -> None:
    """Emit a structured event.

    The event is written as a single log line with the prefix
    ``astina.event`` so a log-based metric can match it (e.g. in Cloud
    Logging, ``jsonPayload.message=~"^astina.event "``).
    """
    payload = {"name": name, "ts": _now(), **fields}
    with _lock:
        _state["counters"][name] = _state["counters"].get(name, 0) + 1
        _state["last_values"][name] = fields
        _state["events"].appendleft(payload)
    # The leading literal is important: it lets log-based-metric filters
    # find these records without parsing the entire JSON.
    logger.info("astina.event " + json.dumps(payload, default=str))


def set_gauge(name: str, value: float) -> None:
    with _lock:
        _state["last_values"][name] = value


def metrics_snapshot() -> str:
    """Return a JSON snapshot for /metrics-style endpoints."""
    with _lock:
        return json.dumps(
            {
                "uptime_s": _now() - _state["started_at"],
                "counters": dict(_state["counters"]),
                "last_values": dict(_state["last_values"]),
                "events": list(_state["events"])[:20],
            },
            default=str,
        )
