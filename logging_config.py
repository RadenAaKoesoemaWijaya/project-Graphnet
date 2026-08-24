"""Centralised logging configuration for ASTINA.

By default the root logger emits a single line of human-readable text per
record. On Cloud Run (and any aggregator that understands JSON), set the
environment variable ``ASTINA_LOG_FORMAT=json`` to switch the format to
Google's *structured-log* JSON shape so that Cloud Logging can parse the
``severity``, ``message``, ``time``, and arbitrary extra fields directly.

Usage
-----
    from logging_config import configure_logging
    configure_logging()        # safe to call multiple times

The configuration only touches handlers attached to the named logger
``graphnet``; other libraries (streamlit, torch, ...) keep their own
formatters so we don't fight them.
"""
import json
import logging
import os
import sys
import time
from typing import Any, Dict


# Cloud Logging's recognised severity labels. We translate Python's
# ``logging`` levels into these so the JSON output works out-of-the-box.
_SEVERITY_MAP = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}


class _JsonFormatter(logging.Formatter):
    """JSON formatter compatible with Google Cloud Logging.

    Reference: https://cloud.google.com/logging/docs/structured-logging
    """

    # Standard LogRecord attributes we never want to echo back. We use a
    # set for O(1) lookups in the hot path.
    _LOGRECORD_ATTRS = frozenset({
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    })

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "severity": _SEVERITY_MAP.get(record.levelno, "INFO"),
            "message": record.getMessage(),
            "time": time.strftime(
                "%Y-%m-%dT%H:%M:%S%z",
                time.gmtime(record.created),
            ),
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        # Pass-through any custom keys (``logger.info("…", extra={...})``)
        for key, value in record.__dict__.items():
            if key in self._LOGRECORD_ATTRS or key in payload:
                continue
            if key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _build_handler() -> logging.Handler:
    """Pick JSON or plain formatter based on the environment variable."""
    fmt = os.environ.get("ASTINA_LOG_FORMAT", "plain").lower().strip()
    if fmt == "json":
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(_JsonFormatter())
    else:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    return handler


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotently attach a single handler to the ``graphnet`` logger.

    Replaces any handler we previously added so the formatter change
    (plain ↔ JSON) takes effect on a re-run inside the same process.
    """
    logger = logging.getLogger("graphnet")
    logger.setLevel(level)
    # Remove handlers we own (those that came from this module — they all
    # carry the ``_astina_owned`` attribute set below). Other libraries'
    # handlers are untouched.
    for h in list(logger.handlers):
        if getattr(h, "_astina_owned", False):
            logger.removeHandler(h)
    handler = _build_handler()
    handler._astina_owned = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.propagate = False
