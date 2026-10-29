"""Atomic, thread-safe persistence for background training status."""

import json
import logging
import os
import threading
import time
import uuid

logger = logging.getLogger(__name__)

_STATUS_WRITE_LOCK = threading.Lock()
_REPLACE_RETRIES = 5


def write_training_status(status_path, payload):
    """Atomically write a training status, retrying transient Windows file locks."""
    directory = os.path.dirname(os.path.abspath(status_path))
    os.makedirs(directory, exist_ok=True)
    temp_path = f"{status_path}.{uuid.uuid4().hex}.tmp"

    with _STATUS_WRITE_LOCK:
        try:
            with open(temp_path, "w", encoding="utf-8") as status_file:
                json.dump(payload, status_file)

            for attempt in range(_REPLACE_RETRIES):
                try:
                    os.replace(temp_path, status_path)
                    return
                except PermissionError:
                    if attempt + 1 == _REPLACE_RETRIES:
                        logger.exception(
                            "Failed to replace training status file after %s attempts: %s",
                            _REPLACE_RETRIES,
                            status_path,
                        )
                        raise
                    time.sleep(0.02 * (2 ** attempt))
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
