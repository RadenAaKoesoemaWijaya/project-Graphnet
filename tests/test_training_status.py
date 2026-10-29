import json
from concurrent.futures import ThreadPoolExecutor

import training_status


def test_training_status_writes_are_safe_for_concurrent_workers(tmp_path):
    status_path = tmp_path / "training.json"

    def write_status(index):
        training_status.write_training_status(
            status_path, {"status": "running", "progress": index / 32}
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_status, range(32)))

    with status_path.open(encoding="utf-8") as status_file:
        status = json.load(status_file)

    assert status["status"] == "running"
    assert 0 <= status["progress"] <= 1
    assert list(tmp_path.glob("*.tmp")) == []


def test_training_status_retries_transient_windows_permission_error(
    tmp_path, monkeypatch
):
    status_path = tmp_path / "training.json"
    replace = training_status.os.replace
    attempts = 0

    def fail_twice_then_replace(source, target):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient file lock")
        replace(source, target)

    monkeypatch.setattr(training_status.os, "replace", fail_twice_then_replace)

    training_status.write_training_status(status_path, {"status": "completed"})

    assert attempts == 3
    with status_path.open(encoding="utf-8") as status_file:
        assert json.load(status_file) == {"status": "completed"}
    assert list(tmp_path.glob("*.tmp")) == []
