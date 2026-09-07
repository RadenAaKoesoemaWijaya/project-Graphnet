import asyncio
import os
from pathlib import Path
import subprocess
import sys

import sitecustomize


def test_windows_event_loop_policy_is_selector_when_supported():
    sitecustomize.configure_windows_event_loop()

    if sys.platform == "win32":
        assert isinstance(
            asyncio.get_event_loop_policy(),
            asyncio.WindowsSelectorEventLoopPolicy,
        )


def test_sitecustomize_loads_in_streamlit_style_subprocess():
    if sys.platform != "win32":
        return

    project_root = str(Path(__file__).resolve().parents[1])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (project_root, environment.get("PYTHONPATH", "")) if path
    )
    result = subprocess.run(
        [sys.executable, "-c", "import asyncio; print(type(asyncio.get_event_loop_policy()).__name__)"],
        capture_output=True,
        text=True,
        env=environment,
        check=True,
    )

    assert "WindowsSelectorEventLoopPolicy" in result.stdout