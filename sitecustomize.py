"""Process-wide Python startup safeguards for the local Streamlit runtime."""

import asyncio
import sys


def configure_windows_event_loop() -> None:
    """Avoid noisy Proactor socket-close errors on Windows Streamlit runs."""
    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


configure_windows_event_loop()