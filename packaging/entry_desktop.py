"""PyInstaller desktop launcher for rad-ai-sentinel."""

from __future__ import annotations

import os
import sys
import threading
import webbrowser
from pathlib import Path


def _app_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "rad_ai_sentinel" / "app" / "main.py"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1] / "src" / "rad_ai_sentinel" / "app" / "main.py"


def main() -> int:
    app_path = _app_path()
    if "--self-check" in sys.argv:
        import rad_ai_sentinel  # noqa: F401

        if not app_path.exists():
            print(f"Missing dashboard app: {app_path}", file=sys.stderr)
            return 1
        print("rad-ai-sentinel desktop launcher OK")
        return 0

    from streamlit.web import cli as stcli

    port = os.environ.get("RAD_AI_SENTINEL_PORT", "8501")
    url = f"http://localhost:{port}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        "localhost",
        "--server.port",
        port,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    return int(stcli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
