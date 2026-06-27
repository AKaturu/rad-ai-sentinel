"""Build the rad-ai-sentinel desktop application with PyInstaller."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "native"
BUILD = ROOT / "build" / "pyinstaller"
PACKAGE = "rad_ai_sentinel"


def _add_data_arg(source: Path, destination: str) -> str:
    return f"{source}{os.pathsep}{destination}"


def build_desktop(*, clean: bool = True) -> Path:
    if clean:
        shutil.rmtree(DIST, ignore_errors=True)
        shutil.rmtree(BUILD, ignore_errors=True)

    entry = ROOT / "packaging" / "entry_desktop.py"
    app_file = ROOT / "src" / PACKAGE / "app" / "main.py"
    args = [
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        "RadAISentinel",
        "--distpath",
        str(DIST),
        "--workpath",
        str(BUILD / "work"),
        "--specpath",
        str(BUILD / "spec"),
        "--add-data",
        _add_data_arg(app_file, f"{PACKAGE}/app"),
        "--collect-data",
        PACKAGE,
        "--collect-data",
        "streamlit",
        "--collect-submodules",
        "streamlit",
        "--collect-all",
        "plotly",
        "--hidden-import",
        "streamlit.web.cli",
        str(entry),
    ]
    PyInstaller.__main__.run(args)
    return DIST / "RadAISentinel"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-clean", action="store_true", help="Keep previous build folders.")
    args = parser.parse_args()
    output = build_desktop(clean=not args.no_clean)
    print(f"Built desktop app at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
