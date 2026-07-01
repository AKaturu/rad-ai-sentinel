"""Lightweight dashboard accessibility and responsive-layout guardrails."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "src" / "rad_ai_sentinel" / "app" / "main.py"

REQUIRED_SNIPPETS = {
    "wide Streamlit layout": 'st.set_page_config(page_title="rad-ai-sentinel", layout="wide")',
    "labeled CSV uploader": 'st.sidebar.file_uploader("Monitoring CSV"',
    "page title": 'st.title("rad-ai-sentinel")',
    "download buttons": "download_button(",
    "dashboard tabs": "st.tabs(",
    "responsive dataframe width": 'width="stretch"',
}


def main() -> int:
    source = APP.read_text(encoding="utf-8")
    missing = [name for name, snippet in REQUIRED_SNIPPETS.items() if snippet not in source]
    if missing:
        for name in missing:
            print(f"Missing dashboard guardrail: {name}")
        return 1
    print("Dashboard accessibility/responsive guardrails passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
