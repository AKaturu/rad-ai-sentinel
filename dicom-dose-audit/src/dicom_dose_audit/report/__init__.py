"""HTML and PDF quality-improvement report generation."""

from __future__ import annotations

from .generate import ReportArtifacts, generate_dose_report, render_report_html

__all__ = [
    "ReportArtifacts",
    "generate_dose_report",
    "render_report_html",
]
