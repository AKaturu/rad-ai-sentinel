"""HTML + PDF report generation from a single Jinja2 template."""

from __future__ import annotations

from .generate import ReportArtifacts, generate_monitoring_report, render_report_html

__all__ = ["ReportArtifacts", "generate_monitoring_report", "render_report_html"]
