"""Export payload helpers for dashboard and CLI surfaces."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .analysis import (
    MonitoringAnalysis,
    alerts_frame,
    analysis_summary_dict,
    drift_frame,
    missing_data_frame,
    site_calibration_frame,
    stratified_metrics_frame,
    summary_metrics_frame,
    version_comparison_frame,
)
from .report import render_report_html


@dataclass(frozen=True)
class ExportPayload:
    """In-memory downloadable artifact."""

    file_name: str
    mime: str
    data: str | bytes


def build_monitoring_export_payloads(analysis: MonitoringAnalysis) -> dict[str, ExportPayload]:
    """Build CSV, JSON, and HTML exports from one reviewed analysis object."""
    return {
        "validated_csv": ExportPayload(
            "validated_monitoring_data.csv",
            "text/csv",
            analysis.dataframe.to_csv(index=False),
        ),
        "summary_csv": ExportPayload(
            "summary_metrics.csv",
            "text/csv",
            summary_metrics_frame(analysis).to_csv(index=False),
        ),
        "stratified_csv": ExportPayload(
            "stratified_metrics.csv",
            "text/csv",
            stratified_metrics_frame(analysis).to_csv(index=False),
        ),
        "missing_csv": ExportPayload(
            "missing_data.csv",
            "text/csv",
            missing_data_frame(analysis).to_csv(index=False),
        ),
        "alerts_csv": ExportPayload(
            "alerts.csv",
            "text/csv",
            alerts_frame(analysis).to_csv(index=False),
        ),
        "drift_csv": ExportPayload(
            "drift.csv",
            "text/csv",
            drift_frame(analysis).to_csv(index=False),
        ),
        "site_calibration_csv": ExportPayload(
            "site_calibration_drift.csv",
            "text/csv",
            site_calibration_frame(analysis).to_csv(index=False),
        ),
        "versions_csv": ExportPayload(
            "model_versions.csv",
            "text/csv",
            version_comparison_frame(analysis).to_csv(index=False),
        ),
        "metrics_json": ExportPayload(
            "metrics_summary.json",
            "application/json",
            json.dumps(analysis_summary_dict(analysis), indent=2) + "\n",
        ),
        "report_html": ExportPayload(
            "rad_ai_sentinel_report.html",
            "text/html",
            render_report_html(analysis),
        ),
    }
