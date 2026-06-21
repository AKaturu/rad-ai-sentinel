"""HTML and optional PDF report generation."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import matplotlib
import pandas as pd
from jinja2 import Environment, select_autoescape

from ..analysis import (
    MonitoringAnalysis,
    alerts_frame,
    analysis_summary_dict,
    drift_frame,
    missing_data_frame,
    rolling_auroc_frame,
    run_monitoring_analysis,
    stratified_metrics_frame,
    summary_metrics_frame,
    version_comparison_frame,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ReportArtifacts:
    """Generated report paths."""

    html: Path
    pdf: Path | None
    pdf_error: Path | None = None


def generate_monitoring_report(
    data: pd.DataFrame | MonitoringAnalysis,
    output_dir: str | Path,
    *,
    basename: str = "rad_ai_sentinel_report",
    include_pdf: bool = True,
    n_resamples: int = 200,
) -> ReportArtifacts:
    """Render a complete monitoring report."""
    analysis = (
        data
        if isinstance(data, MonitoringAnalysis)
        else run_monitoring_analysis(data, n_resamples=n_resamples)
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    html = render_report_html(analysis)
    html_path = out / f"{basename}.html"
    html_path.write_text(html, encoding="utf-8")

    pdf_path: Path | None = None
    error_path: Path | None = None
    if include_pdf:
        pdf_path = out / f"{basename}.pdf"
        try:
            from weasyprint import HTML

            HTML(string=html, base_url=str(out)).write_pdf(pdf_path)
        except Exception as exc:  # pragma: no cover - depends on system libraries
            pdf_path = None
            error_path = out / f"{basename}.pdf.error.txt"
            error_path.write_text(str(exc), encoding="utf-8")

    return ReportArtifacts(html=html_path, pdf=pdf_path, pdf_error=error_path)


def render_report_html(analysis: MonitoringAnalysis) -> str:
    """Render the report into an HTML string."""
    template_text = files("rad_ai_sentinel.report").joinpath("templates", "report.html").read_text()
    css = files("rad_ai_sentinel.report").joinpath("templates", "styles.css").read_text()
    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    template = env.from_string(template_text)
    return template.render(
        css=css,
        summary=analysis_summary_dict(analysis),
        start_date=analysis.start_date.strftime("%Y-%m-%d"),
        end_date=analysis.end_date.strftime("%Y-%m-%d"),
        n=analysis.metrics.n,
        n_positive=analysis.metrics.n_positive,
        prevalence=f"{analysis.metrics.prevalence:.1%}",
        alert_count=len(analysis.alerts.alerts),
        critical_alerts=analysis.alerts.n_critical,
        warning_alerts=analysis.alerts.n_warning,
        summary_table=_to_html(summary_metrics_frame(analysis)),
        drift_table=_to_html(drift_frame(analysis)),
        alerts_table=_to_html(alerts_frame(analysis), empty="No stop-rule alerts fired."),
        stratified_table=_to_html(stratified_metrics_frame(analysis)),
        missing_table=_to_html(missing_data_frame(analysis)),
        versions_table=_to_html(
            version_comparison_frame(analysis),
            empty="Only one model version was available for comparison.",
        ),
        roc_plot=_roc_png(analysis),
        pr_plot=_pr_png(analysis),
        calibration_plot=_calibration_png(analysis),
        rolling_plot=_rolling_png(analysis),
    )


def _to_html(df: pd.DataFrame, *, empty: str = "No rows.") -> str:
    if df.empty:
        return f'<p class="empty">{empty}</p>'
    return df.to_html(index=False, classes="data-table", border=0, na_rep="")


def _figure_to_data_uri() -> str:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close()
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _roc_png(analysis: MonitoringAnalysis) -> str:
    roc = analysis.metrics.roc
    plt.figure(figsize=(5.2, 3.5))
    plt.plot(roc.x, roc.y, color="#1769aa", linewidth=2.5, label=f"AUROC {roc.area.estimate:.3f}")
    plt.plot([0, 1], [0, 1], color="#8a8f98", linestyle="--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    return _figure_to_data_uri()


def _pr_png(analysis: MonitoringAnalysis) -> str:
    pr = analysis.metrics.pr
    plt.figure(figsize=(5.2, 3.5))
    plt.plot(pr.x, pr.y, color="#1f7a4d", linewidth=2.5, label=f"AUPRC {pr.area.estimate:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    return _figure_to_data_uri()


def _calibration_png(analysis: MonitoringAnalysis) -> str:
    cal = analysis.metrics.calibration
    plt.figure(figsize=(5.2, 3.5))
    plt.plot([0, 1], [0, 1], color="#8a8f98", linestyle="--", label="Perfect")
    plt.plot(
        cal.prob_pred,
        cal.prob_true,
        marker="o",
        color="#b54d12",
        linewidth=2.5,
        label=f"ECE {cal.ece:.3f}",
    )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Calibration Curve")
    plt.legend(loc="upper left")
    return _figure_to_data_uri()


def _rolling_png(analysis: MonitoringAnalysis) -> str:
    roll = rolling_auroc_frame(analysis).dropna()
    plt.figure(figsize=(8, 3.5))
    if not roll.empty:
        plt.plot(roll["date"], roll["auroc"], marker="o", color="#1769aa", linewidth=2.0)
    plt.ylim(0, 1)
    plt.xlabel("Window center")
    plt.ylabel("AUROC")
    plt.title("Rolling AUROC")
    return _figure_to_data_uri()
