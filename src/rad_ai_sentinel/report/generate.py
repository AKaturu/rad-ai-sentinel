"""HTML and optional PDF report generation.

Dual-engine PDF strategy:
  - **Primary**: WeasyPrint (full CSS layout, requires GTK3 on Windows)
  - **Fallback**: fpdf2 (pure Python, no native deps, uses DejaVu Sans if available)
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

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
    site_calibration_frame,
    stratified_metrics_frame,
    summary_metrics_frame,
    version_comparison_frame,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette (shared across WeasyPrint CSS and fpdf2 fallback)
# ---------------------------------------------------------------------------
INK = "#1e2329"
MUTED = "#64707d"
LINE = "#d7dde5"
PANEL = "#f5f7fa"
BLUE = "#1769aa"
GREEN = "#1f7a4d"
ORANGE = "#b54d12"

# PDF font family name used by fpdf2 helper functions.
# Set once in ``_write_pdf_fpdf2`` after font registration.


@dataclass(frozen=True)
class ReportArtifacts:
    """Generated report paths."""

    html: Path
    pdf: Path | None
    pdf_error: Path | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_monitoring_report(
    data: pd.DataFrame | MonitoringAnalysis,
    output_dir: str | Path,
    *,
    basename: str = "rad_ai_sentinel_report",
    include_pdf: bool = True,
    n_resamples: int = 200,
    audit_log: str | Path | None = None,
    audit_actor: str = "rad-ai-sentinel",
) -> ReportArtifacts:
    """Render a complete monitoring report."""
    from ..audit import append_audit_event, build_artifact_event

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
            _write_pdf_weasyprint(html, pdf_path, out)
        except Exception as exc:
            logger.info("WeasyPrint PDF failed (%s), falling back to fpdf2", exc)
            try:
                _write_pdf_fpdf2(analysis, pdf_path)
            except Exception as exc2:
                logger.warning("fpdf2 PDF also failed (%s), writing error file", exc2)
                pdf_path = None
                error_path = out / f"{basename}.pdf.error.txt"
                error_path.write_text(
                    f"WeasyPrint error:\n{exc}\n\nfpdf2 fallback error:\n{exc2}\n",
                    encoding="utf-8",
                )

    if audit_log:
        append_audit_event(
            audit_log,
            build_artifact_event(
                event_type="report_generated",
                actor=audit_actor,
                artifact=html_path,
                details={
                    "pdf": str(pdf_path) if pdf_path else None,
                    "alerts": len(analysis.alerts.alerts),
                    "critical_alerts": analysis.alerts.n_critical,
                },
            ),
        )

    return ReportArtifacts(html=html_path, pdf=pdf_path, pdf_error=error_path)


# ---------------------------------------------------------------------------
# WeasyPrint PDF (primary engine)
# ---------------------------------------------------------------------------


def _write_pdf_weasyprint(html: str, pdf_path: Path, base_url: Path) -> None:
    """Generate PDF using WeasyPrint. Raises on failure."""
    from weasyprint import HTML

    HTML(string=html, base_url=str(base_url)).write_pdf(pdf_path)


# ---------------------------------------------------------------------------
# fpdf2 PDF fallback (pure Python, no native deps)
# ---------------------------------------------------------------------------


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    """Convert ``'#RRGGBB'`` to ``(r, g, b)``."""
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _find_dejavu_sans() -> Path | None:
    """Locate *DejaVuSans.ttf* on the system for Unicode PDF rendering."""
    candidates = [
        Path("C:/Windows/Fonts/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/DejaVuSans.ttf"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _register_fonts(pdf: Any) -> str:
    """Register Unicode fonts and return the font family name to use."""
    dejavu = _find_dejavu_sans()
    if dejavu:
        d = dejavu.parent
        pdf.add_font("DejaVu", "", str(d / "DejaVuSans.ttf"), uni=True)
        pdf.add_font("DejaVu", "B", str(d / "DejaVuSans-Bold.ttf"), uni=True)
        pdf.add_font("DejaVu", "I", str(d / "DejaVuSans-Oblique.ttf"), uni=True)
        pdf.add_font("DejaVu", "BI", str(d / "DejaVuSans-BoldOblique.ttf"), uni=True)
        return "DejaVu"
    return "Helvetica"  # built-in latin-1 fallback


def _write_pdf_fpdf2(
    analysis: MonitoringAnalysis, pdf_path: Path, font_family: str = "Helvetica"
) -> None:
    """Generate a styled PDF using fpdf2 as a cross-platform fallback."""
    from fpdf import FPDF

    summary = analysis_summary_dict(analysis)
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(15, 15, 15)

    font_family = _register_fonts(pdf)

    # --- Cover / Header ---
    pdf.add_page()
    _pdf_header(pdf, summary, font_family)
    _pdf_kpi_boxes(pdf, summary, font_family)

    # --- Stop-Rule Alerts ---
    _pdf_section(pdf, "Stop-Rule Alerts", 1, font_family)
    alerts_df = alerts_frame(analysis)
    if alerts_df.empty:
        pdf.set_font(font_family, "I", 10)
        pdf.set_text_color(*_hex_to_rgb(MUTED))
        pdf.cell(0, 6, "No stop-rule alerts fired.", new_x="LMARGIN", new_y="NEXT")
    else:
        _pdf_table(pdf, alerts_df, font_family)

    # --- Executive Metrics ---
    _pdf_section(pdf, "Executive Metrics", 1, font_family)
    _pdf_table(pdf, summary_metrics_frame(analysis), font_family)

    # --- Plots (ROC + PR) ---
    pdf.add_page()
    _pdf_section(pdf, "ROC Curve", 1, font_family)
    _pdf_embed_plot(pdf, analysis, "roc")
    _pdf_section(pdf, "Precision-Recall Curve", 1, font_family)
    _pdf_embed_plot(pdf, analysis, "pr")

    # --- Calibration + Rolling ---
    pdf.add_page()
    _pdf_section(pdf, "Calibration Curve", 1, font_family)
    _pdf_embed_plot(pdf, analysis, "calibration")
    _pdf_section(pdf, "Rolling AUROC", 1, font_family)
    _pdf_embed_plot(pdf, analysis, "rolling")

    # --- Temporal Drift ---
    _pdf_section(pdf, "Temporal Drift", 1, font_family)
    _pdf_table(pdf, drift_frame(analysis), font_family)

    # --- Site Calibration Drift ---
    _pdf_section(pdf, "Site-Level Calibration Drift", 1, font_family)
    _pdf_table(pdf, site_calibration_frame(analysis), font_family)

    # --- Subgroup / Scanner / Site ---
    pdf.add_page()
    _pdf_section(pdf, "Subgroup, Scanner, and Site Performance", 1, font_family)
    _pdf_table(pdf, stratified_metrics_frame(analysis), font_family)

    # --- Missing Data ---
    _pdf_section(pdf, "Missing-Data Analysis", 1, font_family)
    _pdf_table(pdf, missing_data_frame(analysis), font_family)

    # --- Model-Version Comparison ---
    _pdf_section(pdf, "Model-Version Comparison", 1, font_family)
    versions_df = version_comparison_frame(analysis)
    if versions_df.empty:
        pdf.set_font(font_family, "I", 10)
        pdf.set_text_color(*_hex_to_rgb(MUTED))
        pdf.cell(0, 6, "Only one model version was available.", new_x="LMARGIN", new_y="NEXT")
    else:
        _pdf_table(pdf, versions_df, font_family)

    # --- Footer note ---
    pdf.ln(10)
    pdf.set_draw_color(*_hex_to_rgb(LINE))
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)
    pdf.set_font(font_family, "I", 8)
    pdf.set_text_color(*_hex_to_rgb(MUTED))
    pdf.multi_cell(
        0,
        4,
        "This report monitors an existing model\u2019s outputs. "
        "It is not intended to train, validate, or certify an imaging AI model for clinical use.",
    )

    pdf.output(str(pdf_path))


# ---- fpdf2 helper functions ----


def _pdf_header(pdf: Any, summary: dict[str, Any], font_family: str = "Helvetica") -> None:
    """Render the report header block."""
    # Eyebrow
    pdf.set_font(font_family, "B", 9)
    pdf.set_text_color(*_hex_to_rgb(BLUE))
    pdf.cell(0, 5, "RADIOLOGY AI PERFORMANCE MONITOR", new_x="LMARGIN", new_y="NEXT")

    # Title
    pdf.set_font(font_family, "B", 22)
    pdf.set_text_color(*_hex_to_rgb(INK))
    pdf.cell(0, 12, "rad-ai-sentinel Monitoring Report", new_x="LMARGIN", new_y="NEXT")

    # Subtitle
    pdf.set_font(font_family, "", 10)
    pdf.set_text_color(*_hex_to_rgb(MUTED))
    n = summary["n"]
    n_pos = summary["n_positive"]
    prev = f"{n_pos / n:.1%}" if n > 0 else "N/A"
    subtitle = (
        f"{summary['start_date']} to {summary['end_date']}  |  "
        f"{n} studies  |  {n_pos} positive  |  prevalence {prev}"
    )
    pdf.cell(0, 6, subtitle, new_x="LMARGIN", new_y="NEXT")

    # Divider
    pdf.ln(3)
    pdf.set_draw_color(*_hex_to_rgb(LINE))
    pdf.set_line_width(0.8)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(8)


def _pdf_kpi_boxes(pdf: Any, summary: dict[str, Any], font_family: str = "Helvetica") -> None:
    """Render four KPI metric boxes in a row."""
    box_w = 40.5
    box_h = 22
    gap = 3.5
    start_x = 15
    y = pdf.get_y()

    metrics = [
        (
            "Alerts",
            str(summary["alerts"]),
            f"{summary['critical_alerts']} crit, {summary['warning_alerts']} warn",
        ),
        ("AUROC", f"{summary['auroc']:.3f}", "overall discrimination"),
        ("PSI", f"{summary['psi']:.3f}", f"{summary['psi_level']} drift"),
        ("Sensitivity", f"{summary['sensitivity']:.3f}", "current threshold"),
    ]

    for i, (label, value, desc) in enumerate(metrics):
        x = start_x + i * (box_w + gap)
        # Background box
        pdf.set_fill_color(*_hex_to_rgb(PANEL))
        pdf.set_draw_color(*_hex_to_rgb(LINE))
        pdf.rect(x, y, box_w, box_h, style="DF")
        # Label
        pdf.set_xy(x + 3, y + 2)
        pdf.set_font(font_family, "", 8)
        pdf.set_text_color(*_hex_to_rgb(MUTED))
        pdf.cell(box_w - 6, 4, label, new_x="LMARGIN", new_y="NEXT")
        # Value
        pdf.set_xy(x + 3, y + 7)
        pdf.set_font(font_family, "B", 16)
        pdf.set_text_color(*_hex_to_rgb(INK))
        pdf.cell(box_w - 6, 8, value, new_x="LMARGIN", new_y="NEXT")
        # Description
        pdf.set_xy(x + 3, y + 15)
        pdf.set_font(font_family, "", 7)
        pdf.set_text_color(*_hex_to_rgb(MUTED))
        pdf.cell(box_w - 6, 4, desc, new_x="LMARGIN", new_y="NEXT")

    pdf.set_y(y + box_h + 8)


def _pdf_section(pdf: Any, title: str, spacing: int = 1, font_family: str = "Helvetica") -> None:
    """Render a section heading."""
    pdf.ln(spacing * 4)
    pdf.set_font(font_family, "B", 14)
    pdf.set_text_color(*_hex_to_rgb(INK))
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)


def _pdf_table(pdf: Any, df: pd.DataFrame, font_family: str = "Helvetica") -> None:
    """Render a pandas DataFrame as a styled table in the PDF."""
    if df.empty:
        pdf.set_font(font_family, "I", 10)
        pdf.set_text_color(*_hex_to_rgb(MUTED))
        pdf.cell(0, 6, "No data.", new_x="LMARGIN", new_y="NEXT")
        return

    cols = [str(c) for c in df.columns]
    col_count = len(cols)

    available_w = 180  # A4 width minus margins
    col_w = available_w / col_count
    if col_w < 18:
        col_w = 18

    # Switch to landscape if columns overflow
    if col_count * col_w > available_w:
        pdf.add_page(orientation="L")
        col_w = 257 / col_count

    row_h = 6

    # Header row
    pdf.set_font(font_family, "B", 8)
    pdf.set_fill_color(*_hex_to_rgb(PANEL))
    pdf.set_text_color(*_hex_to_rgb(INK))
    for col_name in cols:
        pdf.cell(col_w, row_h, col_name[:24], border=1, fill=True, new_x="RIGHT", new_y="TOP")
    pdf.ln(row_h)

    # Data rows
    pdf.set_font(font_family, "", 8)
    pdf.set_text_color(*_hex_to_rgb(INK))
    for _, row in df.iterrows():
        if pdf.get_y() + row_h > pdf.h - 20:
            pdf.add_page()
            pdf.set_font(font_family, "B", 8)
            pdf.set_fill_color(*_hex_to_rgb(PANEL))
            for col_name in cols:
                pdf.cell(
                    col_w, row_h, col_name[:24], border=1, fill=True, new_x="RIGHT", new_y="TOP"
                )
            pdf.ln(row_h)
            pdf.set_font(font_family, "", 8)

        for col_name in cols:
            val = row[col_name]
            if pd.isna(val):
                text = ""
            elif isinstance(val, float):
                text = f"{val:.4g}"
            else:
                text = str(val)
            pdf.cell(col_w, row_h, text[:28], border=1, new_x="RIGHT", new_y="TOP")
        pdf.ln(row_h)

    pdf.ln(4)


def _pdf_embed_plot(pdf: Any, analysis: MonitoringAnalysis, plot_type: str) -> None:
    """Generate and embed a matplotlib figure into the PDF."""
    buf = io.BytesIO()

    if plot_type == "roc":
        roc = analysis.metrics.roc
        plt.figure(figsize=(5.2, 3.5))
        plt.plot(roc.x, roc.y, color=BLUE, linewidth=2.5, label=f"AUROC {roc.area.estimate:.3f}")
        plt.plot([0, 1], [0, 1], color="#8a8f98", linestyle="--", label="Chance")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
    elif plot_type == "pr":
        pr = analysis.metrics.pr
        plt.figure(figsize=(5.2, 3.5))
        plt.plot(pr.x, pr.y, color=GREEN, linewidth=2.5, label=f"AUPRC {pr.area.estimate:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
    elif plot_type == "calibration":
        cal = analysis.metrics.calibration
        plt.figure(figsize=(5.2, 3.5))
        plt.plot([0, 1], [0, 1], color="#8a8f98", linestyle="--", label="Perfect")
        plt.plot(
            cal.prob_pred,
            cal.prob_true,
            marker="o",
            color=ORANGE,
            linewidth=2.5,
            label=f"ECE {cal.ece:.3f}",
        )
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed event rate")
        plt.title("Calibration Curve")
        plt.legend(loc="upper left")
    elif plot_type == "rolling":
        roll = rolling_auroc_frame(analysis).dropna()
        plt.figure(figsize=(8, 3.5))
        if not roll.empty:
            plt.plot(roll["date"], roll["auroc"], marker="o", color=BLUE, linewidth=2.0)
        plt.ylim(0, 1)
        plt.xlabel("Window center")
        plt.ylabel("AUROC")
        plt.title("Rolling AUROC")
    else:
        return

    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close()

    buf.seek(0)
    img_w = 160
    img_h = img_w * 0.65
    pdf.image(buf, x=15, w=img_w, h=img_h)
    pdf.ln(6)


# ---------------------------------------------------------------------------
# HTML rendering (shared Jinja2 template)
# ---------------------------------------------------------------------------


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
        site_calibration_table=_to_html(
            site_calibration_frame(analysis),
            empty="No site-level calibration drift rows.",
        ),
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
    plt.plot(roc.x, roc.y, color=BLUE, linewidth=2.5, label=f"AUROC {roc.area.estimate:.3f}")
    plt.plot([0, 1], [0, 1], color="#8a8f98", linestyle="--", label="Chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    return _figure_to_data_uri()


def _pr_png(analysis: MonitoringAnalysis) -> str:
    pr = analysis.metrics.pr
    plt.figure(figsize=(5.2, 3.5))
    plt.plot(pr.x, pr.y, color=GREEN, linewidth=2.5, label=f"AUPRC {pr.area.estimate:.3f}")
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
        color=ORANGE,
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
        plt.plot(roll["date"], roll["auroc"], marker="o", color=BLUE, linewidth=2.0)
    plt.ylim(0, 1)
    plt.xlabel("Window center")
    plt.ylabel("AUROC")
    plt.title("Rolling AUROC")
    return _figure_to_data_uri()
