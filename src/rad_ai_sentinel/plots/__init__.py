"""Plotting helpers for ROC/PR curves, calibration, drift, and dashboards."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go

from ..analysis import MonitoringAnalysis, rolling_auroc_frame, stratified_metrics_frame
from ..config import COL_MODEL_VERSION, COL_Y_PRED_PROBA


def roc_figure(analysis: MonitoringAnalysis) -> go.Figure:
    roc = analysis.metrics.roc
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roc.x,
            y=roc.y,
            mode="lines",
            name=f"AUROC {roc.area.estimate:.3f}",
            line={"color": "#1769aa", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance",
            line={"color": "#8a8f98", "dash": "dash"},
        )
    )
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        legend_orientation="h",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return fig


def pr_figure(analysis: MonitoringAnalysis) -> go.Figure:
    pr = analysis.metrics.pr
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pr.x,
            y=pr.y,
            mode="lines",
            name=f"AUPRC {pr.area.estimate:.3f}",
            line={"color": "#1f7a4d", "width": 3},
        )
    )
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        legend_orientation="h",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return fig


def calibration_figure(analysis: MonitoringAnalysis) -> go.Figure:
    cal = analysis.metrics.calibration
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"color": "#8a8f98", "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cal.prob_pred,
            y=cal.prob_true,
            mode="lines+markers",
            name=f"Observed, ECE {cal.ece:.3f}",
            line={"color": "#b54d12", "width": 3},
        )
    )
    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Mean predicted probability",
        yaxis_title="Observed event rate",
        legend_orientation="h",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return fig


def rolling_auroc_figure(analysis: MonitoringAnalysis) -> go.Figure:
    df = rolling_auroc_frame(analysis).dropna()
    fig = px.line(df, x="date", y="auroc", markers=True, title="Rolling AUROC")
    fig.update_traces(line={"color": "#1769aa", "width": 3})
    fig.update_layout(
        yaxis_range=[0, 1],
        xaxis_title="Window center",
        yaxis_title="AUROC",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return fig


def subgroup_metric_figure(
    analysis: MonitoringAnalysis,
    *,
    metric: str = "sensitivity",
    top_n: int = 24,
) -> go.Figure:
    table = stratified_metrics_frame(analysis)
    if table.empty or metric not in table.columns:
        return go.Figure()
    plot_df = table.dropna(subset=[metric]).copy()
    plot_df["label"] = plot_df["stratifier"] + ": " + plot_df["level"]
    plot_df = plot_df.sort_values(metric).tail(top_n)
    fig = px.bar(
        plot_df,
        x=metric,
        y="label",
        orientation="h",
        color="stratifier",
        title=f"Subgroup {metric.replace('_', ' ').title()}",
        hover_data=["n", "tp", "fp", "tn", "fn"],
    )
    fig.update_layout(
        xaxis_range=[0, 1],
        xaxis_title=metric.replace("_", " ").title(),
        yaxis_title="",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return fig


def score_distribution_figure(analysis: MonitoringAnalysis) -> go.Figure:
    df = analysis.dataframe.copy()
    fig = px.histogram(
        df,
        x=COL_Y_PRED_PROBA,
        color=COL_MODEL_VERSION,
        nbins=30,
        barmode="overlay",
        histnorm="probability density",
        title="Prediction Score Distribution by Model Version",
    )
    fig.update_traces(opacity=0.70)
    fig.update_layout(
        xaxis_title="Predicted probability",
        yaxis_title="Density",
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
    )
    return fig


__all__ = [
    "calibration_figure",
    "pr_figure",
    "roc_figure",
    "rolling_auroc_figure",
    "score_distribution_figure",
    "subgroup_metric_figure",
]
