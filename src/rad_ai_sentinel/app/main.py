"""Streamlit dashboard for rad-ai-sentinel."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from rad_ai_sentinel.analysis import (
    alerts_frame,
    drift_frame,
    missing_data_frame,
    run_monitoring_analysis,
    stratified_metrics_frame,
    summary_metrics_frame,
    version_comparison_frame,
)
from rad_ai_sentinel.config import AlertThresholds
from rad_ai_sentinel.data import generate_synthetic_monitoring_data
from rad_ai_sentinel.plots import (
    calibration_figure,
    pr_figure,
    roc_figure,
    rolling_auroc_figure,
    score_distribution_figure,
    subgroup_metric_figure,
)
from rad_ai_sentinel.report import render_report_html

st.set_page_config(page_title="rad-ai-sentinel", layout="wide")


@st.cache_data(show_spinner=False)
def _synthetic(n: int, seed: int) -> pd.DataFrame:
    return generate_synthetic_monitoring_data(n=n, seed=seed)


@st.cache_data(show_spinner=False)
def _analyze(csv_bytes: bytes | None, n: int, seed: int, thresholds: AlertThresholds):
    df = _synthetic(n, seed) if csv_bytes is None else pd.read_csv(io.BytesIO(csv_bytes))
    return run_monitoring_analysis(df, thresholds=thresholds, n_resamples=80)


def _thresholds_from_sidebar() -> AlertThresholds:
    with st.sidebar.expander("Alert Thresholds", expanded=False):
        min_auroc = st.slider("Minimum AUROC", 0.50, 0.99, 0.80, 0.01)
        max_drop = st.slider("Maximum AUROC drop", 0.01, 0.30, 0.05, 0.01)
        min_sens = st.slider("Minimum sensitivity", 0.50, 0.99, 0.80, 0.01)
        min_spec = st.slider("Minimum specificity", 0.50, 0.99, 0.80, 0.01)
        psi_minor = st.slider("Minor PSI", 0.01, 0.24, 0.10, 0.01)
        psi_major = st.slider("Major PSI", 0.10, 0.60, 0.25, 0.01)
        gap = st.slider("Sensitivity gap", 0.01, 0.30, 0.05, 0.01)
    return AlertThresholds(
        min_auroc=min_auroc,
        max_auroc_drop_relative=max_drop,
        min_sensitivity=min_sens,
        min_specificity=min_spec,
        psi_minor=psi_minor,
        psi_major=psi_major,
        max_subgroup_sens_gap=gap,
    )


st.sidebar.title("rad-ai-sentinel")
mode = st.sidebar.radio("Data", ["Synthetic demo", "Upload CSV"])
uploaded_bytes: bytes | None = None
n = st.sidebar.slider("Synthetic rows", 300, 3000, 1200, 100)
seed = st.sidebar.number_input("Seed", min_value=0, max_value=100000, value=42, step=1)
if mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Monitoring CSV", type=["csv"])
    if uploaded is not None:
        uploaded_bytes = uploaded.getvalue()
    else:
        st.sidebar.info("Using synthetic data until a CSV is uploaded.")
thresholds = _thresholds_from_sidebar()

analysis = _analyze(uploaded_bytes, n, int(seed), thresholds)
summary = summary_metrics_frame(analysis)
alerts = alerts_frame(analysis)

st.title("rad-ai-sentinel")
st.caption(
    "Radiology AI performance monitoring across time, sites, scanners, subgroups, and versions."
)

metric_cols = st.columns(5)
metric_cols[0].metric("Studies", f"{analysis.metrics.n:,}")
metric_cols[1].metric("AUROC", f"{analysis.metrics.roc.area.estimate:.3f}")
metric_cols[2].metric("AUPRC", f"{analysis.metrics.pr.area.estimate:.3f}")
metric_cols[3].metric("Sensitivity", f"{analysis.metrics.binary.sensitivity.estimate:.3f}")
metric_cols[4].metric("PSI", f"{analysis.drift.psi_value:.3f}", analysis.drift.psi_level)

if alerts.empty:
    st.success("No stop-rule alerts fired.")
else:
    st.error(
        f"{analysis.alerts.n_critical} critical and {analysis.alerts.n_warning} warning alerts fired."
    )
    st.dataframe(alerts, width="stretch", hide_index=True)

overview, calibration, subgroups, drift, missing, versions, report = st.tabs(
    ["Overview", "Calibration", "Subgroups", "Drift", "Missing Data", "Versions", "Report"]
)

with overview:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(roc_figure(analysis), width="stretch")
    with right:
        st.plotly_chart(pr_figure(analysis), width="stretch")
    st.dataframe(summary, width="stretch", hide_index=True)

with calibration:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(calibration_figure(analysis), width="stretch")
    with right:
        st.plotly_chart(score_distribution_figure(analysis), width="stretch")

with subgroups:
    st.plotly_chart(subgroup_metric_figure(analysis), width="stretch")
    st.dataframe(stratified_metrics_frame(analysis), width="stretch", hide_index=True)

with drift:
    left, right = st.columns([1, 2])
    with left:
        st.dataframe(drift_frame(analysis), width="stretch", hide_index=True)
    with right:
        st.plotly_chart(rolling_auroc_figure(analysis), width="stretch")

with missing:
    st.dataframe(missing_data_frame(analysis), width="stretch", hide_index=True)

with versions:
    st.dataframe(version_comparison_frame(analysis), width="stretch", hide_index=True)

with report:
    html = render_report_html(analysis)
    st.download_button(
        "Download HTML Report",
        html,
        file_name="rad_ai_sentinel_report.html",
        mime="text/html",
    )
    st.download_button(
        "Download Analyzed CSV",
        analysis.dataframe.to_csv(index=False),
        file_name="validated_monitoring_data.csv",
        mime="text/csv",
    )
