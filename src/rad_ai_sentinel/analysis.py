"""Orchestration layer for complete AI performance monitoring analyses."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import (
    COL_MODEL_VERSION,
    COL_STUDY_DATE,
    COL_Y_PRED_BINARY,
    COL_Y_PRED_PROBA,
    COL_Y_TRUE,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_THRESHOLDS,
    AlertThresholds,
)
from .drift import (
    AlertReport,
    DriftResult,
    MissingDataReport,
    VersionComparison,
    analyze_missing_data,
    check_alerts,
    compare_all_versions,
    compute_drift,
)
from .metrics import FullMetricsResult, compute_all_metrics
from .metrics.binary import BinaryMetrics, compute_binary_metrics
from .metrics.stratified import max_subgroup_gap
from .schemas import validate_dataframe


@dataclass(frozen=True)
class MonitoringAnalysis:
    """Complete result bundle shared by CLI, dashboard, and report."""

    dataframe: pd.DataFrame
    metrics: FullMetricsResult
    drift: DriftResult
    missing: MissingDataReport
    alerts: AlertReport
    version_comparisons: list[VersionComparison]
    baseline_auroc: float | None
    current_auroc: float | None
    current_binary: BinaryMetrics | None
    subgroup_sensitivity_gap: float
    reference_fraction: float

    @property
    def start_date(self) -> pd.Timestamp:
        return pd.to_datetime(self.dataframe[COL_STUDY_DATE]).min()

    @property
    def end_date(self) -> pd.Timestamp:
        return pd.to_datetime(self.dataframe[COL_STUDY_DATE]).max()


def load_and_validate_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV and validate it against the public contract."""
    return validate_dataframe(pd.read_csv(path))


def run_monitoring_analysis(
    df: pd.DataFrame,
    *,
    thresholds: AlertThresholds = DEFAULT_THRESHOLDS,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    reference_fraction: float = 0.30,
    n_resamples: int = 200,
) -> MonitoringAnalysis:
    """Validate data and run the full monitoring pipeline."""
    validated = validate_dataframe(df).sort_values(COL_STUDY_DATE).reset_index(drop=True)
    metrics = compute_all_metrics(
        validated,
        confidence=confidence,
        n_resamples=n_resamples,
    )
    drift = compute_drift(
        validated,
        reference_fraction=reference_fraction,
        psi_minor=thresholds.psi_minor,
        psi_major=thresholds.psi_major,
    )
    missing = analyze_missing_data(validated)
    version_comparisons = compare_all_versions(validated, confidence=confidence)

    baseline_df, current_df = _split_reference_current(validated, reference_fraction)
    baseline_auroc = _safe_auroc(baseline_df)
    current_auroc = _safe_auroc(current_df)
    current_binary = None
    if len(current_df) > 0:
        current_binary = compute_binary_metrics(
            current_df[COL_Y_TRUE].values,
            current_df[COL_Y_PRED_BINARY].values,
            confidence=confidence,
        )

    subgroup_gap = _max_sensitivity_gap(metrics)
    alerts = check_alerts(
        current_auroc=current_auroc if current_auroc is not None else metrics.roc.area.estimate,
        baseline_auroc=baseline_auroc,
        current_sensitivity=(
            current_binary.sensitivity.estimate
            if current_binary is not None
            else metrics.binary.sensitivity.estimate
        ),
        current_specificity=(
            current_binary.specificity.estimate
            if current_binary is not None
            else metrics.binary.specificity.estimate
        ),
        psi_value=drift.psi_value,
        subgroup_sens_gap=subgroup_gap,
        thresholds=thresholds,
    )

    return MonitoringAnalysis(
        dataframe=validated,
        metrics=metrics,
        drift=drift,
        missing=missing,
        alerts=alerts,
        version_comparisons=version_comparisons,
        baseline_auroc=baseline_auroc,
        current_auroc=current_auroc,
        current_binary=current_binary,
        subgroup_sensitivity_gap=subgroup_gap,
        reference_fraction=reference_fraction,
    )


def write_analysis_outputs(analysis: MonitoringAnalysis, output_dir: str | Path) -> dict[str, Path]:
    """Persist analysis tables as CSV plus a compact JSON summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": out / "summary_metrics.csv",
        "stratified": out / "stratified_metrics.csv",
        "missing": out / "missing_data.csv",
        "alerts": out / "alerts.csv",
        "drift": out / "drift.csv",
        "versions": out / "model_versions.csv",
        "json": out / "metrics_summary.json",
    }
    summary_metrics_frame(analysis).to_csv(outputs["summary"], index=False)
    stratified_metrics_frame(analysis).to_csv(outputs["stratified"], index=False)
    missing_data_frame(analysis).to_csv(outputs["missing"], index=False)
    alerts_frame(analysis).to_csv(outputs["alerts"], index=False)
    drift_frame(analysis).to_csv(outputs["drift"], index=False)
    version_comparison_frame(analysis).to_csv(outputs["versions"], index=False)
    outputs["json"].write_text(
        json.dumps(analysis_summary_dict(analysis), indent=2),
        encoding="utf-8",
    )
    return outputs


def summary_metrics_frame(analysis: MonitoringAnalysis) -> pd.DataFrame:
    """Top-line metrics as a tidy table."""
    binary = analysis.metrics.binary
    rows = [
        _metric_row("Sensitivity", binary.sensitivity.estimate, binary.sensitivity),
        _metric_row("Specificity", binary.specificity.estimate, binary.specificity),
        _metric_row("PPV", binary.ppv.estimate, binary.ppv),
        _metric_row("NPV", binary.npv.estimate, binary.npv),
        _metric_row("Accuracy", binary.accuracy.estimate, binary.accuracy),
        _metric_row("F1", binary.f1, None),
        _metric_row("AUROC", analysis.metrics.roc.area.estimate, analysis.metrics.roc.area),
        _metric_row("AUPRC", analysis.metrics.pr.area.estimate, analysis.metrics.pr.area),
        _metric_row(
            "Brier score",
            analysis.metrics.calibration.brier.estimate,
            analysis.metrics.calibration.brier,
        ),
        _metric_row("Expected calibration error", analysis.metrics.calibration.ece, None),
        _metric_row("Prevalence", analysis.metrics.prevalence, None),
    ]
    return pd.DataFrame(rows)


def stratified_metrics_frame(analysis: MonitoringAnalysis) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stratifier, results in analysis.metrics.stratified.items():
        for result in results:
            m = result.metrics
            rows.append(
                {
                    "stratifier": stratifier,
                    "level": result.level,
                    "n": result.n,
                    "sensitivity": _round(m.sensitivity.estimate),
                    "specificity": _round(m.specificity.estimate),
                    "ppv": _round(m.ppv.estimate),
                    "npv": _round(m.npv.estimate),
                    "accuracy": _round(m.accuracy.estimate),
                    "prevalence": _round(m.prevalence),
                    "tp": m.counts.tp,
                    "fp": m.counts.fp,
                    "tn": m.counts.tn,
                    "fn": m.counts.fn,
                }
            )
    return pd.DataFrame(rows)


def missing_data_frame(analysis: MonitoringAnalysis) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": item.column,
                "n_missing": item.n_missing,
                "n_total": item.n_total,
                "pct_missing": _round(item.pct_missing),
                "is_metadata": item.is_metadata,
                "mar_indicator": item.column in analysis.missing.mar_indicators,
            }
            for item in analysis.missing.per_column
        ]
    )


def alerts_frame(analysis: MonitoringAnalysis) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "severity": alert.severity,
                "rule": alert.rule,
                "message": alert.message,
                "observed": _round(alert.observed),
                "threshold": _round(alert.threshold),
            }
            for alert in analysis.alerts.alerts
        ],
        columns=["severity", "rule", "message", "observed", "threshold"],
    )


def drift_frame(analysis: MonitoringAnalysis) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "PSI",
                "value": _round(analysis.drift.psi_value),
                "level": analysis.drift.psi_level,
            },
            {
                "metric": "KL divergence",
                "value": _round(analysis.drift.kl_value),
                "level": "",
            },
            {
                "metric": "Baseline AUROC",
                "value": _round(analysis.baseline_auroc),
                "level": "reference window",
            },
            {
                "metric": "Current AUROC",
                "value": _round(analysis.current_auroc),
                "level": "current window",
            },
            {
                "metric": "Subgroup sensitivity gap",
                "value": _round(analysis.subgroup_sensitivity_gap),
                "level": "max gap",
            },
        ]
    )


def rolling_auroc_frame(analysis: MonitoringAnalysis) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(analysis.drift.rolling.dates),
            "auroc": analysis.drift.rolling.auroc,
        }
    )


def version_comparison_frame(analysis: MonitoringAnalysis) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for version, metrics in analysis.metrics.versions.items():
        rows.append(
            {
                "comparison_type": "version_summary",
                "version_a": version,
                "version_b": "",
                "sensitivity_a": _round(metrics.sensitivity.estimate),
                "sensitivity_b": None,
                "specificity_a": _round(metrics.specificity.estimate),
                "specificity_b": None,
                "auroc_a": None,
                "auroc_b": None,
                "auc_diff": None,
                "p_value": None,
            }
        )
    for comparison in analysis.version_comparisons:
        rows.append(
            {
                "comparison_type": "pairwise",
                "version_a": comparison.version_a,
                "version_b": comparison.version_b,
                "sensitivity_a": _round(comparison.metrics_a.sensitivity.estimate),
                "sensitivity_b": _round(comparison.metrics_b.sensitivity.estimate),
                "specificity_a": _round(comparison.metrics_a.specificity.estimate),
                "specificity_b": _round(comparison.metrics_b.specificity.estimate),
                "auroc_a": _round(comparison.delong.auc1) if comparison.delong else "",
                "auroc_b": _round(comparison.delong.auc2) if comparison.delong else "",
                "auc_diff": _round(comparison.delong.auc_diff) if comparison.delong else "",
                "p_value": _round(comparison.delong.p_value) if comparison.delong else "",
            }
        )
    return pd.DataFrame(rows)


def analysis_summary_dict(analysis: MonitoringAnalysis) -> dict[str, Any]:
    """Small JSON-serializable summary for automation and CI smoke tests."""
    return {
        "n": analysis.metrics.n,
        "n_positive": analysis.metrics.n_positive,
        "start_date": analysis.start_date.strftime("%Y-%m-%d"),
        "end_date": analysis.end_date.strftime("%Y-%m-%d"),
        "auroc": _round(analysis.metrics.roc.area.estimate),
        "auprc": _round(analysis.metrics.pr.area.estimate),
        "sensitivity": _round(analysis.metrics.binary.sensitivity.estimate),
        "specificity": _round(analysis.metrics.binary.specificity.estimate),
        "psi": _round(analysis.drift.psi_value),
        "psi_level": analysis.drift.psi_level,
        "alerts": len(analysis.alerts.alerts),
        "critical_alerts": analysis.alerts.n_critical,
        "warning_alerts": analysis.alerts.n_warning,
        "model_versions": sorted(analysis.dataframe[COL_MODEL_VERSION].dropna().unique().tolist()),
    }


def _split_reference_current(
    df: pd.DataFrame, reference_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = max(min(int(len(df) * reference_fraction), len(df) - 1), 1)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def _safe_auroc(df: pd.DataFrame) -> float | None:
    if len(df) < 2 or df[COL_Y_TRUE].nunique() < 2:
        return None
    try:
        return float(roc_auc_score(df[COL_Y_TRUE].values, df[COL_Y_PRED_PROBA].values))
    except ValueError:
        return None


def _max_sensitivity_gap(metrics: FullMetricsResult) -> float:
    gap = 0.0
    for results in metrics.stratified.values():
        gap = max(gap, max_subgroup_gap(results, "sensitivity"))
    return float(gap)


def _metric_row(name: str, value: float, ci: Any | None) -> dict[str, Any]:
    return {
        "metric": name,
        "estimate": _round(value),
        "ci_lower": _round(ci.lower) if ci is not None else None,
        "ci_upper": _round(ci.upper) if ci is not None else None,
    }


def _round(value: float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), 4)
