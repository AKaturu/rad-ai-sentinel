"""The metrics engine: discrimination, calibration, and confidence intervals.

This module provides a unified ``compute_all_metrics()`` entry point that runs the
full metric suite (binary 2x2, curves, calibration, stratified) on a validated
dataframe and returns a structured result for the CLI, dashboard, and report.

Sub-modules are also importable individually for targeted analysis:
  - ``binary``: sensitivity, specificity, PPV, NPV, accuracy, F1 (Wilson CIs)
  - ``curves``: AUROC, AUPRC (BCa bootstrap CIs), ROC/PR curve data
  - ``calibration``: Brier score, ECE, reliability diagram data
  - ``stratified``: per-subgroup metrics (site, scanner, modality, demographics)
  - ``ci``: Wilson CI, BCa bootstrap CI primitives
  - ``delong``: DeLong test for pairwise AUROC comparison
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..config import (
    COL_MODEL_VERSION,
    COL_Y_PRED_BINARY,
    COL_Y_PRED_PROBA,
    COL_Y_TRUE,
    DEFAULT_CONFIDENCE_LEVEL,
)
from .binary import BinaryMetrics, compute_binary_metrics
from .calibration import CalibrationMetrics, compute_calibration
from .curves import CurveResult, compute_pr, compute_roc
from .stratified import StratifiedResult, stratify_all


@dataclass(frozen=True)
class FullMetricsResult:
    """The complete metric output for one evaluation window (or full dataset)."""

    # Overall (aggregate) metrics.
    binary: BinaryMetrics
    roc: CurveResult
    pr: CurveResult
    calibration: CalibrationMetrics

    # Stratified metrics: dict[stratifier, list[StratifiedResult]].
    stratified: dict[str, list[StratifiedResult]] = field(default_factory=dict)

    # Per-model-version summary (if multiple versions present).
    versions: dict[str, BinaryMetrics] = field(default_factory=dict)

    # Metadata about the evaluation window.
    n: int = 0
    n_positive: int = 0
    prevalence: float = 0.0


def compute_all_metrics(
    df: pd.DataFrame,
    *,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    stratifiers: list[str] | None = None,
    n_resamples: int = 300,
) -> FullMetricsResult:
    """Run the full metric suite on a validated dataframe.

    Parameters
    ----------
    df:
        Validated dataframe (schema-checked via ``validate_dataframe()``).
    confidence:
        Confidence level for all CIs.
    stratifiers:
        Stratifier columns to analyse; defaults to all available stratifiers.
    n_resamples:
        Bootstrap resamples for AUROC, AUPRC, and Brier confidence intervals.

    Returns
    -------
    FullMetricsResult
    """
    y_true = df[COL_Y_TRUE].values
    y_pred = df[COL_Y_PRED_BINARY].values
    y_proba = df[COL_Y_PRED_PROBA].values
    n = len(df)
    n_positive = int(y_true.sum())
    prevalence = n_positive / n if n > 0 else 0.0

    # Overall metrics.
    binary = compute_binary_metrics(y_true, y_pred, confidence=confidence)
    roc = compute_roc(y_true, y_proba, confidence=confidence, n_resamples=n_resamples)
    pr = compute_pr(y_true, y_proba, confidence=confidence, n_resamples=n_resamples)
    calibration = compute_calibration(y_true, y_proba, n_bins=10, n_resamples=n_resamples)

    # Stratified metrics (only for columns present in df).
    stratified = stratify_all(df, stratifiers=stratifiers, confidence=confidence)

    # Per-version summary (if model_version column is present).
    versions: dict[str, BinaryMetrics] = {}
    if COL_MODEL_VERSION in df.columns:
        for version, vdf in df.groupby(COL_MODEL_VERSION, dropna=True):
            if len(vdf) >= 10:
                versions[str(version)] = compute_binary_metrics(
                    vdf[COL_Y_TRUE].values,
                    vdf[COL_Y_PRED_BINARY].values,
                    confidence=confidence,
                )

    return FullMetricsResult(
        binary=binary,
        roc=roc,
        pr=pr,
        calibration=calibration,
        stratified=stratified,
        versions=versions,
        n=n,
        n_positive=n_positive,
        prevalence=prevalence,
    )


__all__ = [
    "BinaryMetrics",
    "CalibrationMetrics",
    "CurveResult",
    "FullMetricsResult",
    "StratifiedResult",
    "compute_all_metrics",
]
