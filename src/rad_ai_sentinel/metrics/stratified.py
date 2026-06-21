"""Stratified performance analysis.

Computes the full binary metric set within each level of one or more categorical
stratifier columns (site, scanner, modality, age group, sex, race/ethnicity).
Demographic stratifiers are handled with extra documentation about sample sizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import (
    COL_Y_PRED_BINARY,
    COL_Y_TRUE,
    DEFAULT_CONFIDENCE_LEVEL,
    DEMOGRAPHIC_COLUMNS,
    OPERATIONAL_COLUMNS,
)
from .binary import BinaryMetrics, compute_binary_metrics


@dataclass(frozen=True)
class StratifiedResult:
    """Metrics for one level within a stratifier."""

    stratifier: str  # column name
    level: str  # value of the stratifier
    n: int  # sample size in this level
    metrics: BinaryMetrics


def stratify(
    df: pd.DataFrame,
    stratifier: str,
    *,
    y_true_col: str = COL_Y_TRUE,
    y_pred_col: str = COL_Y_PRED_BINARY,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    min_n: int = 10,
) -> list[StratifiedResult]:
    """Compute binary metrics within each level of a single stratifier.

    Levels with fewer than ``min_n`` samples are included but flagged in the
    result (their confidence intervals will be wide, which is informative).

    Parameters
    ----------
    df:
        Validated dataframe.
    stratifier:
        Column name to stratify by (e.g. ``"sex"``, ``"site"``).
    y_true_col, y_pred_col:
        Names of the label and prediction columns.
    confidence:
        Confidence level for Wilson CIs.
    min_n:
        Minimum sample size to compute metrics; levels with n < min_n get a
        placeholder with ``nan`` metrics.

    Returns
    -------
    list[StratifiedResult]
        One result per level of the stratifier, sorted by level name.
    """
    if stratifier not in df.columns:
        return []

    results: list[StratifiedResult] = []
    for level, grp in df.groupby(stratifier, dropna=True):
        grp = grp[[y_true_col, y_pred_col]].dropna()
        n = len(grp)
        if n < min_n:
            # Still record the level so the user can see it has insufficient data.
            from .binary import CI, ConfusionCounts

            nan_ci = CI(estimate=float("nan"), lower=float("nan"), upper=float("nan"))
            nan_metrics = BinaryMetrics(
                sensitivity=nan_ci,
                specificity=nan_ci,
                ppv=nan_ci,
                npv=nan_ci,
                accuracy=nan_ci,
                f1=float("nan"),
                prevalence=float("nan"),
                counts=ConfusionCounts(tp=0, fp=0, tn=0, fn=0),
            )
            results.append(
                StratifiedResult(
                    stratifier=stratifier,
                    level=str(level),
                    n=n,
                    metrics=nan_metrics,
                )
            )
            continue

        m = compute_binary_metrics(grp[y_true_col], grp[y_pred_col], confidence=confidence)
        results.append(
            StratifiedResult(
                stratifier=stratifier,
                level=str(level),
                n=n,
                metrics=m,
            )
        )

    results.sort(key=lambda r: r.level)
    return results


def stratify_all(
    df: pd.DataFrame,
    stratifiers: list[str] | None = None,
    **kwargs: Any,
) -> dict[str, list[StratifiedResult]]:
    """Stratify by all (or a subset of) available stratifier columns.

    Returns a dict mapping each stratifier name to its list of results.
    """
    if stratifiers is None:
        stratifiers = list(DEMOGRAPHIC_COLUMNS + OPERATIONAL_COLUMNS)
    available = [s for s in stratifiers if s in df.columns]
    return {s: stratify(df, s, **kwargs) for s in available}


def max_subgroup_gap(
    results: list[StratifiedResult],
    metric: str = "sensitivity",
) -> float:
    """Compute the maximum absolute gap in a metric across subgroup levels.

    Returns 0.0 if fewer than two levels have non-NaN estimates.
    """
    values = [
        getattr(r.metrics, metric).estimate
        for r in results
        if not np.isnan(getattr(r.metrics, metric).estimate)
    ]
    if len(values) < 2:
        return 0.0
    return float(max(values) - min(values))
