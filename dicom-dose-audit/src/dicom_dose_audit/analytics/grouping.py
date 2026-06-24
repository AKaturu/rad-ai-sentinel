"""Protocol, scanner, and patient-size grouping with summary statistics.

Groups dose studies by protocol, scanner (model or manufacturer), site, and
patient-size category. For each group, computes n, missing count, and
distributional summaries (mean, median, P25/P75/P90, std, min, max) for CTDIvol
and DLP.

Dose values are allowed to be null — missing-dose rows are counted in the
``missing`` column and excluded from the numeric summaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..config import (
    COL_CTDI_VOL,
    COL_DLP,
    COL_PROTOCOL,
    STRATIFIER_COLUMNS,
)


@dataclass(frozen=True)
class GroupSummary:
    """Summary statistics for one metric within one protocol group."""

    stratifier: str
    level: str
    metric: str  # "ctdi_vol_mgy" or "dlp_mgy_cm"
    n: int
    missing: int
    mean: float | None
    median: float | None
    p25: float | None
    p75: float | None
    p90: float | None
    std: float | None
    min_val: float | None
    max_val: float | None


def group_summary(
    df: pd.DataFrame,
    *,
    stratifier: str = COL_PROTOCOL,
    metric: str = COL_CTDI_VOL,
) -> list[GroupSummary]:
    """Compute per-group summary statistics for a dose metric.

    Parameters
    ----------
    df:
        Validated dose dataframe.
    stratifier:
        Column name to group by (protocol, scanner_model, site, size_category).
    metric:
        Dose metric column (ctdi_vol_mgy or dlp_mgy_cm).

    Returns
    -------
    list of GroupSummary, one per group level.
    """
    if stratifier not in df.columns or metric not in df.columns:
        return []

    groups: list[GroupSummary] = []
    for level, sub in df.groupby(stratifier, observed=True):
        vals = pd.to_numeric(sub[metric], errors="coerce")
        present = vals.dropna()
        n = len(vals)
        missing = n - len(present)
        groups.append(
            GroupSummary(
                stratifier=stratifier,
                level=str(str(level)),
                metric=metric,
                n=n,
                missing=missing,
                mean=_safe(present.mean()) if len(present) > 0 else None,
                median=_safe(present.median()) if len(present) > 0 else None,
                p25=_safe(present.quantile(0.25)) if len(present) > 0 else None,
                p75=_safe(present.quantile(0.75)) if len(present) > 0 else None,
                p90=_safe(present.quantile(0.90)) if len(present) > 0 else None,
                std=_safe(present.std()) if len(present) > 1 else None,
                min_val=_safe(present.min()) if len(present) > 0 else None,
                max_val=_safe(present.max()) if len(present) > 0 else None,
            )
        )
    return sorted(groups, key=lambda g: g.level)


def group_summary_all_stratifiers(
    df: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = (COL_CTDI_VOL, COL_DLP),
) -> list[GroupSummary]:
    """Summarize all dose metrics across every stratifier column."""
    results: list[GroupSummary] = []
    for stratifier in STRATIFIER_COLUMNS:
        if stratifier not in df.columns:
            continue
        for metric in metrics:
            if metric not in df.columns:
                continue
            results.extend(group_summary(df, stratifier=stratifier, metric=metric))
    return results


def group_summary_dataframe(summaries: list[GroupSummary]) -> pd.DataFrame:
    """Convert a list of GroupSummary to a tidy DataFrame for output."""
    rows: list[dict[str, Any]] = []
    for g in summaries:
        rows.append(
            {
                "stratifier": g.stratifier,
                "level": g.level,
                "metric": g.metric,
                "n": g.n,
                "missing": g.missing,
                "mean": _round(g.mean),
                "median": _round(g.median),
                "p25": _round(g.p25),
                "p75": _round(g.p75),
                "p90": _round(g.p90),
                "std": _round(g.std),
                "min": _round(g.min_val),
                "max": _round(g.max_val),
            }
        )
    return pd.DataFrame(rows)


def _safe(val: float | None) -> float | None:
    if val is None:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return float(val)


def _round(val: float | None, ndigits: int = 4) -> float | None:
    if val is None:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return round(float(val), ndigits)
