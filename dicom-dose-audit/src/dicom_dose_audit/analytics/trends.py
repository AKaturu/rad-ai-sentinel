"""Monthly dose trends.

Computes the median CTDIvol and DLP per protocol per calendar month. This is
the primary time-series view for quality-improvement tracking: a rising trend
signals a process drift; a falling trend after a protocol change confirms the
intended dose reduction.

Medians are used instead of means because dose distributions are typically
right-skewed (a few high-dose outliers pull the mean). Monthly granularity is
standard for CT dose audits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..config import (
    COL_PROTOCOL,
    COL_STUDY_DATE,
    DOSE_METRIC_COLUMNS,
)


@dataclass(frozen=True)
class MonthlyTrend:
    """One data point in the monthly trend series."""

    year_month: str  # "YYYY-MM"
    protocol: str
    metric: str
    n: int
    median: float | None
    p25: float | None
    p75: float | None
    mean: float | None


def monthly_trends(
    df: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DOSE_METRIC_COLUMNS,
    group_by: str = COL_PROTOCOL,
) -> list[MonthlyTrend]:
    """Compute monthly median dose trends per protocol.

    Parameters
    ----------
    df:
        Validated dose dataframe. ``study_date`` must be coercible to datetime.
    metrics:
        Dose metric columns to trend.
    group_by:
        Column to group by (default: protocol).

    Returns
    -------
    list of MonthlyTrend sorted by year_month then protocol then metric.
    """
    if COL_STUDY_DATE not in df.columns or group_by not in df.columns:
        return []

    dates = pd.to_datetime(df[COL_STUDY_DATE], errors="coerce")
    if dates.isna().all():
        return []

    df = df.copy()
    df["year_month"] = dates.dt.strftime("%Y-%m")

    trends: list[MonthlyTrend] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        for (ym, protocol), sub in df.groupby(["year_month", group_by], observed=True):
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            n = len(vals)
            trends.append(
                MonthlyTrend(
                    year_month=str(ym),
                    protocol=str(protocol),
                    metric=metric,
                    n=n,
                    median=_safe(vals.median()) if n > 0 else None,
                    p25=_safe(vals.quantile(0.25)) if n > 0 else None,
                    p75=_safe(vals.quantile(0.75)) if n > 0 else None,
                    mean=_safe(vals.mean()) if n > 0 else None,
                )
            )
    return sorted(trends, key=lambda t: (t.year_month, t.protocol, t.metric))


def trends_dataframe(trends: list[MonthlyTrend]) -> pd.DataFrame:
    """Convert trends to a tidy DataFrame for output/plotting."""
    rows: list[dict[str, Any]] = []
    for t in trends:
        rows.append(
            {
                "year_month": t.year_month,
                "protocol": t.protocol,
                "metric": t.metric,
                "n": t.n,
                "median": _round(t.median),
                "p25": _round(t.p25),
                "p75": _round(t.p75),
                "mean": _round(t.mean),
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
