"""Statistical outlier detection for CT radiation dose.

Flags studies with unusually high CTDIvol or DLP *within their protocol group*
using two independent robust methods:

1. **Tukey IQR fence**: values outside ``[Q1 - k*IQR, Q3 + k*IQR]`` are
   flagged, with ``k=1.5`` (standard outlier) and ``k=3.0`` (far outlier).
   Percentile-based, robust to skew.

2. **MAD modified z-score**: ``0.6745*(x - median) / MAD``; values >= 3.5
   are flagged. Robust to heavy-tailed distributions where the IQR can be
   degenerate.

A **minimum group size** guard (default ``n ≥ 8``) prevents unreliable flagging
in small protocol groups — the fence estimate is statistically unstable below
that threshold.

A flagged outlier is a **statistical** finding, NOT a clinical-safety
determination. Interpretation requires institutional review against validated
benchmarks and qualified medical-physics oversight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..config import (
    COL_PATIENT_ID,
    COL_PROTOCOL,
    COL_STUDY_DATE,
    COL_STUDY_UID,
    DEFAULT_OUTLIER_CONFIG,
    DOSE_METRIC_COLUMNS,
    OutlierConfig,
)


@dataclass(frozen=True)
class OutlierFlag:
    """One flagged outlier study."""

    study_uid: str
    patient_id: str
    study_date: str
    protocol: str
    metric: str
    value: float
    # The fence/boundary for reference.
    lower_fence: float
    upper_fence: float
    iqr: float
    mad_z_score: float
    # Severity: "outlier" (1.5*IQR) or "far_outlier" (3.0*IQR).
    severity: str
    method: str  # "iqr" or "mad" or "both"


def detect_outliers(
    df: pd.DataFrame,
    *,
    config: OutlierConfig = DEFAULT_OUTLIER_CONFIG,
    group_by: str = COL_PROTOCOL,
) -> list[OutlierFlag]:
    """Flag statistical dose outliers per protocol group.

    Parameters
    ----------
    df:
        Validated dose dataframe.
    config:
        Outlier detection parameters (IQR multipliers, min group size, MAD threshold).
    group_by:
        Column to group by (default: protocol).

    Returns
    -------
    list of OutlierFlag, one per flagged study per metric. Empty when groups
    are too small or no outliers exist.
    """
    if group_by not in df.columns:
        return []

    flags: list[OutlierFlag] = []
    for _, sub in df.groupby(group_by, observed=True):
        if len(sub) < config.min_group_size:
            continue
        for metric in DOSE_METRIC_COLUMNS:
            if metric not in sub.columns:
                continue
            flags.extend(_flag_metric(sub, metric, config))
    return flags


def _flag_metric(
    sub: pd.DataFrame,
    metric: str,
    config: OutlierConfig,
) -> list[OutlierFlag]:
    """Apply IQR + MAD outlier detection for one metric in one group."""
    vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
    if len(vals) < config.min_group_size:
        return []

    q1 = float(vals.quantile(0.25))
    q3 = float(vals.quantile(0.75))
    iqr = q3 - q1
    median = float(vals.median())

    # MAD: median absolute deviation, with constant for asymptotic normality.
    mad = float((vals - median).abs().median())
    mad_z_scale = 0.6745 * mad if mad > 0 else 1e-9  # avoid div/0

    flags: list[OutlierFlag] = []
    for _, row in sub.iterrows():
        raw = row[metric]
        if pd.isna(raw):
            continue
        value = float(raw)
        mad_z = 0.6745 * abs(value - median) / mad_z_scale

        iqr_outlier = False
        far_outlier = False
        method = ""

        lower = q1 - config.iqr_multiplier * iqr
        upper = q3 + config.iqr_multiplier * iqr
        lower_far = q1 - config.iqr_multiplier_far * iqr
        upper_far = q3 + config.iqr_multiplier_far * iqr

        is_iqr_outlier = value < lower or value > upper
        is_far_outlier = value < lower_far or value > upper_far
        is_mad_outlier = mad_z >= config.mad_z_threshold

        if is_iqr_outlier or is_far_outlier:
            iqr_outlier = True
            if is_far_outlier:
                far_outlier = True

        if iqr_outlier and is_mad_outlier:
            method = "both"
        elif iqr_outlier:
            method = "iqr"
        elif is_mad_outlier:
            method = "mad"
        else:
            continue

        severity = "far_outlier" if far_outlier else "outlier"
        study_date_val = row.get(COL_STUDY_DATE, "")
        study_date_str = str(study_date_val) if not pd.isna(study_date_val) else ""

        flags.append(
            OutlierFlag(
                study_uid=str(row.get(COL_STUDY_UID, "")),
                patient_id=str(row.get(COL_PATIENT_ID, "")),
                study_date=study_date_str,
                protocol=str(row.get(COL_PROTOCOL, "")),
                metric=metric,
                value=value,
                lower_fence=lower,
                upper_fence=upper,
                iqr=iqr,
                mad_z_score=round(mad_z, 4),
                severity=severity,
                method=method,
            )
        )
    return flags


def outliers_dataframe(flags: list[OutlierFlag]) -> pd.DataFrame:
    """Convert flagged outliers to a tidy DataFrame."""
    if not flags:
        return pd.DataFrame(
            columns=[
                "study_uid", "patient_id", "study_date", "protocol",
                "metric", "value", "lower_fence", "upper_fence",
                "iqr", "mad_z_score", "severity", "method",
            ]
        )
    rows: list[dict[str, Any]] = []
    for f in flags:
        rows.append(
            {
                "study_uid": f.study_uid,
                "patient_id": f.patient_id,
                "study_date": f.study_date,
                "protocol": f.protocol,
                "metric": f.metric,
                "value": round(f.value, 4),
                "lower_fence": round(f.lower_fence, 4),
                "upper_fence": round(f.upper_fence, 4),
                "iqr": round(f.iqr, 4),
                "mad_z_score": f.mad_z_score,
                "severity": f.severity,
                "method": f.method,
            }
        )
    return pd.DataFrame(rows)
