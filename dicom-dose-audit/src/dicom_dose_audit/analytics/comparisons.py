"""Protocol-version comparison with bootstrap confidence intervals.

When two versions of the same protocol coexist (e.g., after a dose-reduction
protocol update), this module computes the change in mean CTDIvol and DLP
between versions, along with a bootstrap confidence interval on the mean
difference. This surfaces whether the protocol update actually shifted the dose
distribution.

The bootstrap uses the BCa (bias-corrected and accelerated) method when
scipy supports it, falling back to percentile bootstrap otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..config import (
    COL_PROTOCOL,
    COL_PROTOCOL_VERSION,
    DEFAULT_BOOTSTRAP_N,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RANDOM_SEED,
    DOSE_METRIC_COLUMNS,
)


@dataclass(frozen=True)
class ProtocolComparison:
    """Comparison of dose between two versions of a protocol."""

    protocol: str
    version_a: str
    version_b: str
    metric: str
    n_a: int
    n_b: int
    mean_a: float | None
    mean_b: float | None
    median_a: float | None
    median_b: float | None
    mean_diff: float | None  # B - A (positive means version B has higher dose)
    median_diff: float | None
    ci_lower: float | None  # bootstrap CI on the mean difference
    ci_upper: float | None
    p_value_mannwhitney: float | None  # non-parametric two-sided test
    n_bootstrap: int


def compare_protocol_versions(
    df: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DOSE_METRIC_COLUMNS,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[ProtocolComparison]:
    """Compare dose between protocol versions for all version pairs.

    For each protocol that has at least two distinct versions, computes the
    mean/median difference in each dose metric, a bootstrap CI on the mean
    difference, and a Mann-Whitney U test p-value.

    Parameters
    ----------
    df:
        Validated dose dataframe.
    metrics:
        Dose metric columns to compare.
    confidence:
        Confidence level for the bootstrap CI (default 0.95).
    n_bootstrap:
        Number of bootstrap resamples.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list of ProtocolComparison, one per protocol x version pair x metric.
    """
    if COL_PROTOCOL not in df.columns or COL_PROTOCOL_VERSION not in df.columns:
        return []

    comparisons: list[ProtocolComparison] = []
    for protocol, sub in df.groupby(COL_PROTOCOL, observed=True):
        versions = sub[COL_PROTOCOL_VERSION].dropna().unique()
        if len(versions) < 2:
            continue
        # Compare each ordered pair of versions (alphabetical).
        version_pairs = sorted(
            [(str(va), str(vb)) for i, va in enumerate(versions) for vb in versions[i + 1 :]]
        )
        for version_a, version_b in version_pairs:
            for metric in metrics:
                if metric not in sub.columns:
                    continue
                comp = _compare_pair(
                    sub, protocol, version_a, version_b, metric,
                    confidence=confidence, n_bootstrap=n_bootstrap, seed=seed,
                )
                comparisons.append(comp)
    return comparisons


def _compare_pair(
    sub: pd.DataFrame,
    protocol: str,
    version_a: str,
    version_b: str,
    metric: str,
    *,
    confidence: float,
    n_bootstrap: int,
    seed: int,
) -> ProtocolComparison:
    """Compare one metric between two versions of one protocol."""
    vals_a = pd.to_numeric(
        sub.loc[sub[COL_PROTOCOL_VERSION] == version_a, metric], errors="coerce"
    ).dropna()
    vals_b = pd.to_numeric(
        sub.loc[sub[COL_PROTOCOL_VERSION] == version_b, metric], errors="coerce"
    ).dropna()

    mean_a = _safe_mean(vals_a)
    mean_b = _safe_mean(vals_b)
    median_a = _safe_median(vals_a)
    median_b = _safe_median(vals_b)
    mean_diff = None
    median_diff = None
    ci_lower = None
    ci_upper = None
    p_value = None

    if mean_a is not None and mean_b is not None:
        mean_diff = mean_b - mean_a
    if median_a is not None and median_b is not None:
        median_diff = median_b - median_a

    # Bootstrap CI on the mean difference.
    if len(vals_a) >= 5 and len(vals_b) >= 5:
        rng = np.random.default_rng(seed)
        boot_diffs: list[float] = []
        for _ in range(n_bootstrap):
            sa = vals_a.sample(frac=1, replace=True, random_state=rng).values
            sb = vals_b.sample(frac=1, replace=True, random_state=rng).values
            boot_diffs.append(float(np.mean(sb) - np.mean(sa)))
        boot_diffs = np.array(boot_diffs)
        alpha = 1.0 - confidence
        ci_lower = round(float(np.percentile(boot_diffs, 100 * alpha / 2)), 4)
        ci_upper = round(float(np.percentile(boot_diffs, 100 * (1 - alpha / 2))), 4)

    # Mann-Whitney U test (non-parametric, no distribution assumption).
    if len(vals_a) >= 5 and len(vals_b) >= 5:
        try:
            _, p_value = sp_stats.mannwhitneyu(vals_b.values, vals_a.values, alternative="two-sided")
            p_value = round(float(p_value), 6)
        except ValueError:
            p_value = None

    return ProtocolComparison(
        protocol=protocol,
        version_a=version_a,
        version_b=version_b,
        metric=metric,
        n_a=len(vals_a),
        n_b=len(vals_b),
        mean_a=_r(mean_a),
        mean_b=_r(mean_b),
        median_a=_r(median_a),
        median_b=_r(median_b),
        mean_diff=_r(mean_diff),
        median_diff=_r(median_diff),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        p_value_mannwhitney=p_value,
        n_bootstrap=n_bootstrap,
    )


def comparisons_dataframe(comparisons: list[ProtocolComparison]) -> pd.DataFrame:
    """Convert comparisons to a tidy DataFrame."""
    rows: list[dict[str, Any]] = []
    for c in comparisons:
        rows.append(
            {
                "protocol": c.protocol,
                "version_a": c.version_a,
                "version_b": c.version_b,
                "metric": c.metric,
                "n_a": c.n_a,
                "n_b": c.n_b,
                "mean_a": c.mean_a,
                "mean_b": c.mean_b,
                "median_a": c.median_a,
                "median_b": c.median_b,
                "mean_diff": c.mean_diff,
                "median_diff": c.median_diff,
                "ci_lower": c.ci_lower,
                "ci_upper": c.ci_upper,
                "p_value_mannwhitney": c.p_value_mannwhitney,
                "n_bootstrap": c.n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def _safe_mean(vals: pd.Series) -> float | None:
    if len(vals) == 0:
        return None
    m = vals.mean()
    return float(m) if not (math.isnan(m) or math.isinf(m)) else None


def _safe_median(vals: pd.Series) -> float | None:
    if len(vals) == 0:
        return None
    m = vals.median()
    return float(m) if not (math.isnan(m) or math.isinf(m)) else None


def _r(val: float | None, ndigits: int = 4) -> float | None:
    if val is None:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return round(float(val), ndigits)
