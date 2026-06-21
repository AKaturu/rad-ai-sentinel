"""Temporal drift detection for radiology AI performance monitoring.

Implements four complementary drift signals, plus a rolling-AUROC tracker:
  * Population Stability Index (PSI) — distribution shift in predicted scores
  * KL divergence — information-theoretic distance between reference and current
  * CUSUM — cumulative sum signal for sustained performance degradation
  * Rolling AUROC — time-windowed discrimination tracking

All are computed over a sliding time window against a reference (baseline) window.
The PSI heuristics (>0.1 minor, >0.25 major) follow the widely-cited industry
convention and are the thresholds recommended by the ACR Assess-AI framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import entropy

from ..config import (
    COL_STUDY_DATE,
    COL_Y_PRED_PROBA,
    COL_Y_TRUE,
    DEFAULT_DRIFT_PSI_MAJOR,
    DEFAULT_DRIFT_PSI_MINOR,
    DEFAULT_ROLLING_WINDOW_DAYS,
)


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between reference and current distributions.

    PSI = sum_over_bins ((%_current - %_reference) * ln(%_current / %_reference))

    Heuristic thresholds (widely cited):
        PSI < 0.1  : no significant drift
        0.1 <= PSI < 0.25 : minor drift — monitor closely
        PSI >= 0.25 : major drift — investigate / alert

    Parameters
    ----------
    reference, current:
        1-D arrays of scores (e.g. predicted probabilities). Binned into
        ``n_bins`` equal-width bins spanning [0, 1].
    n_bins:
        Number of bins for the score histogram.

    Returns
    -------
    float
        PSI value (>= 0).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ref_counts = np.histogram(reference, bins=bins)[0].astype(float)
    cur_counts = np.histogram(current, bins=bins)[0].astype(float)

    # Replace zero counts with a small floor to avoid log(0).
    eps = 1e-6
    ref_pct = np.maximum(ref_counts / max(ref_counts.sum(), 1), eps)
    cur_pct = np.maximum(cur_counts / max(cur_counts.sum(), 1), eps)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def kl_divergence(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """KL(reference || current): divergence of current from reference.

    Uses scipy.stats.entropy on the binned distributions. Returns 0 when
    distributions are identical.

    Parameters
    ----------
    reference, current:
        1-D arrays of scores.
    n_bins:
        Number of equal-width bins in [0, 1].
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ref_counts = np.histogram(reference, bins=bins)[0].astype(float)
    cur_counts = np.histogram(current, bins=bins)[0].astype(float)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    # Add small floor to avoid log(0) in entropy().
    eps = 1e-10
    ref_pct = ref_pct + eps
    cur_pct = cur_pct + eps
    # Renormalize.
    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    return float(entropy(ref_pct, cur_pct))


@dataclass(frozen=True)
class CUSUMResult:
    """CUSUM signal over a time series.

    Attributes
    ----------
    s_high:
        Cumulative sum tracking upward shifts (performance getting worse).
    s_low:
        Cumulative sum tracking downward shifts (performance improving).
    detections:
        Boolean mask where |s_high| exceeds the threshold.
    """

    s_high: np.ndarray
    s_low: np.ndarray
    detections: np.ndarray
    threshold: float


def cusum(
    values: np.ndarray,
    threshold: float = 4.0,
    allow_neg: bool = True,
) -> CUSUMResult:
    """Two-sided CUSUM control chart for detecting sustained shifts.

    Parameters
    ----------
    values:
        1-D time series of performance values (e.g. daily AUROC).
    threshold:
        Control limit; signal fires when |S| > threshold.
    allow_neg:
        Whether to track negative shifts (improvements) as well.

    Returns
    -------
    CUSUMResult
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return CUSUMResult(
            s_high=np.array([], dtype=float),
            s_low=np.array([], dtype=float),
            detections=np.array([], dtype=bool),
            threshold=threshold,
        )

    target = values.mean()
    s_high = np.zeros(n, dtype=float)
    s_low = np.zeros(n, dtype=float)

    for i in range(1, n):
        s_high[i] = max(0.0, s_high[i - 1] + (values[i] - target))
        if allow_neg:
            s_low[i] = min(0.0, s_low[i - 1] + (values[i] - target))

    detections = np.abs(s_high) > threshold
    return CUSUMResult(s_high=s_high, s_low=s_low, detections=detections, threshold=threshold)


@dataclass(frozen=True)
class RollingAUCResult:
    """Rolling-window AUROC over time.

    Attributes
    ----------
    dates:
        Center date of each window.
    auroc:
        AUROC in each window (NaN for windows with insufficient data or
        only one class present).
    window_size_days:
        Size of the sliding window.
    """

    dates: np.ndarray
    auroc: np.ndarray
    window_size_days: int


def rolling_auroc(
    df: pd.DataFrame,
    window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    min_samples: int = 20,
) -> RollingAUCResult:
    """Compute AUROC in consecutive time windows.

    Parameters
    ----------
    df:
        Validated dataframe with study_date, y_true, y_pred_proba columns,
        sorted ascending by study_date.
    window_days:
        Width of each sliding window in days.
    min_samples:
        Minimum number of samples in a window to compute AUROC.

    Returns
    -------
    RollingAUCResult
    """
    from sklearn.metrics import roc_auc_score

    # Coerce study_date to datetime defensively (in case caller skipped validation).
    dates_col = pd.to_datetime(df[COL_STUDY_DATE])
    min_date = dates_col.min()
    max_date = dates_col.max()
    n_windows = max(1, int((max_date - min_date).days // (window_days // 2)) + 1)

    auroc_values = []
    date_values = []

    for i in range(n_windows):
        window_start = min_date + pd.Timedelta(days=i * window_days // 2)
        window_end = window_start + pd.Timedelta(days=window_days)
        window_df = df[(dates_col >= window_start) & (dates_col < window_end)]

        if len(window_df) < min_samples:
            auroc_values.append(float("nan"))
        else:
            yt = window_df[COL_Y_TRUE].values
            yp = window_df[COL_Y_PRED_PROBA].values
            unique_labels = np.unique(yt)
            if len(unique_labels) < 2:
                auroc_values.append(float("nan"))
            else:
                try:
                    auroc_values.append(float(roc_auc_score(yt, yp)))
                except ValueError:
                    auroc_values.append(float("nan"))

        date_values.append(window_start + pd.Timedelta(days=window_days // 2))

    return RollingAUCResult(
        dates=np.array(date_values),
        auroc=np.array(auroc_values),
        window_size_days=window_days,
    )


@dataclass(frozen=True)
class DriftResult:
    """Complete drift analysis for a dataset.

    Attributes
    ----------
    psi_value:
        Population Stability Index (reference window vs full dataset).
    psi_level:
        'none', 'minor', or 'major'.
    kl_value:
        KL divergence between reference and current.
    rolling:
        Rolling AUROC over time.
    cusum:
        CUSUM signal on the rolling AUROC series.
    """

    psi_value: float
    psi_level: str
    kl_value: float
    rolling: RollingAUCResult
    cusum: CUSUMResult


def compute_drift(
    df: pd.DataFrame,
    *,
    reference_fraction: float = 0.3,
    n_bins: int = 10,
    psi_minor: float = DEFAULT_DRIFT_PSI_MINOR,
    psi_major: float = DEFAULT_DRIFT_PSI_MAJOR,
    rolling_window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
) -> DriftResult:
    """Run the full drift analysis pipeline.

    Splits the data into a reference window (first ``reference_fraction`` of
    studies by date) and a current window (the rest), then computes PSI, KL,
    rolling AUROC, and CUSUM.

    Parameters
    ----------
    df:
        Validated, sorted-by-date dataframe.
    reference_fraction:
        Fraction of the earliest studies to use as the reference window.
    n_bins:
        Number of bins for PSI/KL histograms.
    psi_minor, psi_major:
        PSI thresholds for drift classification.
    rolling_window_days:
        Window size for rolling AUROC.

    Returns
    -------
    DriftResult
    """
    df_sorted = df.sort_values(COL_STUDY_DATE)
    n = len(df_sorted)
    ref_n = max(int(n * reference_fraction), 10)
    ref_df = df_sorted.iloc[:ref_n]
    cur_df = df_sorted.iloc[ref_n:]

    ref_scores = ref_df[COL_Y_PRED_PROBA].values
    cur_scores = cur_df[COL_Y_PRED_PROBA].values

    psi_val = psi(ref_scores, cur_scores, n_bins=n_bins)
    kl_val = kl_divergence(ref_scores, cur_scores, n_bins=n_bins)

    if psi_val >= psi_major:
        psi_level = "major"
    elif psi_val >= psi_minor:
        psi_level = "minor"
    else:
        psi_level = "none"

    rolling = rolling_auroc(df_sorted, window_days=rolling_window_days)

    # CUSUM on the non-NaN rolling AUROC values.
    valid_mask = ~np.isnan(rolling.auroc)
    if valid_mask.sum() >= 5:
        cusum_res = cusum(rolling.auroc[valid_mask])
    else:
        cusum_res = CUSUMResult(
            s_high=np.array([], dtype=float),
            s_low=np.array([], dtype=float),
            detections=np.array([], dtype=bool),
            threshold=4.0,
        )

    return DriftResult(
        psi_value=psi_val,
        psi_level=psi_level,
        kl_value=kl_val,
        rolling=rolling,
        cusum=cusum_res,
    )
