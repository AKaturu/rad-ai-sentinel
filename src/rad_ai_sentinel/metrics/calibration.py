"""Calibration metrics: Brier score, Expected Calibration Error, and reliability diagram data.

A well-calibrated model means its predicted probabilities faithfully reflect
the true event rate in each bin. Calibration matters in clinical AI because a
radiologist must trust the confidence score when deciding whether to act on a
finding — overconfidence causes missed diagnoses; underconfidence causes
unnecessary follow-ups.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from ..config import (
    COL_SITE,
    COL_STUDY_DATE,
    COL_Y_PRED_PROBA,
    COL_Y_TRUE,
    DEFAULT_CALIBRATION_BINS,
)
from .ci import CI, bootstrap_ci


@dataclass(frozen=True)
class CalibrationMetrics:
    """All calibration outputs for one evaluation window."""

    brier: CI  # Brier score with bootstrap CI (lower = better)
    ece: float  # Expected Calibration Error
    # Data for plotting the reliability diagram (each row = one non-empty bin).
    bin_centers: np.ndarray  # center of each bin
    prob_true: np.ndarray  # observed event rate in each bin
    prob_pred: np.ndarray  # mean predicted probability in each bin
    bin_counts: np.ndarray  # number of samples in each non-empty bin
    calibration_intercept: float  # logistic recalibration intercept
    calibration_slope: float  # logistic recalibration slope


@dataclass(frozen=True)
class CalibrationPointSummary:
    """Point calibration metrics without bootstrap intervals."""

    brier: float
    ece: float
    calibration_intercept: float
    calibration_slope: float


@dataclass(frozen=True)
class SiteCalibrationDrift:
    """Calibration comparison for one site across reference/current windows."""

    site: str
    baseline_n: int
    current_n: int
    baseline_brier: float
    current_brier: float
    brier_delta: float
    baseline_ece: float
    current_ece: float
    ece_delta: float
    baseline_intercept: float
    current_intercept: float
    intercept_delta: float
    baseline_slope: float
    current_slope: float
    slope_delta: float
    status: str


def _brier_of(paired: np.ndarray) -> float:
    """Brier score for a (y_true, y_score) paired array."""
    return float(brier_score_loss(paired[:, 0], paired[:, 1]))


def compute_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
    n_resamples: int = 500,
    strategy: str = "uniform",
) -> CalibrationMetrics:
    """Compute calibration metrics and reliability-diagram data.

    Parameters
    ----------
    y_true, y_score:
        Arrays of ground-truth labels and predicted probabilities.
    n_bins:
        Number of bins for the reliability diagram and ECE calculation.
    n_resamples:
        Bootstrap resamples for the Brier-score confidence interval.
    strategy:
        Binning strategy passed to sklearn's ``calibration_curve``; one of
        ``"uniform"`` (equal-width) or ``"quantile"`` (equal-count).

    Returns
    -------
    CalibrationMetrics
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    # --- Brier score with bootstrap CI ---
    paired = np.column_stack([y_true, y_score])
    brier_ci = bootstrap_ci(paired, _brier_of, n_resamples=n_resamples, paired=True)

    # --- Bin assignments ---
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # --- ECE: weighted average of |observed_rate - mean_predicted| per bin ---
    ece, bin_counts = _expected_calibration_error(y_true, y_score, n_bins=n_bins)

    # Bin centers for plotting.
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # sklearn's calibration_curve for the reliability diagram (returns only
    # non-empty bins in general).
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy=strategy)

    # bin_counts for the non-empty bins (match calibration_curve output length).
    # calibration_curve drops empty bins, so we report counts for non-empty ones.
    mask = bin_counts > 0
    nonempty_counts = bin_counts[mask]
    intercept, slope = _calibration_intercept_slope(y_true, y_score)

    return CalibrationMetrics(
        brier=brier_ci,
        ece=ece,
        bin_centers=bin_centers,
        prob_true=prob_true,
        prob_pred=prob_pred,
        bin_counts=nonempty_counts,
        calibration_intercept=intercept,
        calibration_slope=slope,
    )


def calibration_point_summary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> CalibrationPointSummary:
    """Compute point calibration summaries for drift tables."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    ece, _ = _expected_calibration_error(y_true, y_score, n_bins=n_bins)
    intercept, slope = _calibration_intercept_slope(y_true, y_score)
    return CalibrationPointSummary(
        brier=float(brier_score_loss(y_true, y_score)) if y_true.size else float("nan"),
        ece=ece,
        calibration_intercept=intercept,
        calibration_slope=slope,
    )


def compute_site_calibration_drift(
    df: pd.DataFrame,
    *,
    reference_fraction: float = 0.30,
    min_n: int = 30,
    n_bins: int = DEFAULT_CALIBRATION_BINS,
) -> list[SiteCalibrationDrift]:
    """Compare calibration by site between reference and current windows."""
    if COL_SITE not in df.columns:
        return []

    df_sorted = df.sort_values(COL_STUDY_DATE).reset_index(drop=True)
    split_idx = max(min(int(len(df_sorted) * reference_fraction), len(df_sorted) - 1), 1)
    baseline = df_sorted.iloc[:split_idx]
    current = df_sorted.iloc[split_idx:]
    sites = sorted(set(baseline[COL_SITE].dropna()) | set(current[COL_SITE].dropna()))

    results: list[SiteCalibrationDrift] = []
    for site in sites:
        base_site = baseline[baseline[COL_SITE] == site]
        cur_site = current[current[COL_SITE] == site]
        baseline_n = len(base_site)
        current_n = len(cur_site)
        if baseline_n < min_n or current_n < min_n:
            results.append(
                _empty_site_calibration_drift(
                    str(site),
                    baseline_n=baseline_n,
                    current_n=current_n,
                    status="insufficient_data",
                )
            )
            continue

        base = calibration_point_summary(
            base_site[COL_Y_TRUE].to_numpy(),
            base_site[COL_Y_PRED_PROBA].to_numpy(),
            n_bins=n_bins,
        )
        cur = calibration_point_summary(
            cur_site[COL_Y_TRUE].to_numpy(),
            cur_site[COL_Y_PRED_PROBA].to_numpy(),
            n_bins=n_bins,
        )
        results.append(
            SiteCalibrationDrift(
                site=str(site),
                baseline_n=baseline_n,
                current_n=current_n,
                baseline_brier=base.brier,
                current_brier=cur.brier,
                brier_delta=cur.brier - base.brier,
                baseline_ece=base.ece,
                current_ece=cur.ece,
                ece_delta=cur.ece - base.ece,
                baseline_intercept=base.calibration_intercept,
                current_intercept=cur.calibration_intercept,
                intercept_delta=cur.calibration_intercept - base.calibration_intercept,
                baseline_slope=base.calibration_slope,
                current_slope=cur.calibration_slope,
                slope_delta=cur.calibration_slope - base.calibration_slope,
                status="ok",
            )
        )
    return results


def _expected_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int,
) -> tuple[float, np.ndarray]:
    n = y_true.size
    if n == 0:
        return float("nan"), np.zeros(n_bins, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_score, bin_edges, right=True) - 1, 0, n_bins - 1)
    bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    bin_sums_true = np.bincount(bin_idx, weights=y_true, minlength=n_bins)
    bin_sums_pred = np.bincount(bin_idx, weights=y_score, minlength=n_bins)

    mask = bin_counts > 0
    observed_rate = np.divide(
        bin_sums_true,
        bin_counts,
        out=np.full(n_bins, np.nan, dtype=float),
        where=mask,
    )
    mean_predicted = np.divide(
        bin_sums_pred,
        bin_counts,
        out=np.full(n_bins, np.nan, dtype=float),
        where=mask,
    )
    valid = mask & np.isfinite(observed_rate) & np.isfinite(mean_predicted)
    ece = float(
        np.sum(bin_counts[valid] * np.abs(observed_rate[valid] - mean_predicted[valid])) / max(n, 1)
    )
    return ece, bin_counts


def _calibration_intercept_slope(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    if y_true.size < 3 or np.unique(y_true).size < 2 or np.nanstd(y_score) < 1e-12:
        return float("nan"), float("nan")

    logits = logit(np.clip(y_score, 1e-6, 1.0 - 1e-6))

    def objective(params: np.ndarray) -> float:
        intercept, slope = params
        eta = intercept + slope * logits
        loss = np.sum(np.logaddexp(0.0, eta) - y_true * eta)
        ridge = 1e-6 * float(intercept**2 + slope**2)
        return float(loss + ridge)

    result = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    if not result.success or not np.all(np.isfinite(result.x)):
        return float("nan"), float("nan")
    intercept, slope = result.x
    return float(intercept), float(slope)


def _empty_site_calibration_drift(
    site: str,
    *,
    baseline_n: int,
    current_n: int,
    status: str,
) -> SiteCalibrationDrift:
    nan = float("nan")
    return SiteCalibrationDrift(
        site=site,
        baseline_n=baseline_n,
        current_n=current_n,
        baseline_brier=nan,
        current_brier=nan,
        brier_delta=nan,
        baseline_ece=nan,
        current_ece=nan,
        ece_delta=nan,
        baseline_intercept=nan,
        current_intercept=nan,
        intercept_delta=nan,
        baseline_slope=nan,
        current_slope=nan,
        slope_delta=nan,
        status=status,
    )
