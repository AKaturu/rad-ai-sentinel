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
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from ..config import DEFAULT_CALIBRATION_BINS
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
    n = y_true.size

    # --- Brier score with bootstrap CI ---
    paired = np.column_stack([y_true, y_score])
    brier_ci = bootstrap_ci(paired, _brier_of, n_resamples=n_resamples, paired=True)

    # --- Bin assignments ---
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize returns 1..n_bins+1; we want 0..n_bins-1.
    bin_idx = np.clip(np.digitize(y_score, bin_edges, right=True) - 1, 0, n_bins - 1)

    # --- ECE: weighted average of |observed_rate - mean_predicted| per bin ---
    bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    bin_sums_true = np.bincount(bin_idx, weights=y_true, minlength=n_bins)
    bin_sums_pred = np.bincount(bin_idx, weights=y_score, minlength=n_bins)

    mask = bin_counts > 0
    observed_rate = np.where(mask, bin_sums_true / bin_counts, np.nan)
    mean_predicted = np.where(mask, bin_sums_pred / bin_counts, np.nan)

    # Only sum over non-empty bins where both values are finite.
    valid = mask & np.isfinite(observed_rate) & np.isfinite(mean_predicted)
    ece = float(
        np.sum(bin_counts[valid] * np.abs(observed_rate[valid] - mean_predicted[valid])) / max(n, 1)
    )

    # Bin centers for plotting.
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # sklearn's calibration_curve for the reliability diagram (returns only
    # non-empty bins in general).
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy=strategy)

    # bin_counts for the non-empty bins (match calibration_curve output length).
    # calibration_curve drops empty bins, so we report counts for non-empty ones.
    nonempty_counts = bin_counts[mask]

    return CalibrationMetrics(
        brier=brier_ci,
        ece=ece,
        bin_centers=bin_centers,
        prob_true=prob_true,
        prob_pred=prob_pred,
        bin_counts=nonempty_counts,
    )
