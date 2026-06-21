"""Confidence-interval primitives.

Two complementary methods (chosen by metric type):
  * Wilson score interval — closed-form, for 2x2 proportion metrics
    (sensitivity, specificity, PPV, NPV). Correct coverage even near 0%/100%,
    which matters for rare-event radiology subgroups.
  * BCa (bias-corrected accelerated) bootstrap — for score-based / curve-based
    metrics (AUROC, AUPRC, Brier) where no closed form exists.
  * DeLong covariance — for pairwise AUROC comparison on a common test set.

All functions return ``(lower, upper)`` tuples at the requested confidence level.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from ..config import DEFAULT_BOOTSTRAP_N, DEFAULT_CONFIDENCE_LEVEL, DEFAULT_RANDOM_SEED


class CI(NamedTuple):
    """A point estimate with its (lower, upper) confidence interval."""

    estimate: float
    lower: float
    upper: float

    def __repr__(self) -> str:
        return f"{self.estimate:.4f} [{self.lower:.4f}, {self.upper:.4f}]"


def _z_score(confidence: float) -> float:
    """Two-sided z multiplier for a given confidence level (e.g. 0.95 -> 1.96)."""
    # 0.382593... = 0.5 * sqrt(2/pi); inverse via the probit approximation is
    # messy; use the standard normal PPF through scipy for correctness instead.
    from scipy.stats import norm

    return float(norm.ppf(0.5 + confidence / 2.0))


def wilson_ci(
    successes: int, n: int, confidence: float = DEFAULT_CONFIDENCE_LEVEL
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Parameters
    ----------
    successes:
        Number of "good" outcomes (e.g. true positives, for sensitivity).
    n:
        Total trials in the denominator (e.g. TP+FN for sensitivity).
    confidence:
        Confidence level in (0, 1), e.g. 0.95.

    Returns
    -------
    (lower, upper): tuple[float, float]

    Raises
    ------
    ValueError
        If inputs are invalid (n<=0, successes<0, successes>n).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if successes < 0:
        raise ValueError(f"successes must be non-negative, got {successes}")
    if successes > n:
        raise ValueError(f"successes ({successes}) cannot exceed n ({n})")

    z = _z_score(confidence)
    z2 = z * z
    p = successes / n
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return centre - half, centre + half


def proportion_ci(successes: int, n: int, confidence: float = DEFAULT_CONFIDENCE_LEVEL) -> CI:
    """Convenience: Wilson CI wrapped as a :class:`CI` with the point estimate."""
    point = successes / n
    lo, hi = wilson_ci(successes, n, confidence)
    return CI(estimate=point, lower=lo, upper=hi)


def bootstrap_ci(
    samples: np.ndarray,
    statistic: Callable[..., float],
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_RANDOM_SEED,
    *,
    paired: bool = False,
) -> CI:
    """BCa bootstrap confidence interval for a statistic of ``samples``.

    Parameters
    ----------
    samples:
        1-D array for a single-sample statistic, or 2-D array of shape (n, k)
        when ``paired=True`` (e.g. (y_true, y_score) stacked column-wise).
        Rows are resampled together so paired observations stay linked.
    statistic:
        Callable receiving an array (1-D or 2-D) and returning a scalar float.
    confidence, n_resamples, seed:
        Standard bootstrap controls.
    paired:
        Whether ``samples`` is 2-D and rows should be resampled together.

    Returns
    -------
    CI
        Point estimate = ``statistic(samples)``; interval from percentile method
        on bootstrap replicates. (BCa via scipy has compatibility issues with
        paired 2-D inputs across versions; we implement manual resampling with
        the bias-corrected percentile adjustment for stability.)
    """
    samples = np.asarray(samples)
    rng = np.random.default_rng(seed)

    point = float(statistic(samples))

    n = samples.shape[0]
    if n < 2 or np.all(samples == samples[0]):
        return CI(estimate=point, lower=point, upper=point)

    # Manual row-wise bootstrap resampling (keeps paired rows together).
    indices = rng.integers(0, n, size=(n_resamples, n))
    replicates = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        resampled = samples[indices[i]]
        replicates[i] = float(statistic(resampled))

    replicates = np.sort(replicates)

    # BCa bias correction.
    # z0 = Phi^{-1}(proportion of replicates < point estimate)
    prop_below = np.mean(replicates < point)
    if prop_below == 0.0:
        prop_below = 1.0 / (2 * n_resamples)
    elif prop_below == 1.0:
        prop_below = 1.0 - 1.0 / (2 * n_resamples)
    from scipy.stats import norm

    z0 = norm.ppf(prop_below)

    # Acceleration: estimated via jackknife (for the point estimate).
    jackknife_estimates = np.empty(n, dtype=float)
    for i in range(n):
        left = samples[:i]
        right = samples[i + 1 :]
        jack_idx = np.concatenate([left, right])
        jackknife_estimates[i] = float(statistic(jack_idx))

    jack_mean = jackknife_estimates.mean()
    # a = (sum((jack_mean - jack_i)^3)) / (6 * (sum((jack_mean - jack_i)^2))^1.5)
    diffs = jack_mean - jackknife_estimates
    a_num = np.sum(diffs**3)
    a_den = 6.0 * (np.sum(diffs**2) ** 1.5)
    a = a_num / a_den if a_den != 0 else 0.0

    # BCa quantiles.
    z_alpha_lo = norm.ppf((1.0 - confidence) / 2.0)
    z_alpha_hi = norm.ppf(1.0 - (1.0 - confidence) / 2.0)

    def _bca_percentile(z_alpha: float) -> float:
        z_bc = z0 + (z0 + z_alpha) / (1.0 - a * (z0 + z_alpha))
        return float(norm.cdf(z_bc))

    alpha1 = np.clip(_bca_percentile(z_alpha_lo), 1e-10, 1.0 - 1e-10)
    alpha2 = np.clip(_bca_percentile(z_alpha_hi), 1e-10, 1.0 - 1e-10)

    lower = float(np.percentile(replicates, alpha1 * 100))
    upper = float(np.percentile(replicates, alpha2 * 100))

    return CI(estimate=point, lower=lower, upper=upper)
