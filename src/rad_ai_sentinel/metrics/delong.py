"""In-house DeLong test for comparing two AUROCs on the same test set.

The DeLong method estimates the variance of the AUROC using placement values
(U-statistic theory), then tests the null hypothesis that two correlated
models have equal discrimination on a *common* set of cases.

Reference: DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).
Comparing the Areas Under Two or More Correlated Receiver Operating
Characteristic Curves: A Nonparametric Approach. *Biometrics*, 44(3), 837-845.

Implementation follows Sun & Xu (2014) fast O(n log n) algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class DeLongResult:
    """Result of comparing two AUROCs on the same test set."""

    auc1: float
    auc2: float
    auc_diff: float  # auc1 - auc2
    se_diff: float  # standard error of the difference
    z_stat: float  # z-statistic under H0: auc1 == auc2
    p_value: float  # two-sided p-value
    ci_diff_lower: float  # 95% CI for the difference
    ci_diff_upper: float


def _compute_auc_phats(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Compute placement values (phats) for the DeLong variance estimate.

    For each sample i, phat_i = (number of positive samples with score >= score_i)
    / (total positives) - (number of negative samples with score > score_i)
    / (total negatives).

    This is the O(n log n) algorithm: sort once, use cumulative sums.
    """
    n = len(y_true)
    # Sort by score descending; for ties, put negatives before positives
    # (stable sort preserves original order for equal scores).
    # lexsort sorts by the last key first: (-score descending), then y_true
    # (negatives before positives among ties).
    order = np.lexsort((y_true, -y_score))
    sorted_y = y_true[order]

    pos_mask = sorted_y == 1
    neg_mask = sorted_y == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    # Cumulative counts of positives at or above rank i.
    cum_pos = np.cumsum(pos_mask.astype(float))
    # Cumulative counts of negatives STRICTLY above rank i.
    # At rank i: negatives strictly above = cum_neg up to i-1.
    cum_neg_strict = np.cumsum(neg_mask.astype(float)) - neg_mask.astype(float)

    # phat_i = cum_pos[i] / n_pos - cum_neg_strict[i] / n_neg
    phats = np.where(
        (n_pos > 0) & (n_neg > 0),
        cum_pos / n_pos - cum_neg_strict / n_neg,
        0.0,
    )

    # Undo the sort.
    phats_unsorted = np.empty(n, dtype=float)
    phats_unsorted[order] = phats
    return phats_unsorted


def _delong_variance(y_true: np.ndarray, phats: np.ndarray) -> float:
    """Estimate the variance of AUROC using the placement values.

    Var(AUC) = S10 / (n_pos * n_neg) where S10 is the variance of
    the placement values among the positive class.
    """
    pos_mask = y_true == 1
    n_pos = int(pos_mask.sum())
    n_neg = int((~pos_mask).sum())

    if n_pos < 2 or n_neg < 2:
        return 0.0

    phats_pos = phats[pos_mask]
    # S10: variance of phat among positive samples, scaled
    s10 = np.var(phats_pos, ddof=1)
    return s10 / n_neg


def _delong_covariance(y_true: np.ndarray, phats1: np.ndarray, phats2: np.ndarray) -> float:
    """Covariance of AUROCs from two models on the same test set."""
    pos_mask = y_true == 1
    n_neg = int((~pos_mask).sum())

    if n_neg < 2:
        return 0.0

    phats1_pos = phats1[pos_mask]
    phats2_pos = phats2[pos_mask]

    # Covariance of placement values among positive class
    mean1 = phats1_pos.mean()
    mean2 = phats2_pos.mean()
    cov = np.mean((phats1_pos - mean1) * (phats2_pos - mean2))
    return cov / n_neg


def delong_test(
    y_true: np.ndarray,
    y_score1: np.ndarray,
    y_score2: np.ndarray,
    confidence: float = 0.95,
) -> DeLongResult:
    """DeLong test for AUROC difference on a common test set.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0/1).
    y_score1, y_score2:
        Predicted probabilities from two different models or versions.
    confidence:
        Confidence level for the CI of the AUROC difference.

    Returns
    -------
    DeLongResult
    """
    y_true = np.asarray(y_true, dtype=int)
    y_score1 = np.asarray(y_score1, dtype=float)
    y_score2 = np.asarray(y_score2, dtype=float)

    from sklearn.metrics import roc_auc_score

    auc1 = float(roc_auc_score(y_true, y_score1))
    auc2 = float(roc_auc_score(y_true, y_score2))

    phats1 = _compute_auc_phats(y_true, y_score1)
    phats2 = _compute_auc_phats(y_true, y_score2)

    var1 = _delong_variance(y_true, phats1)
    var2 = _delong_variance(y_true, phats2)
    cov12 = _delong_covariance(y_true, phats1, phats2)

    var_diff = var1 + var2 - 2.0 * cov12
    se_diff = float(np.sqrt(max(var_diff, 1e-15)))

    auc_diff = auc1 - auc2
    z_stat = auc_diff / se_diff
    p_value = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))

    z_crit = float(norm.ppf(0.5 + confidence / 2.0))
    ci_lower = auc_diff - z_crit * se_diff
    ci_upper = auc_diff + z_crit * se_diff

    return DeLongResult(
        auc1=auc1,
        auc2=auc2,
        auc_diff=auc_diff,
        se_diff=se_diff,
        z_stat=z_stat,
        p_value=p_value,
        ci_diff_lower=ci_lower,
        ci_diff_upper=ci_upper,
    )
