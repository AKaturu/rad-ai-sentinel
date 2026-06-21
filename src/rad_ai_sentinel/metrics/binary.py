"""Binary (2x2) discrimination metrics: the confusion-matrix family.

All metrics are derived from the four confusion-matrix counts (TP, FP, TN, FN)
and reported with Wilson-score confidence intervals, which have correct coverage
even for rare outcomes — important for small radiology subgroups.

Definitions follow the standard clinical-AI conventions:
    sensitivity (recall, TPR) = TP / (TP + FN)
    specificity (TNR, TNR)     = TN / (TN + FP)
    PPV (precision)             = TP / (TP + FP)
    NPV                         = TN / (TN + FN)
    accuracy                    = (TP + TN) / N
    F1                          = 2*PPV*sens / (PPV + sens)
    prevalence                  = (TP + FN) / N
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import COL_Y_PRED_BINARY, COL_Y_TRUE, DEFAULT_CONFIDENCE_LEVEL
from .ci import CI, proportion_ci


@dataclass(frozen=True)
class ConfusionCounts:
    """The four cells of a 2x2 confusion matrix plus totals."""

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def positives(self) -> int:
        """Actual positives (P): TP + FN."""
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        """Actual negatives (N): TN + FP."""
        return self.tn + self.fp

    @property
    def predicted_positive(self) -> int:
        return self.tp + self.fp

    @property
    def predicted_negative(self) -> int:
        return self.tn + self.fn


def confusion_counts(y_true: Iterable[int], y_pred: Iterable[int]) -> ConfusionCounts:
    """Compute the 2x2 counts from ground-truth and predicted binary labels."""
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; "
            f"got {y_true_arr.shape} and {y_pred_arr.shape}"
        )

    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    return ConfusionCounts(tp=tp, fp=fp, tn=tn, fn=fn)


@dataclass(frozen=True)
class BinaryMetrics:
    """The full 2x2 metric family, each with a Wilson CI where defined."""

    sensitivity: CI
    specificity: CI
    ppv: CI
    npv: CI
    accuracy: CI
    f1: float
    prevalence: float
    counts: ConfusionCounts

    def as_dict(self) -> dict[str, object]:
        """Flat dict of point estimates (handy for tables/reports)."""
        return {
            "sensitivity": self.sensitivity.estimate,
            "specificity": self.specificity.estimate,
            "ppv": self.ppv.estimate,
            "npv": self.npv.estimate,
            "accuracy": self.accuracy.estimate,
            "f1": self.f1,
            "prevalence": self.prevalence,
            "tp": self.counts.tp,
            "fp": self.counts.fp,
            "tn": self.counts.tn,
            "fn": self.counts.fn,
        }


def _safe_proportion(successes: int, n: int, confidence: float) -> CI:
    """Proportion CI that degrades gracefully when the denominator is 0."""
    if n == 0:
        return CI(estimate=float("nan"), lower=float("nan"), upper=float("nan"))
    return proportion_ci(successes, n, confidence)


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else float("nan")


def compute_binary_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
) -> BinaryMetrics:
    """Compute the full 2x2 metric family with Wilson confidence intervals.

    Parameters
    ----------
    y_true, y_pred:
        Ground-truth and predicted binary labels (0/1).
    confidence:
        Confidence level for the intervals, e.g. 0.95.

    Returns
    -------
    BinaryMetrics
    """
    counts = confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = counts.tp, counts.fp, counts.tn, counts.fn
    n = counts.n

    sens = _safe_proportion(tp, counts.positives, confidence)  # TP / (TP+FN)
    spec = _safe_proportion(tn, counts.negatives, confidence)  # TN / (TN+FP)
    ppv = _safe_proportion(tp, counts.predicted_positive, confidence)  # TP/(TP+FP)
    npv = _safe_proportion(tn, counts.predicted_negative, confidence)  # TN/(TN+FN)
    acc = _safe_proportion(tp + tn, n, confidence)

    # F1 has no clean closed-form CI here; report the point estimate only
    # (its CI is obtained via bootstrap in the curves module if needed).
    f1_point = _safe_ratio(2 * tp, 2 * tp + fp + fn)

    prevalence = _safe_ratio(counts.positives, n)

    return BinaryMetrics(
        sensitivity=sens,
        specificity=spec,
        ppv=ppv,
        npv=npv,
        accuracy=acc,
        f1=f1_point,
        prevalence=prevalence,
        counts=counts,
    )


def binary_metrics_from_df(
    df: pd.DataFrame,
    *,
    y_true_col: str = COL_Y_TRUE,
    y_pred_col: str = COL_Y_PRED_BINARY,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
) -> BinaryMetrics:
    """Convenience entry point: compute 2x2 metrics directly from a dataframe."""
    return compute_binary_metrics(df[y_true_col], df[y_pred_col], confidence=confidence)
