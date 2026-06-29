"""Discrimination curves: ROC and precision-recall, with bootstrap CIs.

AUROC and AUPRC summarise discrimination across all operating thresholds. Both
get BCa bootstrap confidence intervals because no closed-form interval exists.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ..config import DEFAULT_BOOTSTRAP_N, DEFAULT_CONFIDENCE_LEVEL, DEFAULT_RANDOM_SEED
from .ci import CI, bootstrap_ci


@dataclass(frozen=True)
class CurveResult:
    """A computed curve plus its area and bootstrap CI."""

    # x and y coordinates of the curve (monotonic thresholds aside).
    x: np.ndarray
    y: np.ndarray
    thresholds: np.ndarray
    area: CI  # AUROC or AUPRC with its bootstrap CI


def _paired(y_true: Iterable[int], y_score: Iterable[float]) -> np.ndarray:
    """Stack (y_true, y_score) into an (n, 2) float array for paired resampling."""
    yt = np.asarray(list(y_true), dtype=float)
    ys = np.asarray(list(y_score), dtype=float)
    if yt.shape != ys.shape:
        raise ValueError(
            f"y_true and y_score must have the same shape; got {yt.shape} and {ys.shape}"
        )
    return np.column_stack([yt, ys])


def _auroc_of(paired: np.ndarray) -> float:
    return float(roc_auc_score(paired[:, 0], paired[:, 1]))


def _auprc_of(paired: np.ndarray) -> float:
    return float(average_precision_score(paired[:, 0], paired[:, 1]))


def compute_roc(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_RANDOM_SEED,
) -> CurveResult:
    """ROC curve + AUROC with a BCa bootstrap CI.

    Bootstrap resamples keep (label, score) pairs together so the induced
    label distribution in each resample mirrors the original.
    """
    paired = _paired(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(paired[:, 0], paired[:, 1])
    ci = bootstrap_ci(
        paired,
        _auroc_of,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
        paired=True,
    )
    return CurveResult(x=fpr, y=tpr, thresholds=thresholds, area=ci)


def compute_pr(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    confidence: float = DEFAULT_CONFIDENCE_LEVEL,
    n_resamples: int = DEFAULT_BOOTSTRAP_N,
    seed: int = DEFAULT_RANDOM_SEED,
) -> CurveResult:
    """Precision-recall curve + AUPRC with a BCa bootstrap CI."""
    paired = _paired(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(paired[:, 0], paired[:, 1])
    ci = bootstrap_ci(
        paired,
        _auprc_of,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
        paired=True,
    )
    # precision_recall_curve returns precision/recall without a threshold for
    # the first point; pad thresholds to match length for downstream plotting.
    if len(thresholds) < len(precision):
        thresholds = np.pad(thresholds, (1, 0), constant_values=thresholds[0] if len(thresholds) else 1.0)
    assert len(thresholds) == len(precision), f"thresholds {len(thresholds)} != precision {len(precision)}"
    return CurveResult(x=recall, y=precision, thresholds=thresholds, area=ci)
