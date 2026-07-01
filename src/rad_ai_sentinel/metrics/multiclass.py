"""Label-based multi-class monitoring metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ..config import COL_Y_PRED_LABEL, COL_Y_TRUE


@dataclass(frozen=True)
class MulticlassClassMetrics:
    """One-vs-rest metrics for a single class label."""

    label: str
    support: int
    predicted: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    f1: float


@dataclass(frozen=True)
class MulticlassMetrics:
    """Aggregate and per-class metrics for a multi-class model."""

    labels: tuple[str, ...]
    n: int
    accuracy: float
    balanced_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    cohen_kappa: float
    per_class: tuple[MulticlassClassMetrics, ...]
    confusion_matrix: tuple[tuple[int, ...], ...]

    def as_dict(self) -> dict[str, Any]:
        """JSON-serializable summary for CLI and audit workflows."""
        return {
            "n": self.n,
            "labels": list(self.labels),
            "accuracy": _json_value(self.accuracy),
            "balanced_accuracy": _json_value(self.balanced_accuracy),
            "macro_precision": _json_value(self.macro_precision),
            "macro_recall": _json_value(self.macro_recall),
            "macro_f1": _json_value(self.macro_f1),
            "weighted_precision": _json_value(self.weighted_precision),
            "weighted_recall": _json_value(self.weighted_recall),
            "weighted_f1": _json_value(self.weighted_f1),
            "cohen_kappa": _json_value(self.cohen_kappa),
            "per_class": [
                {key: _json_value(value) for key, value in vars(item).items()}
                for item in self.per_class
            ],
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
        }


def compute_multiclass_metrics(
    y_true: Iterable[object],
    y_pred: Iterable[object],
) -> MulticlassMetrics:
    """Compute label-based multi-class summary metrics.

    The metrics intentionally avoid probability-dependent quantities so sites
    can monitor classifiers that emit only final class labels. Calibration and
    one-vs-rest AUROC can be added later when a probability-column convention is
    standardized.
    """
    true = _as_label_array(y_true)
    pred = _as_label_array(y_pred)
    if true.shape != pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape; got {true.shape} and {pred.shape}")
    if len(true) == 0:
        raise ValueError("Multi-class metrics require at least one row")

    labels = tuple(sorted({*true.tolist(), *pred.tolist()}))
    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        pred,
        labels=list(labels),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true,
        pred,
        labels=list(labels),
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true,
        pred,
        labels=list(labels),
        average="weighted",
        zero_division=0,
    )

    per_class = tuple(
        _class_metrics(
            label=label,
            true=true,
            pred=pred,
            support=int(support[idx]),
            ppv=float(precision[idx]),
            sensitivity=float(recall[idx]),
            f1=float(f1[idx]),
        )
        for idx, label in enumerate(labels)
    )
    matrix = confusion_matrix(true, pred, labels=list(labels))

    return MulticlassMetrics(
        labels=labels,
        n=len(true),
        accuracy=float(accuracy_score(true, pred)),
        balanced_accuracy=_balanced_accuracy(per_class),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        weighted_precision=float(weighted_precision),
        weighted_recall=float(weighted_recall),
        weighted_f1=float(weighted_f1),
        cohen_kappa=float(cohen_kappa_score(true, pred, labels=list(labels))),
        per_class=per_class,
        confusion_matrix=tuple(tuple(int(value) for value in row) for row in matrix),
    )


def multiclass_metrics_from_df(
    df: pd.DataFrame,
    *,
    y_true_col: str = COL_Y_TRUE,
    y_pred_col: str = COL_Y_PRED_LABEL,
) -> MulticlassMetrics:
    """Compute multi-class metrics directly from a validated dataframe."""
    return compute_multiclass_metrics(df[y_true_col], df[y_pred_col])


def multiclass_summary_frame(metrics: MulticlassMetrics) -> pd.DataFrame:
    """Top-line multi-class metrics as a tidy table."""
    rows = [
        ("Accuracy", metrics.accuracy),
        ("Balanced accuracy", metrics.balanced_accuracy),
        ("Macro precision", metrics.macro_precision),
        ("Macro recall", metrics.macro_recall),
        ("Macro F1", metrics.macro_f1),
        ("Weighted precision", metrics.weighted_precision),
        ("Weighted recall", metrics.weighted_recall),
        ("Weighted F1", metrics.weighted_f1),
        ("Cohen kappa", metrics.cohen_kappa),
    ]
    return pd.DataFrame(
        [{"metric": name, "estimate": _round(value)} for name, value in rows]
    )


def multiclass_per_class_frame(metrics: MulticlassMetrics) -> pd.DataFrame:
    """One-vs-rest per-class metrics as a dataframe."""
    return pd.DataFrame(
        [
            {
                "class_label": item.label,
                "support": item.support,
                "predicted": item.predicted,
                "sensitivity": _round(item.sensitivity),
                "specificity": _round(item.specificity),
                "ppv": _round(item.ppv),
                "npv": _round(item.npv),
                "f1": _round(item.f1),
                "tp": item.true_positive,
                "fp": item.false_positive,
                "tn": item.true_negative,
                "fn": item.false_negative,
            }
            for item in metrics.per_class
        ]
    )


def multiclass_confusion_frame(metrics: MulticlassMetrics) -> pd.DataFrame:
    """Confusion matrix with actual labels as rows and predicted labels as columns."""
    rows = []
    for actual, values in zip(metrics.labels, metrics.confusion_matrix, strict=True):
        row: dict[str, object] = {"actual_label": actual}
        row.update({f"predicted_{label}": value for label, value in zip(metrics.labels, values, strict=True)})
        rows.append(row)
    return pd.DataFrame(rows)


def _class_metrics(
    *,
    label: str,
    true: np.ndarray,
    pred: np.ndarray,
    support: int,
    ppv: float,
    sensitivity: float,
    f1: float,
) -> MulticlassClassMetrics:
    true_is_label = true == label
    pred_is_label = pred == label
    tp = int(np.sum(true_is_label & pred_is_label))
    fp = int(np.sum(~true_is_label & pred_is_label))
    fn = int(np.sum(true_is_label & ~pred_is_label))
    tn = int(np.sum(~true_is_label & ~pred_is_label))
    return MulticlassClassMetrics(
        label=label,
        support=support,
        predicted=int(np.sum(pred_is_label)),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        sensitivity=sensitivity,
        specificity=_ratio(tn, tn + fp),
        ppv=ppv,
        npv=_ratio(tn, tn + fn),
        f1=f1,
    )


def _as_label_array(values: Iterable[object]) -> np.ndarray:
    return pd.Series(list(values), dtype="string").str.strip().astype(str).to_numpy()


def _ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else float("nan")


def _round(value: float) -> float | None:
    if np.isnan(value) or np.isinf(value):
        return None
    return round(float(value), 4)


def _balanced_accuracy(per_class: tuple[MulticlassClassMetrics, ...]) -> float:
    observed_recalls = [
        item.sensitivity
        for item in per_class
        if item.support > 0 and not np.isnan(item.sensitivity)
    ]
    return float(np.mean(observed_recalls)) if observed_recalls else float("nan")


def _json_value(value: object) -> object:
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value
