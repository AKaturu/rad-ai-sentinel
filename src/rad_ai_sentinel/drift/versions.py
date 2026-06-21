"""Model-version comparison: pairwise AUROC comparison and metric deltas.

When a site updates its AI model from v1 to v2, the ACR practice parameter
requires monitoring the new version's performance against the old. This module
compares metrics across model versions present in the dataset using the DeLong
test for AUROC and simple deltas for 2x2 metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (
    COL_MODEL_VERSION,
    COL_Y_PRED_BINARY,
    COL_Y_PRED_PROBA,
    COL_Y_TRUE,
)
from ..metrics.binary import BinaryMetrics, compute_binary_metrics
from ..metrics.delong import DeLongResult, delong_test


@dataclass(frozen=True)
class VersionComparison:
    """Comparison of two model versions."""

    version_a: str
    version_b: str
    metrics_a: BinaryMetrics
    metrics_b: BinaryMetrics
    delong: DeLongResult | None = None  # None if no common test set


def compare_versions(
    df: pd.DataFrame,
    version_a: str,
    version_b: str,
    *,
    confidence: float = 0.95,
) -> VersionComparison:
    """Compare two model versions head-to-head.

    If both versions were evaluated on the same set of cases (same patient_ids
    on the same dates), the DeLong test is applied. Otherwise, only metric
    deltas are reported.

    Parameters
    ----------
    df:
        Validated dataframe.
    version_a, version_b:
        Model version labels.
    confidence:
        Confidence level for CIs.

    Returns
    -------
    VersionComparison
    """
    df_a = df[df[COL_MODEL_VERSION] == version_a]
    df_b = df[df[COL_MODEL_VERSION] == version_b]

    metrics_a = compute_binary_metrics(
        df_a[COL_Y_TRUE].values, df_a[COL_Y_PRED_BINARY].values, confidence=confidence
    )
    metrics_b = compute_binary_metrics(
        df_b[COL_Y_TRUE].values, df_b[COL_Y_PRED_BINARY].values, confidence=confidence
    )

    # Check if there are overlapping patient_ids (common test set).
    delong_result = None
    common_patients = set(df_a["patient_id"].values) & set(df_b["patient_id"].values)
    if len(common_patients) >= 20:
        # Build a paired dataset using the common patients.
        common_df_a = df_a[df_a["patient_id"].isin(common_patients)].set_index("patient_id")
        common_df_b = df_b[df_b["patient_id"].isin(common_patients)].set_index("patient_id")
        # Inner join on patient_id.
        paired = common_df_a.join(common_df_b, lsuffix="_a", rsuffix="_b", how="inner")
        if len(paired) >= 20:
            y_true = paired[COL_Y_TRUE + "_a"].values  # should be same as _b
            score_a = paired[COL_Y_PRED_PROBA + "_a"].values
            score_b = paired[COL_Y_PRED_PROBA + "_b"].values
            # Only use if labels agree.
            labels_b = paired[COL_Y_TRUE + "_b"].values
            if np.array_equal(y_true, labels_b):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    delong_result = delong_test(y_true, score_a, score_b, confidence=confidence)

    return VersionComparison(
        version_a=version_a,
        version_b=version_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        delong=delong_result,
    )


def compare_all_versions(
    df: pd.DataFrame,
    **kwargs,
) -> list[VersionComparison]:
    """Compare every pair of model versions present in the dataframe."""
    versions = sorted(df[COL_MODEL_VERSION].dropna().unique())
    comparisons = []
    for i in range(len(versions)):
        for j in range(i + 1, len(versions)):
            comparisons.append(compare_versions(df, versions[i], versions[j], **kwargs))
    return comparisons
