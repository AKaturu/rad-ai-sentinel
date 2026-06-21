"""Drift detection, stop-rule alerts, and model-version comparison.

This package implements the post-deployment surveillance layer that
operationalizes the ACR-SIIM Practice Parameter's requirements to "monitor
real-world model performance for drift and safety issues and define stop rules."
"""

from __future__ import annotations

from .alerts import Alert, AlertReport, check_alerts
from .detection import (
    CUSUMResult,
    DriftResult,
    RollingAUCResult,
    compute_drift,
    cusum,
    kl_divergence,
    psi,
    rolling_auroc,
)
from .missing import (
    ColumnMissingness,
    MissingDataReport,
    SubgroupAvailability,
    analyze_missing_data,
)
from .versions import VersionComparison, compare_all_versions, compare_versions

__all__ = [  # noqa: RUF022 - grouped by subpackage for readability
    # alerts
    "Alert",
    "AlertReport",
    "check_alerts",
    # detection
    "CUSUMResult",
    "DriftResult",
    "RollingAUCResult",
    "compute_drift",
    "cusum",
    "kl_divergence",
    "psi",
    "rolling_auroc",
    # missing
    "ColumnMissingness",
    "MissingDataReport",
    "SubgroupAvailability",
    "analyze_missing_data",
    # versions
    "VersionComparison",
    "compare_all_versions",
    "compare_versions",
]
