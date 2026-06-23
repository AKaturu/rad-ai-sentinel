"""Analytics: grouping, missing-dose detection, outliers, comparisons, trends."""

from __future__ import annotations

from .comparisons import ProtocolComparison, compare_protocol_versions
from .grouping import GroupSummary, group_summary
from .missing import MissingDoseReport, analyze_missing_dose
from .outliers import OutlierFlag, detect_outliers
from .trends import MonthlyTrend, monthly_trends

__all__ = [
    "GroupSummary",
    "MissingDoseReport",
    "MonthlyTrend",
    "OutlierFlag",
    "ProtocolComparison",
    "analyze_missing_dose",
    "compare_protocol_versions",
    "detect_outliers",
    "group_summary",
    "monthly_trends",
]
