"""Missing-data analysis for radiology AI monitoring.

Clinical datasets routinely have incomplete demographics and metadata. This
module quantifies missingness rates per column, identifies patterns (MCAR vs
potential MAR indicators), and assesses whether missingness could bias subgroup
performance estimates.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import (
    ALL_STRATIFIER_COLUMNS,
    COL_Y_TRUE,
    REQUIRED_COLUMNS,
)


@dataclass(frozen=True)
class ColumnMissingness:
    """Missingness profile for a single column."""

    column: str
    n_missing: int
    n_total: int
    pct_missing: float
    is_metadata: bool  # True if it's a stratifier (not a required pred column)


@dataclass(frozen=True)
class SubgroupAvailability:
    """How many samples are available for each level of a stratifier."""

    stratifier: str
    total_samples: int
    available_samples: int  # non-null in this column
    pct_available: float
    levels: dict[str, int]  # level_name -> count (excluding NaN)


@dataclass(frozen=True)
class MissingDataReport:
    """Complete missing-data analysis."""

    overall_n: int
    per_column: list[ColumnMissingness]
    subgroup_availability: list[SubgroupAvailability]
    # Potential MAR indicators: columns where missingness correlates with y_true.
    mar_indicators: list[str]


def analyze_missing_data(df: pd.DataFrame) -> MissingDataReport:
    """Analyze missing data patterns and subgroup availability.

    Parameters
    ----------
    df:
        Validated dataframe (schema-checked). May have NaN in optional columns.

    Returns
    -------
    MissingDataReport
    """
    n = len(df)
    per_column: list[ColumnMissingness] = []
    mar_indicators: list[str] = []

    # Analyze all relevant columns.
    interesting_cols = list(ALL_STRATIFIER_COLUMNS)
    # Also check any extra object columns that might be metadata.
    for col in df.columns:
        if col not in REQUIRED_COLUMNS and col not in interesting_cols and df[col].dtype == object:
            interesting_cols.append(col)

    for col in interesting_cols:
        if col not in df.columns:
            continue
        n_missing = int(df[col].isna().sum())
        is_meta = col in ALL_STRATIFIER_COLUMNS
        per_column.append(
            ColumnMissingness(
                column=col,
                n_missing=n_missing,
                n_total=n,
                pct_missing=n_missing / n if n > 0 else 0.0,
                is_metadata=is_meta,
            )
        )

        # MAR indicator check: does missingness in this column correlate
        # with the outcome (y_true)? Use a simple chi-square test.
        if n_missing > 0 and n_missing < n and COL_Y_TRUE in df.columns:
            # 2x2 contingency: (missing vs present) x (positive vs negative).
            missing_mask = df[col].isna()
            ct = pd.crosstab(missing_mask, df[COL_Y_TRUE])
            if ct.shape == (2, 2):
                from scipy.stats import chi2_contingency

                _, p_val, _, _ = chi2_contingency(ct)
                if p_val < 0.05:
                    mar_indicators.append(col)

    # Subgroup availability: for each stratifier, count available samples per level.
    subgroup_avail: list[SubgroupAvailability] = []
    for col in ALL_STRATIFIER_COLUMNS:
        if col not in df.columns:
            continue
        available = df[col].dropna()
        levels = available.value_counts().to_dict()
        subgroup_avail.append(
            SubgroupAvailability(
                stratifier=col,
                total_samples=n,
                available_samples=len(available),
                pct_available=len(available) / n if n > 0 else 0.0,
                levels={str(k): int(v) for k, v in levels.items()},
            )
        )

    return MissingDataReport(
        overall_n=n,
        per_column=per_column,
        subgroup_availability=subgroup_avail,
        mar_indicators=mar_indicators,
    )
