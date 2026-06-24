"""Missing-dose detection.

Counts per-column missingness across the entire cohort and flags individual
studies where CTDIvol and/or DLP are absent. A study with missing dose metadata
is a core audit finding — the tool surfaces it so a medical physicist can
investigate whether the dose was not recorded, the study was cancelled, or the
DICOM export is incomplete.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import COL_CTDI_VOL, COL_DLP


@dataclass(frozen=True)
class ColumnMissingness:
    """Missing-data summary for one column."""

    column: str
    n_missing: int
    n_total: int
    pct_missing: float


@dataclass(frozen=True)
class MissingDoseReport:
    """Missing-dose analysis for the entire cohort."""

    per_column: list[ColumnMissingness]
    n_studies_missing_ctdi: int
    n_studies_missing_dlp: int
    n_studies_missing_both: int
    n_total: int


def analyze_missing_dose(df: pd.DataFrame) -> MissingDoseReport:
    """Compute missing-dose statistics for the cohort.

    Parameters
    ----------
    df:
        Validated dose dataframe.

    Returns
    -------
    MissingDoseReport with per-column missingness and per-study dose flags.
    """
    n_total = len(df)
    per_column: list[ColumnMissingness] = []

    for col in (COL_CTDI_VOL, COL_DLP):
        n_missing = int(df[col].isna().sum()) if col in df.columns else n_total
        per_column.append(
            ColumnMissingness(
                column=col,
                n_missing=n_missing,
                n_total=n_total,
                pct_missing=round(n_missing / max(n_total, 1), 4),
            )
        )

    has_ctdi = df[COL_CTDI_VOL].notna() if COL_CTDI_VOL in df.columns else pd.Series(False, index=df.index)
    has_dlp = df[COL_DLP].notna() if COL_DLP in df.columns else pd.Series(False, index=df.index)

    n_missing_ctdi = int((~has_ctdi).sum())
    n_missing_dlp = int((~has_dlp).sum())
    n_missing_both = int((~has_ctdi & ~has_dlp).sum())

    return MissingDoseReport(
        per_column=per_column,
        n_studies_missing_ctdi=n_missing_ctdi,
        n_studies_missing_dlp=n_missing_dlp,
        n_studies_missing_both=n_missing_both,
        n_total=n_total,
    )


def missing_dose_dataframe(report: MissingDoseReport) -> pd.DataFrame:
    """Convert a MissingDoseReport to a tidy DataFrame."""
    return pd.DataFrame(
        [
            {
                "column": item.column,
                "n_missing": item.n_missing,
                "n_total": item.n_total,
                "pct_missing": item.pct_missing,
            }
            for item in report.per_column
        ]
    )
