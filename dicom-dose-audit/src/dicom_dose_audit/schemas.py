"""Dose-data schema validation for dicom-dose-audit.

Defines and enforces the input contract: which columns are required, their
types, and value ranges. This is the single gate every dose dataset (CSV or
parsed DICOM) must pass before the analytics engine touches it. Errors are
collected in full (lazy validation) so a user sees every problem at once.

Implemented with the explicit ``DataFrameSchema`` API for stability across
pandera releases (the ``DataFrameModel``/``Field(required=False)`` surface has
shifted between minor versions; the explicit schema form has not).

Dose quantities are always non-negative. ``study_date`` is coerced to a
datetime so monthly trends sort correctly regardless of input formatting.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from .config import (
    COL_CTDI_VOL,
    COL_DLP,
    COL_HAS_DOSE_SR,
    COL_KVP,
    COL_PATIENT_ID,
    COL_PROTOCOL,
    COL_PROTOCOL_VERSION,
    COL_SCAN_LENGTH_CM,
    COL_SCANNER_MANUFACTURER,
    COL_SCANNER_MODEL,
    COL_SITE,
    COL_SIZE_CATEGORY,
    COL_SOURCE,
    COL_STUDY_DATE,
    COL_STUDY_UID,
    COL_TUBE_CURRENT,
)

# Non-empty string helper for identifier columns.
_nonempty = pa.Check.str_length(min_value=1)


def _build_schema() -> pa.DataFrameSchema:
    """Construct the validated dose-data contract.

    Required columns reject nulls and empty strings; optional columns may be
    entirely absent (``required=False``) or contain nulls. Dose quantities are
    allowed to be null (missing-dose detection is a core feature) but never
    negative.
    """
    return pa.DataFrameSchema(
        {
            # --- required columns ---
            COL_STUDY_UID: pa.Column(
                str, checks=_nonempty, nullable=False, required=True, coerce=True
            ),
            COL_PATIENT_ID: pa.Column(
                str, checks=_nonempty, nullable=False, required=True, coerce=True
            ),
            COL_STUDY_DATE: pa.Column(
                "datetime64[ns]", nullable=False, required=True, coerce=True
            ),
            COL_PROTOCOL: pa.Column(
                str, checks=_nonempty, nullable=False, required=True, coerce=True
            ),
            COL_CTDI_VOL: pa.Column(
                float,
                checks=pa.Check.ge(0.0),
                nullable=True,  # missing dose is the audit target, not an error
                required=True,
                coerce=True,
            ),
            COL_DLP: pa.Column(
                float,
                checks=pa.Check.ge(0.0),
                nullable=True,
                required=True,
                coerce=True,
            ),
            # --- optional columns ---
            COL_PROTOCOL_VERSION: pa.Column(
                str, nullable=True, required=False, coerce=True
            ),
            COL_SCANNER_MODEL: pa.Column(str, nullable=True, required=False, coerce=True),
            COL_SCANNER_MANUFACTURER: pa.Column(
                str, nullable=True, required=False, coerce=True
            ),
            COL_SITE: pa.Column(str, nullable=True, required=False, coerce=True),
            COL_SIZE_CATEGORY: pa.Column(str, nullable=True, required=False, coerce=True),
            COL_KVP: pa.Column(float, checks=pa.Check.ge(0.0), nullable=True, required=False, coerce=True),
            COL_TUBE_CURRENT: pa.Column(
                float, checks=pa.Check.ge(0.0), nullable=True, required=False, coerce=True
            ),
            COL_SCAN_LENGTH_CM: pa.Column(
                float, checks=pa.Check.ge(0.0), nullable=True, required=False, coerce=True
            ),
            COL_HAS_DOSE_SR: pa.Column(bool, nullable=True, required=False, coerce=True),
            COL_SOURCE: pa.Column(str, nullable=True, required=False, coerce=True),
        },
        coerce=True,
        strict=False,  # tolerate benign extra columns
        add_missing_columns=False,
    )


# Build once at import time; the schema is immutable and reusable.
SCHEMA: pa.DataFrameSchema = _build_schema()


# Column name -> human description, for error messages and docs.
COLUMN_DESCRIPTIONS: dict[str, str] = {
    COL_STUDY_UID: "DICOM Study Instance UID (de-identified or hashed is fine)",
    COL_PATIENT_ID: "de-identified patient identifier",
    COL_STUDY_DATE: "examination date (YYYY-MM-DD)",
    COL_PROTOCOL: "scan protocol / clinical indication, e.g. 'CT Head'",
    COL_PROTOCOL_VERSION: "protocol version label for version comparison",
    COL_SCANNER_MODEL: "scanner model, e.g. 'Discovery CT750'",
    COL_SCANNER_MANUFACTURER: "scanner manufacturer, e.g. GE, Siemens, Philips",
    COL_SITE: "site / facility name",
    COL_SIZE_CATEGORY: "patient-size band: pediatric, small, medium, large",
    COL_CTDI_VOL: "volume CT dose index in mGy (may be null when missing)",
    COL_DLP: "dose-length product in mGy*cm (may be null when missing)",
    COL_KVP: "peak tube potential in kV",
    COL_TUBE_CURRENT: "tube current in mA",
    COL_SCAN_LENGTH_CM: "scan length in cm",
    COL_HAS_DOSE_SR: "True if a Radiation Dose SR was the source",
    COL_SOURCE: "data provenance label, e.g. 'dicom' or 'csv'",
}


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a raw dataframe against the dose-data contract.

    Parameters
    ----------
    df:
        Raw dataframe as read from CSV or produced by the DICOM reader. Dates
        may still be strings; the schema coerces them via ``coerce=True``.

    Returns
    -------
    pandas.DataFrame
        The validated, coerced dataframe.

    Raises
    ------
    pandera.errors.ValidationError
        Collected list of all contract violations (missing required columns,
        bad types, out-of-range values, etc.).
    """
    return SCHEMA.validate(df, lazy=True)
