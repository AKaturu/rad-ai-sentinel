"""CSV schema validation for rad-ai-sentinel.

Defines and enforces the input contract: which columns are required, their
types, and value ranges. This is the single gate every uploaded CSV must pass
before the metrics engine touches it. Errors are collected in full (lazy
validation) so a user sees every problem at once, not one at a time.

Implemented with the explicit ``DataFrameSchema`` API for stability across
pandera releases (the ``DataFrameModel``/``Field(required=False)`` surface has
shifted between minor versions; the explicit schema form has not).
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from .config import (
    COL_AGE_GROUP,
    COL_MODALITY,
    COL_MODEL_VERSION,
    COL_PATIENT_ID,
    COL_RACE_ETHNICITY,
    COL_SCANNER_MANUFACTURER,
    COL_SEX,
    COL_SITE,
    COL_STUDY_DATE,
    COL_Y_PRED_BINARY,
    COL_Y_PRED_PROBA,
    COL_Y_TRUE,
)

# Helper for non-empty string columns.
_nonempty = pa.Check.str_length(min_value=1)


def _build_schema() -> pa.DataFrameSchema:
    """Construct the validated input contract.

    Required columns reject nulls and empty strings; optional metadata columns
    may be entirely absent (``required=False``) or contain nulls.
    """
    return pa.DataFrameSchema(
        {
            # --- required columns ---
            COL_PATIENT_ID: pa.Column(
                str, checks=_nonempty, nullable=False, required=True, coerce=True
            ),
            COL_STUDY_DATE: pa.Column("datetime64[ns]", nullable=False, required=True, coerce=True),
            COL_MODEL_VERSION: pa.Column(
                str, checks=_nonempty, nullable=False, required=True, coerce=True
            ),
            COL_Y_TRUE: pa.Column(
                int,
                checks=pa.Check.isin([0, 1]),
                nullable=False,
                required=True,
                coerce=True,
            ),
            COL_Y_PRED_PROBA: pa.Column(
                float,
                checks=[pa.Check.ge(0.0), pa.Check.le(1.0)],
                nullable=False,
                required=True,
                coerce=True,
            ),
            COL_Y_PRED_BINARY: pa.Column(
                int,
                checks=pa.Check.isin([0, 1]),
                nullable=False,
                required=True,
                coerce=True,
            ),
            # --- optional metadata columns ---
            COL_SITE: pa.Column(str, nullable=True, required=False, coerce=True),
            COL_SCANNER_MANUFACTURER: pa.Column(str, nullable=True, required=False, coerce=True),
            COL_MODALITY: pa.Column(str, nullable=True, required=False, coerce=True),
            COL_AGE_GROUP: pa.Column(str, nullable=True, required=False, coerce=True),
            COL_SEX: pa.Column(str, nullable=True, required=False, coerce=True),
            COL_RACE_ETHNICITY: pa.Column(str, nullable=True, required=False, coerce=True),
        },
        coerce=True,
        strict=False,  # tolerate benign extra columns
        add_missing_columns=False,
    )


# Build once at import time; the schema is immutable and reusable.
SCHEMA: pa.DataFrameSchema = _build_schema()


# Column name -> human description, for error messages and docs.
COLUMN_DESCRIPTIONS: dict[str, str] = {
    COL_PATIENT_ID: "de-identified patient identifier",
    COL_STUDY_DATE: "examination date (YYYY-MM-DD)",
    COL_MODEL_VERSION: "AI model version string, e.g. 'v2.1'",
    COL_Y_TRUE: "ground-truth label, must be 0 or 1",
    COL_Y_PRED_PROBA: "model probability for class 1, must be in [0, 1]",
    COL_Y_PRED_BINARY: "thresholded model prediction, must be 0 or 1",
    COL_SITE: "reading site / facility name",
    COL_SCANNER_MANUFACTURER: "scanner manufacturer, e.g. GE, Siemens, Philips",
    COL_MODALITY: "imaging modality, e.g. CR, DX, CT, MR",
    COL_AGE_GROUP: "age band, e.g. 0-17, 18-39, 40-64, 65+",
    COL_SEX: "sex, e.g. F, M",
    COL_RACE_ETHNICITY: "self-reported race/ethnicity, where available",
}


def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a raw dataframe against the input contract.

    Parameters
    ----------
    df:
        Raw dataframe as read from CSV (dates may still be strings; the schema
        coerces them via ``coerce=True``).

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
