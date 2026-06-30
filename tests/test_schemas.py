"""Tests for the CSV input schema (the gate every upload must pass)."""

from __future__ import annotations

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from rad_ai_sentinel.config import (
    COL_PATIENT_ID,
    COL_Y_PRED_BINARY,
    COL_Y_PRED_PROBA,
    COL_Y_TRUE,
)
from rad_ai_sentinel.schemas import SchemaProfile, validate_dataframe

# We validate with lazy=True, so multi-error failures raise SchemaErrors (plural)
# and single-error validations also raise the collective SchemaErrors. Both
# subclasses share the base SchemaError.
_ValidationFailure = (SchemaError, SchemaErrors)


def test_valid_dataframe_passes_and_coerces(tiny_df: pd.DataFrame) -> None:
    out = validate_dataframe(tiny_df)
    # All columns preserved.
    assert set(tiny_df.columns).issubset(set(out.columns))
    # Coercion happened: y_true is integer, study_date is datetime.
    assert pd.api.types.is_integer_dtype(out[COL_Y_TRUE])
    assert pd.api.types.is_datetime64_any_dtype(out["study_date"])


def test_valid_minimal_dataframe_without_metadata_passes() -> None:
    """Optional metadata columns may be entirely absent."""
    df = pd.DataFrame(
        {
            COL_PATIENT_ID: ["p1", "p2"],
            "study_date": ["2026-01-01", "2026-01-02"],
            "model_version": ["v2.0", "v2.0"],
            COL_Y_TRUE: [1, 0],
            COL_Y_PRED_PROBA: [0.9, 0.1],
            COL_Y_PRED_BINARY: [1, 0],
        }
    )
    out = validate_dataframe(df)
    assert len(out) == 2


def test_production_profile_requires_operational_metadata() -> None:
    """Production monitoring requires site/scanner/modality metadata."""
    df = pd.DataFrame(
        {
            COL_PATIENT_ID: ["p1", "p2"],
            "study_date": ["2026-01-01", "2026-01-02"],
            "model_version": ["v2.0", "v2.0"],
            COL_Y_TRUE: [1, 0],
            COL_Y_PRED_PROBA: [0.9, 0.1],
            COL_Y_PRED_BINARY: [1, 0],
        }
    )
    with pytest.raises(ValueError, match="Production schema profile"):
        validate_dataframe(df, profile=SchemaProfile.PRODUCTION)


def test_out_of_range_probability_raises(tiny_df: pd.DataFrame) -> None:
    bad = tiny_df.copy()
    bad.loc[0, COL_Y_PRED_PROBA] = 1.5
    with pytest.raises(_ValidationFailure):
        validate_dataframe(bad)


def test_negative_probability_raises(tiny_df: pd.DataFrame) -> None:
    bad = tiny_df.copy()
    bad.loc[0, COL_Y_PRED_PROBA] = -0.1
    with pytest.raises(_ValidationFailure):
        validate_dataframe(bad)


def test_invalid_binary_label_raises(tiny_df: pd.DataFrame) -> None:
    bad = tiny_df.copy()
    bad.loc[0, COL_Y_TRUE] = 2
    with pytest.raises(_ValidationFailure):
        validate_dataframe(bad)


def test_invalid_binary_prediction_raises(tiny_df: pd.DataFrame) -> None:
    bad = tiny_df.copy()
    bad.loc[0, COL_Y_PRED_BINARY] = 5
    with pytest.raises(_ValidationFailure):
        validate_dataframe(bad)


def test_missing_required_column_raises(tiny_df: pd.DataFrame) -> None:
    bad = tiny_df.drop(columns=[COL_Y_TRUE])
    with pytest.raises(_ValidationFailure):
        validate_dataframe(bad)


def test_empty_patient_id_raises(tiny_df: pd.DataFrame) -> None:
    bad = tiny_df.copy()
    bad.loc[0, COL_PATIENT_ID] = ""
    with pytest.raises(_ValidationFailure):
        validate_dataframe(bad)


def test_lazy_validation_reports_multiple_errors(tiny_df: pd.DataFrame) -> None:
    """All violations surface at once, not just the first."""
    bad = tiny_df.copy()
    bad.loc[0, COL_Y_PRED_PROBA] = 2.0  # error 1
    bad.loc[1, COL_Y_TRUE] = 9  # error 2
    with pytest.raises(_ValidationFailure) as excinfo:
        validate_dataframe(bad)
    # The SchemaErrors object carries a non-empty failure case table.
    schema_errors = excinfo.value
    assert schema_errors.error_counts is not None
    assert sum(schema_errors.error_counts.values()) >= 2


def test_extra_benign_column_is_tolerated(tiny_df: pd.DataFrame) -> None:
    """strict=False: an unlisted column does not break validation."""
    df = tiny_df.copy()
    df["laterality"] = ["L", "R", "L", "R"]
    out = validate_dataframe(df)
    assert "laterality" in out.columns
