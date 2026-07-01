"""Regression fixtures for CSV validation and small-sample subgroup behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from rad_ai_sentinel.analysis import run_monitoring_analysis, stratified_metrics_frame
from rad_ai_sentinel.schemas import validate_dataframe

FIXTURES = Path(__file__).parent / "fixtures"
_ValidationFailure = (SchemaError, SchemaErrors)


def test_malformed_probability_fixture_fails_validation() -> None:
    df = pd.read_csv(FIXTURES / "malformed_probability.csv")
    with pytest.raises(_ValidationFailure):
        validate_dataframe(df)


def test_missing_subgroup_fields_fixture_public_passes_production_fails() -> None:
    df = pd.read_csv(FIXTURES / "missing_subgroup_fields.csv")
    assert len(validate_dataframe(df, profile="public")) == 4
    with pytest.raises(ValueError, match="Production schema profile"):
        validate_dataframe(df, profile="production")


def test_small_sample_subgroup_fixture_suppresses_unstable_estimates() -> None:
    df = pd.read_csv(FIXTURES / "small_sample_subgroups.csv")
    analysis = run_monitoring_analysis(df, subgroup_min_n=3, n_resamples=20)
    table = stratified_metrics_frame(analysis)
    sex_rows = table[table["stratifier"] == "sex"].set_index("level")
    assert sex_rows.loc["F", "status"] == "insufficient_data"
    assert sex_rows.loc["M", "status"] == "ok"
