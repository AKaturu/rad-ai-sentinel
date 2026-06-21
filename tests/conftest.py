"""Shared pytest fixtures for rad-ai-sentinel tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rad_ai_sentinel.config import (
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


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG so synthetic test data is reproducible."""
    return np.random.default_rng(seed=42)


def make_record(
    i: int,
    *,
    y_true: int,
    y_pred_proba: float,
    study_date: str = "2026-01-01",
    model_version: str = "v2.0",
    site: str = "General Hospital",
    scanner: str = "GE",
    modality: str = "DX",
    age_group: str = "40-64",
    sex: str = "F",
    race: str = "Group A",
) -> dict:
    """Build a single well-formed input row."""
    return {
        COL_PATIENT_ID: f"p{i:04d}",
        COL_STUDY_DATE: study_date,
        COL_MODEL_VERSION: model_version,
        COL_Y_TRUE: y_true,
        COL_Y_PRED_PROBA: y_pred_proba,
        COL_Y_PRED_BINARY: int(round(y_pred_proba)),  # noqa: RUF046 - thresholded 0/1
        COL_SITE: site,
        COL_SCANNER_MANUFACTURER: scanner,
        COL_MODALITY: modality,
        COL_AGE_GROUP: age_group,
        COL_SEX: sex,
        COL_RACE_ETHNICITY: race,
    }


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """A tiny 4-row dataframe with a perfect classifier.

    Confusion matrix at threshold 0.5: TP=2, TN=2, FP=0, FN=0.
    Expected: sensitivity=1.0, specificity=1.0, PPV=1.0, NPV=1.0.
    """
    return pd.DataFrame(
        [
            make_record(1, y_true=1, y_pred_proba=0.9),
            make_record(2, y_true=1, y_pred_proba=0.8),
            make_record(3, y_true=0, y_pred_proba=0.2),
            make_record(4, y_true=0, y_pred_proba=0.1),
        ]
    )


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    """An 8-row dataframe with a realistic mix of TPs/FNs/FPs/TNs.

    Confusion matrix at threshold 0.5:
        TP=3, FN=1, FP=1, TN=3
        sensitivity = 3/4 = 0.75
        specificity = 3/4 = 0.75
        PPV = 3/4 = 0.75
        NPV = 3/4 = 0.75
    """
    rows = [
        make_record(1, y_true=1, y_pred_proba=0.95),  # TP
        make_record(2, y_true=1, y_pred_proba=0.80),  # TP
        make_record(3, y_true=1, y_pred_proba=0.70),  # TP
        make_record(4, y_true=1, y_pred_proba=0.40),  # FN
        make_record(5, y_true=0, y_pred_proba=0.60),  # FP
        make_record(6, y_true=0, y_pred_proba=0.30),  # TN
        make_record(7, y_true=0, y_pred_proba=0.20),  # TN
        make_record(8, y_true=0, y_pred_proba=0.10),  # TN
    ]
    return pd.DataFrame(rows)
