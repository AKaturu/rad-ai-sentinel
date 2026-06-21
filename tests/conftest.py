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
    patient_id_override: str | None = None,
) -> dict:
    """Build a single well-formed input row."""
    return {
        COL_PATIENT_ID: patient_id_override if patient_id_override else f"p{i:04d}",
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


@pytest.fixture
def drift_df(rng) -> pd.DataFrame:
    """A dataset spanning 6 months with a planted distribution shift at month 3.

    Pre-drift (days 0-89): scores well-separated -> high AUROC.
    Post-drift (days 90-179): separation collapses -> lower AUROC + shifted
    score distribution (detectable by PSI).
    ~25 studies/day for ~4500 rows.
    """
    rows = []
    start = pd.Timestamp("2026-01-01")
    for day in range(180):
        d = start + pd.Timedelta(days=day)
        for j in range(25):
            y = 1 if rng.random() < 0.2 else 0
            sep = 0.9 if day < 90 else 0.45  # planted shift
            score = y * 0.5 + sep * (y - 0.5) + rng.normal(0, 0.15)
            score = float(np.clip(score, 0.001, 0.999))
            rows.append(
                make_record(
                    day * 25 + j,
                    y_true=y,
                    y_pred_proba=score,
                    study_date=d.strftime("%Y-%m-%d"),
                    model_version="v2.0",
                    site=rng.choice(["Site A", "Site B"]),
                    scanner=rng.choice(["GE", "Siemens", "Philips"]),
                    modality="DX",
                    age_group=rng.choice(["18-39", "40-64", "65+"]),
                    sex=rng.choice(["F", "M"]),
                    race=rng.choice(["Group A", "Group B", "Group C"]),
                )
            )
    return pd.DataFrame(rows)


@pytest.fixture
def versions_df(rng) -> pd.DataFrame:
    """A dataset with two model versions on a common set of patients.

    v1.0: well-calibrated, higher AUROC.
    v2.0: miscalibrated (overconfident), lower AUROC.
    Both versions score the same 200 patients so DeLong comparison applies.
    """
    rows = []
    start = pd.Timestamp("2026-01-01")
    common_ids = [f"patient_{i:03d}" for i in range(200)]
    for i, pid in enumerate(common_ids):
        y = 1 if rng.random() < 0.3 else 0
        d = start + pd.Timedelta(days=i // 5)
        # v1.0: good separation
        score_v1 = float(np.clip(y * 0.6 + 0.2 + rng.normal(0, 0.12), 0.001, 0.999))
        # v2.0: noisier -> lower AUROC
        score_v2 = float(np.clip(y * 0.45 + 0.25 + rng.normal(0, 0.22), 0.001, 0.999))
        rows.append(
            make_record(
                2 * i,
                y_true=y,
                y_pred_proba=score_v1,
                study_date=d.strftime("%Y-%m-%d"),
                model_version="v1.0",
                patient_id_override=pid,
            )
        )
        rows.append(
            make_record(
                2 * i + 1,
                y_true=y,
                y_pred_proba=score_v2,
                study_date=d.strftime("%Y-%m-%d"),
                model_version="v2.0",
                patient_id_override=pid,
            )
        )
    return pd.DataFrame(rows)
