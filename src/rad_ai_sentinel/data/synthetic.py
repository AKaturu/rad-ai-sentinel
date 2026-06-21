"""Synthetic and public-data adapters for rad-ai-sentinel.

The synthetic generator is deliberately not an imaging model. It creates
plausible post-deployment monitoring rows: existing model scores, ground truth,
exam metadata, model versions, missing demographics, and planted drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
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


def _age_group_from_years(age: float | int | str | None) -> str | None:
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return None
    text = str(age).strip()
    if not text:
        return None
    if text[-1:].isalpha():
        text = text[:-1]
    try:
        years = int(float(text))
    except ValueError:
        return None
    if years < 18:
        return "0-17"
    if years < 40:
        return "18-39"
    if years < 65:
        return "40-64"
    return "65+"


def generate_synthetic_monitoring_data(
    n: int = 1200,
    *,
    seed: int = 42,
    threshold: float = 0.5,
    start_date: str = "2025-10-01",
    include_version_holdout: bool = True,
) -> pd.DataFrame:
    """Generate a realistic monitoring CSV with planted quality issues."""
    rng = np.random.default_rng(seed)
    sites = np.array(["North Hospital", "South Hospital", "Community Imaging"])
    scanners = np.array(["GE", "Siemens", "Philips", "Canon"])
    modalities = np.array(["DX", "CR", "CT"])
    age_groups = np.array(["0-17", "18-39", "40-64", "65+"])
    sexes = np.array(["F", "M"])
    races = np.array(["Asian", "Black", "Hispanic", "White", "Other"])
    dates = pd.date_range(start=start_date, periods=max(n // 8, 1), freq="D")

    rows: list[dict[str, object]] = []
    for i in range(n):
        date = dates[min(i // 8, len(dates) - 1)]
        site = str(rng.choice(sites, p=[0.42, 0.36, 0.22]))
        scanner = str(rng.choice(scanners, p=[0.34, 0.30, 0.24, 0.12]))
        modality = str(rng.choice(modalities, p=[0.65, 0.25, 0.10]))
        age_group = str(rng.choice(age_groups, p=[0.10, 0.25, 0.38, 0.27]))
        sex = str(rng.choice(sexes))
        race: str | None = str(rng.choice(races, p=[0.12, 0.17, 0.15, 0.49, 0.07]))

        logit = -1.45
        logit += {"0-17": -0.45, "18-39": -0.25, "40-64": 0.05, "65+": 0.55}[age_group]
        logit += 0.45 if modality == "CT" else 0.0
        logit += 0.25 if site == "South Hospital" else 0.0
        prevalence = 1.0 / (1.0 + np.exp(-logit))
        y_true = int(rng.random() < prevalence)

        phase = i / max(n - 1, 1)
        version = "v1.0" if phase < 0.48 else "v2.0"
        drifted = phase >= 0.60
        separation = 2.35
        if drifted:
            separation -= 0.65
        if scanner == "Philips" and site == "South Hospital":
            separation -= 0.35
        if age_group == "65+":
            separation -= 0.15
        raw_score = -0.6 + separation * y_true + rng.normal(0.0, 0.85)
        if drifted and scanner == "Philips":
            raw_score += 0.35
        y_pred_proba = float(np.clip(1.0 / (1.0 + np.exp(-raw_score)), 0.001, 0.999))

        if (site == "Community Imaging" and rng.random() < 0.38) or rng.random() < 0.08:
            race = None
        scanner_value: str | None = scanner if rng.random() > 0.035 else None

        rows.append(
            {
                COL_PATIENT_ID: f"demo_{i:05d}",
                COL_STUDY_DATE: date.strftime("%Y-%m-%d"),
                COL_SITE: site,
                COL_SCANNER_MANUFACTURER: scanner_value,
                COL_MODALITY: modality,
                COL_AGE_GROUP: age_group,
                COL_SEX: sex,
                COL_RACE_ETHNICITY: race,
                COL_MODEL_VERSION: version,
                COL_Y_TRUE: y_true,
                COL_Y_PRED_PROBA: y_pred_proba,
                COL_Y_PRED_BINARY: int(y_pred_proba >= threshold),
            }
        )

    if include_version_holdout:
        rows.extend(_paired_version_holdout(rng, threshold, pd.Timestamp(start_date)))

    return pd.DataFrame(rows)


def _paired_version_holdout(
    rng: np.random.Generator,
    threshold: float,
    start: pd.Timestamp,
    n_patients: int = 180,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i in range(n_patients):
        y_true = int(rng.random() < 0.30)
        site = str(rng.choice(["North Hospital", "South Hospital"]))
        scanner = str(rng.choice(["GE", "Siemens", "Philips"]))
        date = start + pd.Timedelta(days=10 + i // 6)
        patient_id = f"holdout_{i:04d}"
        scores = {
            "v1.0": float(np.clip(0.20 + 0.58 * y_true + rng.normal(0.0, 0.12), 0.001, 0.999)),
            "v2.0": float(np.clip(0.25 + 0.45 * y_true + rng.normal(0.0, 0.22), 0.001, 0.999)),
        }
        for version, score in scores.items():
            rows.append(
                {
                    COL_PATIENT_ID: patient_id,
                    COL_STUDY_DATE: date.strftime("%Y-%m-%d"),
                    COL_SITE: site,
                    COL_SCANNER_MANUFACTURER: scanner,
                    COL_MODALITY: "DX",
                    COL_AGE_GROUP: str(rng.choice(["18-39", "40-64", "65+"])),
                    COL_SEX: str(rng.choice(["F", "M"])),
                    COL_RACE_ETHNICITY: str(
                        rng.choice(["Asian", "Black", "Hispanic", "White", "Other"])
                    ),
                    COL_MODEL_VERSION: version,
                    COL_Y_TRUE: y_true,
                    COL_Y_PRED_PROBA: score,
                    COL_Y_PRED_BINARY: int(score >= threshold),
                }
            )
    return rows


def write_synthetic_csv(path: str | Path, **kwargs: object) -> pd.DataFrame:
    """Generate synthetic data and write it to ``path``."""
    df = generate_synthetic_monitoring_data(**kwargs)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def adapt_rsna_pneumonia_labels(
    labels_csv: str | Path,
    output_csv: str | Path,
    *,
    predictions_csv: str | Path | None = None,
    metadata_csv: str | Path | None = None,
    seed: int = 42,
    threshold: float = 0.5,
    model_version: str = "external-model",
) -> pd.DataFrame:
    """Convert RSNA Pneumonia Detection labels into the monitor CSV contract.

    The public challenge provides ground-truth labels and large DICOM/image
    downloads. It does not provide deployed AI outputs, so this adapter accepts
    a user-supplied prediction CSV. If predictions are absent, it creates
    deterministic synthetic scores strictly for pipeline testing.
    """
    labels = pd.read_csv(labels_csv)
    if "patientId" not in labels.columns:
        raise ValueError("RSNA labels CSV must contain a patientId column")
    target_col = "Target" if "Target" in labels.columns else "target"
    if target_col not in labels.columns:
        raise ValueError("RSNA labels CSV must contain Target or target")

    grouped = (
        labels.groupby("patientId", as_index=False)[target_col]
        .max()
        .rename(columns={"patientId": COL_PATIENT_ID, target_col: COL_Y_TRUE})
    )

    out = grouped.copy()
    out[COL_MODEL_VERSION] = model_version
    out[COL_SITE] = "RSNA/NIH CXR8"
    out[COL_MODALITY] = "DX"

    if predictions_csv:
        preds = pd.read_csv(predictions_csv)
        patient_col = COL_PATIENT_ID if COL_PATIENT_ID in preds.columns else "patientId"
        score_col = COL_Y_PRED_PROBA if COL_Y_PRED_PROBA in preds.columns else "prediction"
        if patient_col not in preds.columns or score_col not in preds.columns:
            raise ValueError(
                "Predictions CSV must contain patient_id/patientId and y_pred_proba/prediction"
            )
        preds = preds[[patient_col, score_col]].rename(
            columns={patient_col: COL_PATIENT_ID, score_col: COL_Y_PRED_PROBA}
        )
        out = out.merge(preds, on=COL_PATIENT_ID, how="left")
    else:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 0.18, size=len(out))
        scores = 0.22 + 0.58 * out[COL_Y_TRUE].to_numpy(dtype=float) + noise
        out[COL_Y_PRED_PROBA] = np.clip(scores, 0.001, 0.999)

    if metadata_csv:
        meta = pd.read_csv(metadata_csv)
        patient_col = COL_PATIENT_ID if COL_PATIENT_ID in meta.columns else "patientId"
        if patient_col not in meta.columns:
            raise ValueError("Metadata CSV must contain patient_id or patientId")
        rename_map = {
            patient_col: COL_PATIENT_ID,
            "PatientSex": COL_SEX,
            "sex": COL_SEX,
            "Modality": COL_MODALITY,
            "Manufacturer": COL_SCANNER_MANUFACTURER,
            "StudyDate": COL_STUDY_DATE,
            "study_date": COL_STUDY_DATE,
        }
        available = {k: v for k, v in rename_map.items() if k in meta.columns}
        meta = meta.rename(columns=available)
        if COL_AGE_GROUP not in meta.columns:
            age_col = next((c for c in ("PatientAge", "age", "Age") if c in meta.columns), None)
            if age_col:
                meta[COL_AGE_GROUP] = meta[age_col].map(_age_group_from_years)
        keep = [
            c
            for c in (
                COL_PATIENT_ID,
                COL_STUDY_DATE,
                COL_SEX,
                COL_MODALITY,
                COL_SCANNER_MANUFACTURER,
                COL_AGE_GROUP,
                COL_RACE_ETHNICITY,
            )
            if c in meta.columns
        ]
        out = out.merge(meta[keep].drop_duplicates(COL_PATIENT_ID), on=COL_PATIENT_ID, how="left")

    if COL_STUDY_DATE not in out.columns:
        out[COL_STUDY_DATE] = pd.date_range("2018-01-01", periods=len(out), freq="D").strftime(
            "%Y-%m-%d"
        )
    out[COL_Y_PRED_PROBA] = out[COL_Y_PRED_PROBA].astype(float).clip(0.0, 1.0)
    out[COL_Y_PRED_BINARY] = (out[COL_Y_PRED_PROBA] >= threshold).astype(int)

    for optional_col in (
        COL_SCANNER_MANUFACTURER,
        COL_AGE_GROUP,
        COL_SEX,
        COL_RACE_ETHNICITY,
    ):
        if optional_col not in out.columns:
            out[optional_col] = None

    ordered = [
        COL_PATIENT_ID,
        COL_STUDY_DATE,
        COL_SITE,
        COL_SCANNER_MANUFACTURER,
        COL_MODALITY,
        COL_AGE_GROUP,
        COL_SEX,
        COL_RACE_ETHNICITY,
        COL_MODEL_VERSION,
        COL_Y_TRUE,
        COL_Y_PRED_PROBA,
        COL_Y_PRED_BINARY,
    ]
    out = out[ordered]
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(destination, index=False)
    return out
