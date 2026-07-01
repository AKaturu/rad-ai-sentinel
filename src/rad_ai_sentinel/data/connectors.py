"""De-identified connector examples for operational prediction exports."""

from __future__ import annotations

from pathlib import Path

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
from ..schemas import SchemaProfile, validate_dataframe

PREDICTION_ALIASES: dict[str, tuple[str, ...]] = {
    COL_PATIENT_ID: (
        "patient_id",
        "patientId",
        "patient_hash",
        "accession_number_hash",
        "study_instance_uid_hash",
        "subject.reference",
    ),
    COL_STUDY_DATE: ("study_date", "StudyDate", "effectiveDateTime", "observation_datetime"),
    COL_MODEL_VERSION: ("model_version", "algorithm_version", "modelVersion", "software_version"),
    COL_Y_TRUE: ("y_true", "ground_truth", "target", "label", "outcome"),
    COL_Y_PRED_PROBA: ("y_pred_proba", "prediction", "score", "probability", "model_score"),
    COL_SITE: ("site", "facility", "serviceProvider.display", "MSH.4"),
    COL_SCANNER_MANUFACTURER: (
        "scanner_manufacturer",
        "Manufacturer",
        "device.manufacturer",
        "Device.manufacturer",
    ),
    COL_MODALITY: ("modality", "Modality", "modality.coding.code", "OBR.24"),
    COL_AGE_GROUP: ("age_group", "ageBand", "PatientAgeGroup"),
    COL_SEX: ("sex", "PatientSex", "administrativeGender"),
    COL_RACE_ETHNICITY: ("race_ethnicity", "raceEthnicity", "race_ethnicity_group"),
}


def adapt_prediction_export(
    export_csv: str | Path,
    output_csv: str | Path,
    *,
    metadata_csv: str | Path | None = None,
    threshold: float = 0.5,
    model_version: str = "external-model",
    schema_profile: str | SchemaProfile = SchemaProfile.PRODUCTION,
) -> pd.DataFrame:
    """Normalize a de-identified prediction export into the monitoring schema."""
    predictions = pd.read_csv(export_csv)
    normalized = normalize_prediction_export(
        predictions,
        threshold=threshold,
        model_version=model_version,
    )

    if metadata_csv:
        metadata = normalize_operational_metadata(pd.read_csv(metadata_csv))
        metadata_cols = [
            c for c in metadata.columns if c != COL_PATIENT_ID and c not in normalized.columns
        ]
        normalized = normalized.merge(
            metadata[[COL_PATIENT_ID, *metadata_cols]].drop_duplicates(COL_PATIENT_ID),
            on=COL_PATIENT_ID,
            how="left",
        )

    for optional in (
        COL_SITE,
        COL_SCANNER_MANUFACTURER,
        COL_MODALITY,
        COL_AGE_GROUP,
        COL_SEX,
        COL_RACE_ETHNICITY,
    ):
        if optional not in normalized.columns:
            normalized[optional] = None

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
    validated = validate_dataframe(normalized, profile=schema_profile)
    validated = validated[[*ordered, *[c for c in validated.columns if c not in ordered]]]
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(destination, index=False)
    return validated


def normalize_prediction_export(
    df: pd.DataFrame,
    *,
    threshold: float = 0.5,
    model_version: str = "external-model",
) -> pd.DataFrame:
    """Rename common prediction-export aliases to the public CSV contract."""
    out = _rename_aliases(df, PREDICTION_ALIASES)
    if COL_MODEL_VERSION not in out.columns:
        out[COL_MODEL_VERSION] = model_version
    if COL_Y_PRED_BINARY not in out.columns and COL_Y_PRED_PROBA in out.columns:
        out[COL_Y_PRED_BINARY] = (out[COL_Y_PRED_PROBA].astype(float) >= threshold).astype(int)
    return out


def normalize_operational_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize FHIR/HL7-inspired metadata aliases used by local registries."""
    return _rename_aliases(df, PREDICTION_ALIASES)


def write_connector_templates(output_dir: str | Path, *, force: bool = False) -> dict[str, Path]:
    """Write safe de-identified connector example CSVs and notes."""
    destination = Path(output_dir)
    files = {
        "readme": destination / "README.md",
        "pacs_ris": destination / "pacs_ris_prediction_export_template.csv",
        "orchestration": destination / "ai_orchestration_export_template.csv",
        "metadata": destination / "fhir_hl7_metadata_template.csv",
    }
    existing = [path for path in files.values() if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing connector templates: {names}")

    destination.mkdir(parents=True, exist_ok=True)
    files["readme"].write_text(_connector_readme(), encoding="utf-8")
    _pacs_ris_template().to_csv(files["pacs_ris"], index=False)
    _orchestration_template().to_csv(files["orchestration"], index=False)
    _metadata_template().to_csv(files["metadata"], index=False)
    return files


def _rename_aliases(df: pd.DataFrame, aliases: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        if canonical in df.columns:
            continue
        source = next((candidate for candidate in candidates if candidate in df.columns), None)
        if source:
            rename_map[source] = canonical
    return df.rename(columns=rename_map).copy()


def _pacs_ris_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "accession_number_hash": "acc_hash_0001",
                "study_date": "2026-01-01",
                "facility": "Example Hospital",
                "Manufacturer": "ExampleManufacturer",
                "Modality": "DX",
                "algorithm_version": "triage-v1.0",
                "ground_truth": 1,
                "model_score": 0.82,
            },
            {
                "accession_number_hash": "acc_hash_0002",
                "study_date": "2026-01-02",
                "facility": "Example Hospital",
                "Manufacturer": "ExampleManufacturer",
                "Modality": "DX",
                "algorithm_version": "triage-v1.0",
                "ground_truth": 0,
                "model_score": 0.21,
            },
        ]
    )


def _orchestration_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "study_instance_uid_hash": "study_hash_0001",
                "observation_datetime": "2026-01-01T10:30:00",
                "site": "Example Hospital",
                "scanner_manufacturer": "ExampleManufacturer",
                "modality": "CT",
                "modelVersion": "orchestrator-model-v3",
                "label": 1,
                "prediction": 0.77,
            }
        ]
    )


def _metadata_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject.reference": "acc_hash_0001",
                "effectiveDateTime": "2026-01-01T10:30:00",
                "serviceProvider.display": "Example Hospital",
                "device.manufacturer": "ExampleManufacturer",
                "modality.coding.code": "DX",
                "administrativeGender": "female",
                "ageBand": "40-64",
                "raceEthnicity": "Not exported",
            }
        ]
    )


def _connector_readme() -> str:
    return """# De-identified Connector Templates

These templates show common CSV shapes exported from PACS/RIS worklists and AI
orchestration tools. They are examples only: do not export PHI, raw accession
numbers, DICOM UIDs, reports, names, dates of birth, or free-text identifiers.

Use hashed study/patient identifiers, operational metadata approved by the local
governance process, model probabilities, thresholded predictions, and outcome
labels. The `production` schema profile requires site, scanner manufacturer, and
modality so local monitoring does not silently collapse operational context.
"""
