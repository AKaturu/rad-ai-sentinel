"""Reusable public-data case-study scaffolds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_rsna_case_study_template(output_dir: str | Path, *, force: bool = False) -> dict[str, Path]:
    """Write a safe RSNA external-prediction case-study scaffold.

    The scaffold is documentation plus CSV templates. It intentionally does not
    download RSNA data, generate model predictions, or imply clinical validity.
    """
    destination = Path(output_dir)
    files = {
        "readme": destination / "README.md",
        "predictions": destination / "predictions_template.csv",
        "metadata": destination / "metadata_template.csv",
        "analysis_plan": destination / "analysis_plan.md",
    }
    existing = [path for path in files.values() if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing template files: {names}")

    destination.mkdir(parents=True, exist_ok=True)
    files["readme"].write_text(_readme_text(), encoding="utf-8")
    files["analysis_plan"].write_text(_analysis_plan_text(), encoding="utf-8")
    _prediction_template().to_csv(files["predictions"], index=False)
    _metadata_template().to_csv(files["metadata"], index=False)
    return files


def _prediction_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"patientId": "example_patient_0001", "prediction": 0.73},
            {"patientId": "example_patient_0002", "prediction": 0.18},
        ]
    )


def _metadata_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patientId": "example_patient_0001",
                "StudyDate": "2018-01-01",
                "PatientSex": "F",
                "PatientAge": "064Y",
                "Modality": "DX",
                "Manufacturer": "ExampleManufacturer",
            },
            {
                "patientId": "example_patient_0002",
                "StudyDate": "2018-01-02",
                "PatientSex": "M",
                "PatientAge": "045Y",
                "Modality": "DX",
                "Manufacturer": "ExampleManufacturer",
            },
        ]
    )


def _readme_text() -> str:
    return """# RSNA External-Prediction Case Study Template

This folder is a safe scaffold for a public-data monitoring case study using
RSNA Pneumonia Detection Challenge labels plus predictions from a fixed external
model. It is not a clinical validation template and should not be used to make
patient-care, deployment, regulatory, or model-superiority claims.

## Required Inputs

1. `stage_2_train_labels.csv` from the RSNA Pneumonia Detection Challenge.
2. A prediction CSV with `patientId` and `prediction` columns, where
   `prediction` is the model probability for the positive class.
3. Optional de-identified metadata with `patientId`, `StudyDate`, `PatientSex`,
   `PatientAge`, `Modality`, and `Manufacturer`.

## Example Commands

```bash
rad-ai-sentinel adapt-rsna stage_2_train_labels.csv outputs/rsna_monitoring.csv \\
  --predictions-csv predictions_template.csv \\
  --metadata-csv metadata_template.csv \\
  --threshold 0.5 \\
  --model-version fixed-external-model-v1

rad-ai-sentinel report --csv outputs/rsna_monitoring.csv --output outputs/rsna_report
```

## Claim Boundary

Acceptable language: "This example demonstrates the monitoring pipeline on a
public challenge-label dataset paired with externally generated predictions."

Avoid: claims that the model is clinically valid, deployed, prospectively
monitored, FDA-cleared, or superior to another model unless those claims are
supported by an independently reviewed study design.
"""


def _analysis_plan_text() -> str:
    return """# Analysis Plan

## Objective

Demonstrate the `rad-ai-sentinel` monitoring workflow on RSNA challenge labels
paired with predictions from a fixed external model.

## Inclusion Rules

- Include only rows with a unique `patientId`.
- Collapse multiple RSNA boxes per patient to a binary patient-level target via
  the adapter's max-label rule.
- Exclude rows without a model prediction from performance summaries, or document
  the missing-prediction count if imputation is intentionally used elsewhere.

## Required Reporting

- Input file names and retrieval dates.
- Prediction model name/version and how predictions were generated.
- Threshold used for `y_pred_binary`.
- Missing metadata counts by field.
- Overall AUROC/AUPRC, calibration, subgroup summaries, and drift outputs.

## Limitations

- RSNA challenge labels are not post-deployment monitoring data.
- Public labels do not establish local site performance.
- Metadata fields may be incomplete and are not a substitute for a governed
  monitoring plan.
"""
