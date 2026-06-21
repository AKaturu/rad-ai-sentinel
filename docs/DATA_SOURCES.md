# Public Data Sources

`rad-ai-sentinel` monitors existing AI predictions. Public radiology datasets are useful for smoke tests, but most do not include deployed AI model outputs. For publication-quality work, pair one of these datasets with predictions from a fixed external model or with institutional post-deployment predictions.

## RSNA Pneumonia Detection Challenge

Recommended for an immediate public-data smoke test.

- Source: RSNA 2018 Pneumonia Detection Challenge.
- Content: 30,000 frontal chest radiographs derived from the NIH CXR8 public dataset, expert pneumonia labels, bounding boxes, and DICOM metadata.
- Useful metadata: patient age, patient sex, projection, modality.
- Limitation: no deployed model predictions, scanner manufacturer, race/ethnicity, or real site drift.

Adapter:

```bash
rad-ai-sentinel adapt-rsna stage_2_train_labels.csv outputs/rsna_monitoring.csv
rad-ai-sentinel report --csv outputs/rsna_monitoring.csv --output outputs/rsna_report
```

With real model predictions:

```bash
rad-ai-sentinel adapt-rsna stage_2_train_labels.csv outputs/rsna_monitoring.csv \
  --predictions-csv my_model_predictions.csv \
  --metadata-csv dicom_metadata_extract.csv \
  --model-version pneumonia-model-v1
```

Expected prediction CSV columns:

- `patientId` or `patient_id`
- `prediction` or `y_pred_proba`

## MIMIC-CXR

Recommended for richer institutional-style research after credentialing.

- Source: PhysioNet MIMIC-CXR.
- Content: de-identified DICOM chest radiographs, reports, and metadata from Beth Israel Deaconess Medical Center.
- Strength: closer to an institutional data workflow than challenge data.
- Limitation: requires data-use agreement and credentialing; no deployed AI predictions by default.

## NIH Chest X-ray Dataset

Useful for open image-label smoke tests and external validation demos.

- Source: NIH Clinical Center Chest X-ray dataset.
- Content: de-identified chest x-ray images and weak labels.
- Strength: open access with attribution requirements.
- Limitation: weak labels, limited operational metadata, no deployed model outputs.

## CheXpert

Useful for external validation and uncertainty-label research.

- Source: Stanford ML Group / Stanford AIMI.
- Content: chest radiographs with uncertainty labels and radiologist-labeled reference sets.
- Strength: strong evaluation framing.
- Limitation: access terms apply; no deployed model outputs by default.
