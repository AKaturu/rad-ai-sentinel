# rad-ai-sentinel

**An open-source framework for site-, scanner-, subgroup-, version-, and time-stratified surveillance of radiology AI performance.**

`rad-ai-sentinel` operationalizes post-deployment monitoring for existing imaging AI model outputs. It does **not** train an imaging model. Bring a CSV of predictions, ground truth, and study metadata; the tool validates the file, computes performance and drift metrics, surfaces stop-rule alerts, and generates a downloadable monitoring report.

The project is motivated by the ACR-SIIM Practice Parameter for Imaging AI, approved by the ACR Council on May 5, 2026, which emphasizes AI inventory/version tracking, local acceptance testing, ongoing performance monitoring, drift/safety evaluation, stop rules, and privacy controls.

## Demo

![rad-ai-sentinel demo walkthrough](screenshots/demo.gif)

PDF export walkthrough:

<video src="screenshots/demo-pdf-export.webm" controls width="100%"></video>

Direct video link: [screenshots/demo-pdf-export.webm](screenshots/demo-pdf-export.webm)

## Screenshots

![Dashboard overview](screenshots/dashboard-overview.png)

![Subgroup performance](screenshots/dashboard-subgroups.png)

![Report downloads](screenshots/dashboard-report.png)

![Mobile dashboard](screenshots/dashboard-mobile.png)

## What It Produces

- Sensitivity, specificity, PPV, NPV, accuracy, F1, prevalence, AUROC, and AUPRC.
- Wilson confidence intervals for 2x2 metrics and bootstrap CIs for curve/calibration metrics.
- Calibration analysis with Brier score, expected calibration error, and reliability curves.
- Subgroup performance by age group, sex, race/ethnicity, site, scanner manufacturer, and modality.
- Scanner- and site-specific performance tables.
- Missing-data analysis, including subgroup availability and outcome-associated missingness flags.
- Temporal drift detection with PSI, KL divergence, rolling AUROC, and CUSUM.
- Configurable stop-rule alerts.
- Model-version comparisons, including DeLong AUROC comparison when versions share a common case set.
- Machine-readable CSV/JSON outputs.
- Downloadable HTML report, with optional PDF export when WeasyPrint system libraries are installed.

## Quick Start

```bash
pip install -e ".[dev]"

# Generate synthetic monitoring data, metrics, and an HTML report.
rad-ai-sentinel demo

# Analyze your own monitoring CSV.
rad-ai-sentinel compute --csv path/to/predictions.csv --output outputs/analysis

# Generate a report.
rad-ai-sentinel report --csv path/to/predictions.csv --output outputs/report

# Launch the dashboard.
rad-ai-sentinel serve
```

Then open [http://localhost:8501](http://localhost:8501).

## Docker

```bash
docker build -t rad-ai-sentinel .
docker run --rm -p 8501:8501 rad-ai-sentinel
```

## CSV Format

| column | type | required | description |
|---|---:|:---:|---|
| `patient_id` | string | yes | de-identified patient identifier |
| `study_date` | date | yes | examination date, `YYYY-MM-DD` |
| `site` | string | no | site, facility, or reader group |
| `scanner_manufacturer` | string | no | scanner manufacturer |
| `modality` | string | no | modality such as `DX`, `CR`, `CT`, `MR` |
| `age_group` | string | no | age band such as `18-39`, `40-64`, `65+` |
| `sex` | string | no | sex where available and appropriate |
| `race_ethnicity` | string | no | race/ethnicity where available and appropriate |
| `model_version` | string | yes | model version label |
| `y_true` | int | yes | ground-truth label, 0 or 1 |
| `y_pred_proba` | float | yes | model probability for class 1, from 0 to 1 |
| `y_pred_binary` | int | yes | thresholded prediction, 0 or 1 |

Optional metadata columns may be absent or partially missing. Missingness is reported rather than silently ignored.

## Public Data Smoke Test

Public radiology datasets usually include images, labels, and metadata, but not deployed AI outputs. For a public smoke test, use the RSNA Pneumonia Detection Challenge labels and add either your own model predictions or deterministic synthetic scores for pipeline validation:

```bash
rad-ai-sentinel adapt-rsna stage_2_train_labels.csv outputs/rsna_monitoring.csv
rad-ai-sentinel report --csv outputs/rsna_monitoring.csv --output outputs/rsna_report
```

With model predictions:

```bash
rad-ai-sentinel adapt-rsna stage_2_train_labels.csv outputs/rsna_monitoring.csv \
  --predictions-csv my_model_predictions.csv \
  --metadata-csv dicom_metadata_extract.csv \
  --model-version pneumonia-model-v1
```

See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for RSNA/NIH, MIMIC-CXR, NIH ChestX-ray, and CheXpert notes.

## Development

```bash
python -m ruff check .
python -m pytest
```

Current local verification:

- `python -m ruff check .`
- `python -m pytest` -> 61 passed

## Safety Note

This project is for monitoring and research workflows. It is not a medical device, does not certify clinical performance, and should not be used to make patient-care decisions without appropriate institutional review, validation, governance, privacy controls, and clinical oversight.

## License

MIT. See [LICENSE](LICENSE).
