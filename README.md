# rad-ai-sentinel

[![CI](https://github.com/AKaturu/rad-ai-sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/AKaturu/rad-ai-sentinel/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**An open-source framework for site-, scanner-, subgroup-, version-, and time-stratified surveillance of radiology AI performance.**

`rad-ai-sentinel` operationalizes post-deployment monitoring for existing imaging AI model outputs. It does **not** train an imaging model. Bring a CSV of predictions, ground truth, and study metadata; the tool validates the file, computes performance and drift metrics, surfaces stop-rule alerts, and generates a downloadable monitoring report.

The project is motivated by the ACR-SIIM Practice Parameter for Imaging AI, approved by the ACR Council on May 5, 2026, which emphasizes AI inventory/version tracking, local acceptance testing, ongoing performance monitoring, drift/safety evaluation, stop rules, and privacy controls.

![rad-ai-sentinel demo walkthrough](screenshots/demo.gif)

## Evidence Status

| Evidence | Status |
|---|---|
| Unit and integration tests | Complete |
| Synthetic end-to-end evaluation | Complete |
| Public-data evaluation | Not completed |
| Independent expert review | Not completed |
| Institutional validation | Not completed |
| Prospective clinical validation | Not completed |

This software is a research prototype and is not intended for independent clinical decision-making.

## Capabilities

- Sensitivity, specificity, PPV, NPV, accuracy, F1, prevalence, AUROC, AUPRC
- Wilson confidence intervals for 2x2 metrics and bootstrap CIs for curve/calibration metrics
- Calibration analysis (Brier score, expected calibration error, reliability curves)
- Subgroup performance by age group, sex, race/ethnicity, site, scanner, modality
- Missing-data analysis and outcome-associated missingness flags
- Temporal drift detection (PSI, KL divergence, Wasserstein/KS score drift, rolling AUROC, CUSUM)
- Calibration slope/intercept and site-level calibration drift
- Configurable stop-rule alerts and model-version comparison (DeLong test)
- Monitoring-plan JSON, model inventory templates, alert review metadata, and audit-log events
- Public versus production schema profiles and de-identified PACS/RIS/AI orchestration export templates
- Label-based multi-class monitoring with per-class one-vs-rest metrics and confusion matrices
- Machine-readable CSV, JSON, HTML, and optional PDF exports
- Streamlit dashboard, CLI, synthetic data generator, RSNA public-data adapter

## Quick Start

```bash
pip install -e ".[dev]"

# Generate synthetic monitoring data, metrics, and an HTML report:
rad-ai-sentinel demo

# Analyze your own monitoring CSV:
rad-ai-sentinel compute --csv path/to/predictions.csv --output outputs/analysis

# Analyze a multi-class label CSV:
rad-ai-sentinel compute-multiclass --csv path/to/multiclass_predictions.csv --output outputs/multiclass

# Create governance and connector templates:
rad-ai-sentinel monitoring-plan-template monitoring_plan.json
rad-ai-sentinel inventory-template model_inventory.csv
rad-ai-sentinel connector-templates connector_examples
rad-ai-sentinel study-protocol-template study_protocol.json
rad-ai-sentinel study-protocol-validate study_protocol.json

# Launch the dashboard:
rad-ai-sentinel serve
```

Then open [http://localhost:8501](http://localhost:8501).

## Input CSV Format

Binary monitoring requires:

| Column | Description |
|---|---|
| `patient_id` | De-identified patient or case identifier |
| `study_date` | Examination date |
| `model_version` | Deployed model version |
| `y_true` | Binary ground-truth label, `0` or `1` |
| `y_pred_proba` | Predicted probability for class `1`, from `0` to `1` |
| `y_pred_binary` | Thresholded prediction, `0` or `1` |

Multi-class monitoring requires:

| Column | Description |
|---|---|
| `patient_id` | De-identified patient or case identifier |
| `study_date` | Examination date |
| `model_version` | Deployed model version |
| `y_true` | Ground-truth class label |
| `y_pred_label` | Predicted class label |

Both modes accept optional `site`, `scanner_manufacturer`, `modality`, `age_group`, `sex`, and `race_ethnicity` columns. The `production` schema profile requires `site`, `scanner_manufacturer`, and `modality`.

## Limitations

- Public radiology datasets generally do not include deployed model outputs; public smoke tests need either real model predictions or clearly labeled synthetic scores
- Multi-class monitoring currently reports label-based accuracy, macro/weighted metrics, per-class one-vs-rest metrics, and confusion matrices; probability-based multi-class calibration is not yet implemented
- Subgroup findings must be interpreted with sample-size and missingness context
- This project is not a medical device and does not certify clinical performance
- Not intended for patient-care decisions without institutional review, validation, and clinical oversight

## Documentation

| Topic | File |
|---|---|
| Product requirements | [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) |
| Architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Data sources (RSNA, MIMIC-CXR, NIH, CheXpert) | [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) |
| Desktop releases | [docs/DESKTOP_RELEASES.md](docs/DESKTOP_RELEASES.md) |
| Study protocol workflow | [docs/STUDY_PROTOCOL.md](docs/STUDY_PROTOCOL.md) |
| Roadmap | [docs/ROADMAP.md](docs/ROADMAP.md) |
| Research notes | [docs/RESEARCH.md](docs/RESEARCH.md) |
| CSV format reference | [README.md](README.md) (Input CSV Format section) |
| Contribution guide | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Security reporting | [SECURITY.md](SECURITY.md) |

## License

MIT. See [LICENSE](LICENSE).
