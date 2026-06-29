# Changelog

## 0.1.0 - 2026-06-28

### Added
- Binary classification metrics (sensitivity, specificity, PPV, NPV, accuracy, F1, prevalence)
- Curve metrics (AUROC, AUPRC) with bootstrap confidence intervals
- Calibration analysis (Brier score, expected calibration error, reliability curves)
- Subgroup analysis by age group, sex, race/ethnicity, site, scanner manufacturer, modality
- Missing-data analysis and outcome-associated missingness flags
- Temporal drift detection (PSI, KL divergence, rolling AUROC, CUSUM)
- Configurable stop-rule alerts
- Model-version comparison with DeLong AUROC test
- CSV, JSON, HTML, and optional PDF export formats
- Streamlit dashboard with interactive exploration and report downloads
- RSNA Pneumonia Detection Challenge adapter for public data smoke testing
- Synthetic data generator for demos and pipeline validation
- CLI commands (demo, compute, report, serve, adapt-rsna, rsna-case-study-template)
- Native desktop release packaging (Windows, macOS, Linux)
- Dockerfile for containerized deployment
- GitHub Actions CI (lint, test, demo smoke)
- Full test suite (schema validation, metrics, drift, product surfaces)
