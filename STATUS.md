# Status

## Current Release
**v0.1.0** (2026-06-28) — MVP release.

## Implemented Features
- Binary classification metrics: sensitivity, specificity, PPV, NPV, accuracy, F1, prevalence
- Curve metrics: AUROC, AUPRC with bootstrap confidence intervals
- Calibration analysis: Brier score, expected calibration error, reliability curves
- Subgroup analysis by age, sex, race/ethnicity, site, scanner, modality
- Missing-data analysis and outcome-associated missingness flags
- Temporal drift detection: PSI, KL divergence, rolling AUROC, CUSUM
- Configurable stop-rule alerts
- Model-version comparison including DeLong AUROC test
- CSV, JSON, HTML, and optional PDF exports
- Streamlit dashboard with interactive exploration
- RSNA Pneumonia Detection Challenge adapter for public data smoke testing
- Synthetic data generator for demos and pipeline validation
- Native desktop release packaging (Windows, macOS, Linux)

## Validation Status
- **Unit tests**: Pass (core metrics, drift, schema, product-surface tests)
- **Synthetic end-to-end test**: Complete (demo pipeline generates synthetic data, runs metrics, produces report)
- **Public-data evaluation**: Not completed (RSNA adapter available for public smoke test but no published evaluation against a clinical benchmark)
- **Expert review**: Not completed
- **Institutional validation**: Not completed
- **Prospective clinical validation**: Not completed

## Planned Work
- Configurable monitoring-plan file (model owner, intended use, thresholds, review cadence)
- Expanded public data-source adapters
- Multi-model comparison dashboards
- External validation with clinical AI monitoring data
