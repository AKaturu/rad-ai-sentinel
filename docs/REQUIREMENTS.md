# Requirements

## Goal

Build `rad-ai-sentinel`, a runnable Python package for post-deployment radiology AI performance monitoring. The tool monitors existing model outputs; it does not train an imaging model.

## Functional Requirements

- Accept CSVs with model probabilities, binary predictions, ground truth, study dates, site/scanner/modality metadata, demographic subgroup fields when appropriately available, and model version.
- Validate the CSV contract and return actionable schema failures.
- Compute sensitivity, specificity, PPV, NPV, accuracy, F1, AUROC, AUPRC, Brier score, ECE, confidence intervals, and calibration curves.
- Stratify performance by site, scanner manufacturer, modality, age group, sex, and race/ethnicity.
- Analyze missing metadata and flag missingness associated with outcome labels.
- Detect temporal drift using score-distribution PSI, KL divergence, rolling AUROC, and CUSUM.
- Evaluate configurable stop-rule thresholds.
- Compare model versions, including paired DeLong AUROC testing when the same cases have scores from two versions.
- Provide a CLI, Streamlit dashboard, downloadable HTML report, optional PDF report, synthetic demo data, Dockerfile, and GitHub Actions workflow.
- Provide a path to use public sample radiology data for smoke testing without claiming the public labels are deployed AI results.

## Non-Functional Requirements

- Keep the repository runnable locally with `pip install -e ".[dev]"`.
- Keep data processing deterministic for demo/test fixtures.
- Avoid storing PHI or real clinical data in the repository.
- Make report outputs understandable for clinical informatics review while clearly stating that the project is not a medical device or certification tool.

## Acceptance Criteria

- `python -m pytest` passes.
- `rad-ai-sentinel demo` generates a synthetic CSV, metrics tables, and an HTML report.
- `rad-ai-sentinel compute --csv <file>` writes machine-readable outputs.
- `rad-ai-sentinel report --csv <file>` writes a downloadable report.
- `rad-ai-sentinel serve` launches the dashboard.
- README includes screenshots, a demo-video link, quick-start commands, and sample-data guidance.
- `STATUS.md` is current at handoff.

## Scope Boundaries

- In scope: monitoring existing predictions and labels.
- Out of scope: training, fine-tuning, hosting, or certifying an imaging model.
- Out of scope: storing patient-identifiable data, DICOM routing, PACS integration, authentication, and clinical deployment hardening.
