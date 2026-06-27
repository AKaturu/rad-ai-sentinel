# PROJECT_STATE

## Project Overview

### Project Name
rad-ai-sentinel

### Goal
Build a runnable radiology AI performance monitoring framework that accepts existing model prediction CSVs, computes clinical AI surveillance metrics, detects drift and stop-rule breaches, stratifies by site/scanner/subgroup/version, and produces CLI, dashboard, and report outputs.

### Current Status
Phase 10 - MVP complete with cross-platform PDF export, tested, documented, and polished for public GitHub review.

---

## Completed Features

### Feature: Core Metrics and Validation

#### Validation
Validated with schema tests and metric tests covering binary metrics, confidence intervals, AUROC/AUPRC, calibration, stratification, and DeLong comparison.

#### Tests Added
`tests/test_schemas.py`, `tests/test_metrics.py`

### Feature: Drift, Missingness, Alerts, and Version Comparison

#### Validation
Validated with drift tests for PSI, KL divergence, rolling AUROC, CUSUM, stop-rule alerts, missing-data summaries, and version comparison.

#### Tests Added
`tests/test_drift.py`

### Feature: Product Surfaces

#### Validation
Validated synthetic data generation, analysis output writing, HTML report rendering, RSNA adapter behavior, and CLI adapter invocation.

#### Tests Added
`tests/test_product_surfaces.py`

### Feature: Dashboard and Media

#### Validation
Streamlit launched locally at `http://localhost:8501`. Browser automation verified the page title and AUROC metric, then captured desktop, subgroup, report, mobile, and animated walkthrough assets.

#### Tests Added
Manual/browser validation plus screenshots in `screenshots/`.

---

## Current Work

### Active Feature
None.

### Progress
All requested MVP surfaces are implemented: Python package, CLI, Streamlit dashboard, synthetic data, public-data adapter, report generation, Dockerfile, GitHub Actions, screenshots, and demo GIF.

GitHub presentation polish is complete: README badges and repository guide were added, contribution/security documents were added, and package/license metadata now points to `AKaturu/rad-ai-sentinel`.

### Remaining Work
No blocking work remains for the requested MVP.

---

## Next Actions

1. Review README screenshots and wording on GitHub after pushing.
2. PDF export now works on all platforms: WeasyPrint (Linux/Docker/CI with GTK) or fpdf2 fallback (Windows, no native deps). Run `rad-ai-sentinel demo` and both `.html` and `.pdf` are produced.
3. For a publication abstract, run `adapt-rsna` with RSNA labels plus predictions from a fixed external model, or use credentialed MIMIC-CXR/institutional predictions.

---

## Risks

### Open Questions
None blocking.

### Known Issues
None. Previously, Windows PDF export was broken because WeasyPrint requires GTK3 native libraries. This is now resolved with a dual-engine PDF strategy: WeasyPrint is tried first (for full CSS layout on Linux/Docker/CI), and fpdf2 (pure Python, no native deps) serves as an automatic fallback that produces a styled PDF with all metrics tables, plots, and KPI boxes.

### Technical Concerns
Public radiology datasets generally do not include deployed model outputs, so public smoke tests need either real model predictions or clearly labeled synthetic scores. Subgroup findings should be interpreted with sample-size and missingness context.

---

## Resume Instructions

Start with `README.md`, then `docs/ARCHITECTURE.md`. The shared orchestration entry point is `src/rad_ai_sentinel/analysis.py`; CLI commands live in `src/rad_ai_sentinel/cli.py`; the dashboard is `src/rad_ai_sentinel/app/main.py`; reports are in `src/rad_ai_sentinel/report/generate.py`.

Verify with:

```bash
python -m ruff check .
python -m pytest
rad-ai-sentinel demo --output outputs/demo --n 1200 --seed 42
rad-ai-sentinel serve
```

The single next concrete step, if continuing beyond this MVP, is to connect a real external model prediction CSV to the RSNA adapter and add a short case study under `docs/`.
