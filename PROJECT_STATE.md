# PROJECT_STATE

## Project Overview

### Project Name
rad-ai-sentinel

### Goal
Post-deployment monitoring for radiology AI outputs, with metrics, drift, subgroup, governance, connector, and reporting surfaces for local review workflows.

### Current Status
Phase 7 (Validation) complete for the post-v0.1.0 governance, analytics, connector, and release-hardening pass.

---

## Completed Features

### Feature: Governance Controls

#### Validation
Verified with `tests/test_governance_connectors.py` and CLI smoke for `monitoring-plan-template`.

#### Tests Added
`tests/test_governance_connectors.py` covers monitoring-plan templates, inventory CSV/JSON loading, alert-review metadata, and audit-log events.

### Feature: Analytics And Schema Hardening

#### Validation
Verified with full pytest suite and focused analytics/schema tests.

#### Tests Added
`tests/test_metrics.py`, `tests/test_drift.py`, `tests/test_schemas.py`, and `tests/test_regression_fixtures.py` cover calibration slope/intercept, site calibration drift, richer score drift, schema profiles, and subgroup sample-size suppression.

### Feature: Data Connectors And Release Hardening

#### Validation
Verified connector adapter tests, dashboard guardrail script, and workflow/doc review.

#### Tests Added
`tests/test_governance_connectors.py` covers connector templates and export normalization. `scripts/check_dashboard_accessibility.py` is now run in CI.

---

## Current Work

### Active Feature
None.

### Progress
Implementation and validation complete for the selected roadmap slice.

### Remaining Work
External clinical validation activities remain evidence gaps and require public data/model predictions, expert review, institutional validation, or prospective clinical monitoring outside this repository.

---

## Next Actions

1. Decide whether to cut a post-v0.1.0 release tag for these changes.
2. Add multi-class monitoring as a separate schema and analysis mode.
3. Add richer drift methods for input metadata, not only model score distributions.

---

## Risks

### Open Questions
No code-blocking questions.

### Known Issues
Public-data evaluation, independent expert review, institutional validation, and prospective clinical validation are still not complete.

### Technical Concerns
The dashboard guardrail is intentionally lightweight and not a substitute for full browser-based accessibility testing.

---

## Resume Instructions

Start with `src/rad_ai_sentinel/analysis.py`, `src/rad_ai_sentinel/governance.py`, and `src/rad_ai_sentinel/data/connectors.py`. Verify with:

```bash
PYTHONPATH=src python -m pytest
python -m ruff check .
python scripts/check_dashboard_accessibility.py
```

On this Windows workstation, use a repo-local pytest temp directory if the user temp folder denies access:

```powershell
New-Item -ItemType Directory -Force -Path .pytest-tmp | Out-Null
$env:PYTHONPATH = "src"
$env:TEMP = (Resolve-Path .pytest-tmp).Path
$env:TMP = (Resolve-Path .pytest-tmp).Path
python -m pytest --basetemp .pytest-tmp\run
```
