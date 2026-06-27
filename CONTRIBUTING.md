# Contributing

Thanks for helping improve rad-ai-sentinel. This project sits near clinical AI governance, so contributions should be conservative, testable, and clear about evidence boundaries.

## Development Setup

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quality Checks

Run before opening a pull request:

```bash
python -m ruff check .
python -m pytest
```

For user-facing changes, also run:

```bash
rad-ai-sentinel demo --output outputs/demo --n 1200 --seed 42
```

## Contribution Rules

- Do not commit PHI, credentials, institutional exports, private model outputs, or patient-level screenshots.
- Use synthetic data for examples and automated tests.
- Add tests for metric, drift, report, adapter, or schema changes.
- Document any new assumptions about thresholds, stop rules, missingness, or subgroup analysis.
- Avoid clinical-performance claims unless they are explicitly tied to validated data and stated limitations.

## Review Priorities

Maintainers will prioritize correctness, reproducibility, privacy, and clear separation between monitoring outputs and clinical decision-making.
