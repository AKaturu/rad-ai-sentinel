# Study Protocol Workflow

Use a locked study protocol before analyzing external public data, institutionally governed exports, or prospective monitoring runs. The protocol is intentionally machine-readable so it can be archived with the evidence package and checked during review.

## Create a Template

```bash
rad-ai-sentinel study-protocol-template study_protocol.json
```

Edit the placeholders before analysis:

- `study_id` and `title`
- public or institutional `data_source`
- frozen `prediction_source`
- primary and secondary endpoints
- minimum case count
- drift methods and subgroup requirements
- alert-threshold strategy
- reviewer roles and registration URL

## Validate Before Analysis

```bash
rad-ai-sentinel study-protocol-validate study_protocol.json
```

Archive the protocol SHA-256 with the analysis outputs. If the protocol changes after results are inspected, create a new version and document the reason.

## Publication Guardrails

- Public-data runs with simulated scores are software demonstrations, not clinical validation.
- Real model outputs should be frozen before outcome analysis.
- Alert thresholds should be pre-specified or derived only from the training/baseline period.
- Subgroup findings require sample-size context and independent review before operational use.
- This project does not certify clinical performance and is not a medical device.
