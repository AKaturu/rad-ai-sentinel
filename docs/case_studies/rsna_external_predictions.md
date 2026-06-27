# RSNA External-Prediction Case Study

Use this template when you want a public, reproducible monitoring example that
pairs RSNA Pneumonia Detection Challenge labels with predictions from a fixed
external model.

Generate the scaffold:

```bash
rad-ai-sentinel rsna-case-study-template docs/case_studies/rsna_external_predictions
```

The generated folder contains:

- `README.md`: setup instructions and claim-boundary language.
- `predictions_template.csv`: the expected `patientId,prediction` format.
- `metadata_template.csv`: optional de-identified metadata columns recognized by
  `adapt-rsna`.
- `analysis_plan.md`: a short methods checklist for reporting.

This workflow is for software demonstration and methods development. Do not
describe outputs as clinical validation, deployment monitoring, or regulatory
evidence unless a separate reviewed study supports those claims.
