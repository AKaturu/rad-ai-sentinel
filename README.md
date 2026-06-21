# rad-ai-sentinel

**An open-source framework for site-, scanner-, subgroup- and time-stratified surveillance of radiology AI performance.**

`rad-ai-sentinel` operationalizes the post-deployment monitoring that the
**[ACR-SIIM Practice Parameter for Imaging AI](https://www.acr.org/News-and-Publications/Media-Center/2026/first-practice-parameter-for-imaging-ai)**
(approved by the ACR Council, May 2026) now requires of imaging facilities:
monitoring real-world model performance for **drift** and safety issues, defining
**stop rules**, and keeping an inventory of AI tools including their **versions**.

You do **not** train an imaging model. You give `rad-ai-sentinel` the predictions
an existing AI model already made, alongside the ground truth and study metadata,
and it produces a complete surveillance report.

---

## What it does

Upload a CSV containing a model's predictions and ground truth plus study
metadata, and `rad-ai-sentinel` produces:

- **Discrimination**: sensitivity, specificity, PPV, NPV (with Wilson confidence
  intervals), AUROC and AUPRC (with bootstrap CIs).
- **Calibration**: Brier score, Expected Calibration Error (ECE), and a
  reliability diagram.
- **Subgroup performance**: stratified by age group, sex, race/ethnicity.
- **Scanner- and site-specific performance**: stratified by site, scanner
  manufacturer, and modality.
- **Missing-data analysis**: missingness rates and per-subgroup impact.
- **Temporal drift detection**: Population Stability Index (PSI), KL divergence,
  rolling AUROC, and a CUSUM signal.
- **Threshold / stop-rule alerts**: configurable breaches that fire ACR-style
  stop rules.
- **Model-version comparison**: metric deltas with DeLong testing on AUROC.
- **A downloadable AI monitoring report** in both HTML and PDF.

## Quick start

```bash
# install (development)
pip install -e ".[dev]"

# run the demo (generates synthetic data with planted issues + a full report)
rad-ai-sentinel demo

# compute metrics from your own CSV
rad-ai-sentinel compute --csv path/to/predictions.csv --output reports/

# generate a PDF + HTML report
rad-ai-sentinel report --csv path/to/predictions.csv --output reports/

# launch the web dashboard
rad-ai-sentinel serve
```

Then open http://localhost:8501.

## CSV format

| column | type | required | description |
|---|---|---|---|
| `patient_id` | string | yes | de-identified patient identifier |
| `study_date` | date | yes | examination date (YYYY-MM-DD) |
| `site` | string | no | reading site / facility |
| `scanner_manufacturer` | string | no | e.g. GE, Siemens, Philips |
| `modality` | string | no | e.g. CR, DX, CT, MR |
| `age_group` | string | no | e.g. 0-17, 18-39, 40-64, 65+ |
| `sex` | string | no | e.g. F, M |
| `race_ethnicity` | string | no | self-reported, where available |
| `model_version` | string | yes | e.g. v2.0, v2.1 |
| `y_true` | int (0/1) | yes | ground-truth label |
| `y_pred_proba` | float [0,1] | yes | model probability for class 1 |
| `y_pred_binary` | int (0/1) | yes | thresholded model prediction |

Optional metadata columns may be missing; `rad-ai-sentinel` reports and accounts
for missingness. See [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) for how to
connect real datasets (MIMIC-CXR, CheXpert, NIH ChestX-ray14) once you have DUA
access.

## License

MIT — see [LICENSE](LICENSE).
