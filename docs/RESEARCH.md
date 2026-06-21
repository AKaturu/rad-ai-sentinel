# Research Summary

## Domain Context

The American College of Radiology announced the first ACR-SIIM Practice Parameter for Imaging AI on May 5, 2026. The announcement emphasizes AI governance, inventory/version tracking, local acceptance testing, monitoring for drift and safety issues, stop rules, and privacy controls. `rad-ai-sentinel` maps those themes to a lightweight open-source monitoring workflow.

## Public Sample Data Options

### Best Immediate Smoke Test: RSNA Pneumonia Detection Challenge

RSNA's 2018 Pneumonia Detection Challenge provides labels and downloads for a 30,000-image chest radiograph subset derived from the NIH CXR8 public dataset. Its dataset description notes DICOM format and DICOM tags including patient sex, patient age, and projection. The data does not include deployed AI outputs, so this project includes `rad-ai-sentinel adapt-rsna` to combine the labels with external model predictions or deterministic synthetic scores for pipeline testing.

### Rich Institutional Research Option: MIMIC-CXR

MIMIC-CXR is a de-identified BIDMC chest radiograph database with DICOM images and free-text radiology reports. It is suitable for institutional-style research once the user completes PhysioNet credentialing and data-use requirements. It is not bundled because access is controlled and the files are large.

### Additional Options

- NIH Chest X-ray dataset: open de-identified images from the NIH Clinical Center with attribution requirements.
- CheXpert: Stanford chest radiograph dataset with uncertainty labels and radiologist-labeled reference sets; useful for external validation research, subject to access terms.

## Technology Choices

- `pandas` for tabular ingestion and aggregation.
- `pandera` for explicit input schema validation.
- `scikit-learn` for AUROC, AUPRC, calibration, and base metrics.
- `scipy` and NumPy for confidence intervals, drift statistics, and DeLong support.
- `plotly` for interactive dashboard figures.
- `matplotlib` for static report figures that can render into HTML/PDF without browser export dependencies.
- `jinja2` for report templating.
- `typer` and `rich` for a readable CLI.
- `streamlit` for a fast MVP dashboard.
- `weasyprint` for optional PDF generation when system libraries are available.

## Known Constraints

- Public radiology datasets generally contain images, labels, and metadata, not post-deployment AI outputs. Any public-data smoke test needs an existing model's predictions or explicitly synthetic predictions.
- Demographic subgroup analysis should be reported with missingness and sample-size context; the tool should not imply fairness conclusions from underpowered strata.
- PDF export can depend on system graphics libraries, so HTML is the guaranteed report artifact.
