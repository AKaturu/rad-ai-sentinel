# Roadmap

This roadmap tracks planned work after the initial public MVP. Items are grouped by release value rather than promised dates.

## Near Term

- Publish cross-platform desktop downloads for Windows, macOS, and Linux through GitHub Actions release artifacts.
- Add a short public case-study template that shows how to pair RSNA labels with externally generated model predictions without implying clinical validity.
- Expand dashboard export options for CSV, JSON, HTML, and PDF artifacts from the same reviewed analysis run.
- Add regression fixtures for malformed monitoring CSVs, missing subgroup fields, and small-sample subgroup behavior.

## Clinical AI Governance

- Add configurable monitoring plans that record model owner, intended use, operating threshold, review cadence, subgroup requirements, and stop-rule contacts.
- Support model inventory import/export so a site can track multiple deployed algorithms and versions.
- Add review-status metadata for alerts, including reviewer, disposition, notes, and follow-up action.
- Create a read-only audit log for generated reports and reviewed alert decisions.

## Analytics

- Add additional calibration summaries, including calibration slope/intercept and site-level calibration drift.
- Add confidence-interval controls for subgroup reporting and suppress unstable estimates below user-configured sample-size floors.
- Add richer temporal drift methods for model score distributions and input metadata.
- Support multi-class monitoring as a separate schema and analysis mode.

## Data Connectors

- Add de-identified adapter examples for common prediction-export patterns from PACS/RIS/AI orchestration systems.
- Add stricter schema profiles for public benchmark adaptation versus local production monitoring.
- Add optional FHIR/HL7-inspired metadata import helpers for sites that already maintain operational model registries.

## Release Hardening

- Sign or checksum release artifacts when the project starts publishing tagged releases.
- Add installation smoke tests for packaged desktop artifacts in CI.
- Add accessibility and responsive-layout checks for the dashboard.
- Publish versioned documentation pages once the command and CSV contracts stabilize.

## Out of Scope

- Training diagnostic models.
- Making patient-care decisions.
- Certifying regulatory compliance or replacing local clinical, physics, privacy, or AI governance review.
