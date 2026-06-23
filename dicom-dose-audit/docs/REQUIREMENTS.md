# Requirements

## Goal

An open-source CT radiation-dose audit tool that reads CTDIvol and DLP from
DICOM metadata (CT image headers) or CT Radiation Dose Structured Reports
(RDSR), groups studies by protocol / scanner / patient-size category, detects
missing dose metadata, flags statistical outliers, compares protocol versions,
plots monthly trends, and generates quality-improvement reports.

## Functional requirements

1. **Ingest dose data** from three sources into one normalized contract:
   - CT Image DICOM headers (`CTDIvol` (0018,9345), `DLP` (0018,9934), plus
     protocol, scanner, kVp, tube current, study date).
   - CT Radiation Dose SR (RDSR) content trees (mean CTDIvol, DLP, scanned
     length per irradiation event).
   - A clean CSV dose log matching the documented contract.
2. **Group studies** by protocol, scanner (model + manufacturer), site, and
   patient-size category, with per-group summary statistics (n, missing, mean,
   median, P25/P75/P90, std, min, max).
3. **Detect missing dose metadata**: per-column missingness and per-study
   missing-CTDIvol / missing-DLP flags.
4. **Flag statistical outliers** per protocol using robust methods (Tukey IQR
   fence + MAD modified z-score cross-check), with a minimum-group-size guard.
5. **Compare protocol versions**: mean/median CTDIvol and DLP delta between
   versions of the same protocol, with a bootstrap confidence interval on the
   mean difference.
6. **Plot monthly trends**: median CTDIvol and DLP per protocol per month.
7. **Generate a quality-improvement report** in HTML and optional PDF
   (WeasyPrint primary, fpdf2 pure-Python fallback).

## Non-functional requirements

- Schema-validated inputs (pandera, lazy collection of all errors).
- Deterministic synthetic data for tests and demo (no real patient data).
- Cross-platform PDF (Windows-safe fallback).
- CLI + Streamlit dashboard.
- `ruff check .` clean; `pytest` green.

## Acceptance criteria

- `dicom-dose-audit demo` parses synthetic DICOM files, runs the full audit,
  and writes CSV/JSON outputs plus an HTML and PDF report.
- Synthetic CT image and RDSR DICOM datasets round-trip through the reader.
- Outlier detection flags planted high outliers and respects min-group-size.
- Protocol-version comparison reports a delta with CI when versions coexist.
- Monthly trend series contains one row per protocol per month.

## Scope boundaries (safety)

The public tool identifies **statistical outliers** only. It does NOT declare
any study clinically unsafe, and it is not a medical device. Diagnostic
reference levels and clinical thresholds require institutional review,
validated benchmarks, and qualified medical-physics oversight.
