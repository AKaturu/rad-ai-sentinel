# Research

## DICOM CT dose fundamentals

- **CTDIvol** (volume CT Dose Index): the standard per-scan dose metric in
  **mGy** (IEC 60601-2-44). Stored in CT image headers at tag
  `(0018,9345)`.
- **DLP** (Dose-Length Product) = CTDIvol × scan length, in **mGy·cm**. It is
  often NOT a standalone image-header tag; it may appear at `(0018,9934)` in
  some enhanced objects but is most reliably inside the RDSR content tree.
- **DRL** (Diagnostic Reference Level): typically the 75th percentile of
  observed dose per protocol from a national/local survey. An advisory
  optimization target, **not** a safety limit.
- **SSDE** (Size-Specific Dose Estimate): CTDIvol corrected for patient size
  via water-equivalent diameter. We stratify by a `size_category` label rather
  than computing SSDE.

## Where dose lives

| Source | CTDIvol | DLP | Notes |
|---|---|---|---|
| CT Image header | `(0018,9345)` | `(0018,9934)` (sometimes) | Subset; may be absent. |
| RDSR (CT Radiation Dose SR) | content item, DCM code `113838` (Mean CTDIvol) | content item, DCM code `113814` (DLP) | Superset; per irradiation event. |

RDSR encodes dose as coded **content items** inside a recursive
`ContentSequence`, not flat attributes — so parsing requires walking the SR
document tree and matching concept-name code values.

## Library choices

- **pydicom 3.x**: read CT image headers and traverse RDSR content sequences.
  Supports building synthetic Datasets in memory for deterministic test data.
- **pandera**: lazy schema validation (matches the sibling project).
- **matplotlib + plotly**: static report plots + interactive dashboard.
- **WeasyPrint + fpdf2**: dual-engine PDF (WeasyPrint primary with GTK; fpdf2
  pure-Python fallback for Windows/CI), matching the proven sibling strategy.

## Outlier methodology

- **Tukey IQR fence**: flag values outside `[Q1 − k·IQR, Q3 + k·IQR]`, `k=1.5`
  (standard) / `3.0` (far). Robust to skew because it is percentile-based.
- **MAD modified z-score**: `0.6745·(x − median) / MAD`; flag ≥ 3.5. A
  cross-check for heavy-tailed protocol groups where IQR can be degenerate.
- **Minimum group size**: require `n ≥ 8` per protocol before flagging — small
  groups make fence estimates unreliable.
- These flag **statistical** outliers, not clinical unsafety.

## Risks

- RDSR structure varies across vendors/versions → parser must be tolerant and
  fall back to image headers. Partial reads are reported, not fatal.
- Real patient DICOM must never ship → all test/demo data is synthesized with
  pydicom in memory.
- DLP frequently missing from image headers → missing-dose detection is a core
  feature, and DLP can optionally be reconstructed as CTDIvol × scan length.
