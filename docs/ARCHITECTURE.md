# Architecture

## Components

- `schemas`: validates binary and multi-class CSV contracts with `pandera`.
- `metrics`: computes binary metrics, multi-class metrics, curves, calibration, confidence intervals, stratification, and DeLong comparisons.
- `drift`: computes PSI, KL divergence, rolling AUROC, CUSUM, missing-data analysis, alerts, and version comparisons.
- `data`: generates synthetic monitoring data and adapts RSNA, PACS/RIS, and AI orchestration exports into the monitoring schema.
- `governance`: loads monitoring plans, model inventories, and alert review metadata.
- `audit`: appends checksum-backed JSONL events and reads audit logs without mutation.
- `analysis`: orchestrates binary validation, metrics, drift, missingness, alerts, version comparisons, and output tables.
- `multiclass_analysis`: orchestrates label-based multi-class validation, metrics, confusion matrices, and output tables.
- `plots`: creates interactive Plotly figures for the dashboard.
- `report`: renders HTML and optional PDF reports from the shared analysis object.
- `cli`: exposes `demo`, `compute`, `compute-multiclass`, `report`, `serve`, governance templates, inventory helpers, and data adapters.
- `app`: Streamlit dashboard.

## Data Flow

```mermaid
flowchart LR
  CSV["Monitoring CSV"] --> Schema["Schema validation"]
  Schema --> Analysis["run_monitoring_analysis"]
  Analysis --> Metrics["Metrics and calibration"]
  Analysis --> Drift["Drift and alerts"]
  Analysis --> Missing["Missing data"]
  Analysis --> Versions["Version comparison"]
  Analysis --> CLI["CLI outputs"]
  Analysis --> Report["HTML/PDF report"]
  Analysis --> Dashboard["Streamlit dashboard"]
```

## Interfaces

- `run_monitoring_analysis(df) -> MonitoringAnalysis`
- `write_analysis_outputs(analysis, output_dir) -> dict[str, Path]`
- `run_multiclass_monitoring_analysis(df) -> MulticlassMonitoringAnalysis`
- `write_multiclass_analysis_outputs(analysis, output_dir) -> dict[str, Path]`
- `generate_monitoring_report(data, output_dir) -> ReportArtifacts`
- `generate_synthetic_monitoring_data(n, seed) -> pd.DataFrame`
- `adapt_rsna_pneumonia_labels(labels_csv, output_csv, ...) -> pd.DataFrame`
- `adapt_prediction_export(export_csv, output_csv, ...) -> pd.DataFrame`
- `load_monitoring_plan(path) -> MonitoringPlan`
- `load_model_inventory(path) -> list[ModelInventoryItem]`
- `read_audit_log(path) -> tuple[dict, ...]`

## Failure Modes

- Missing required columns: schema validation fails before analysis.
- Optional metadata missing: analysis continues and reports missingness.
- Single-class windows: AUROC is omitted for that window rather than crashing.
- PDF system dependency missing: HTML report is still produced and a PDF error file is written.
- Public data without predictions: RSNA adapter marks generated scores as synthetic pipeline-test predictions by requiring an explicit command path rather than silently treating them as real deployment data.
- Public/production context mismatch: the public schema profile tolerates sparse benchmark metadata; the production profile requires operational site, scanner, and modality fields.
- Audit traceability: generated report/analysis events write artifact checksums, not row-level clinical data.
- Multi-class inputs: routed through a separate label schema and output path so binary stop-rule alerts are not applied to class-label tasks by accident.
