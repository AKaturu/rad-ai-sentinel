"""Command-line interface for rad-ai-sentinel."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .analysis import (
    load_and_validate_csv,
    run_monitoring_analysis,
    summary_metrics_frame,
    write_analysis_outputs,
)
from .data import (
    adapt_prediction_export,
    adapt_rsna_pneumonia_labels,
    write_connector_templates,
    write_rsna_case_study_template,
    write_synthetic_csv,
)
from .governance import (
    load_alert_reviews,
    load_model_inventory,
    load_monitoring_plan,
    write_model_inventory_template,
    write_monitoring_plan_template,
)
from .metrics.multiclass import multiclass_summary_frame
from .multiclass_analysis import (
    run_multiclass_monitoring_analysis,
    write_multiclass_analysis_outputs,
)
from .protocol import load_study_protocol, write_study_protocol_template
from .report import generate_monitoring_report

app = typer.Typer(
    name="rad-ai-sentinel",
    help="Post-deployment surveillance for radiology AI model outputs.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def demo(
    output: Annotated[Path, typer.Option(help="Directory for demo data and reports.")] = Path(
        "outputs/demo"
    ),
    n: Annotated[int, typer.Option(help="Number of synthetic production rows.")] = 1200,
    seed: Annotated[int, typer.Option(help="Random seed for reproducible data.")] = 42,
    pdf: Annotated[bool, typer.Option("--pdf/--no-pdf", help="Also attempt PDF export.")] = True,
) -> None:
    """Generate synthetic data and a complete monitoring report."""
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "synthetic_monitoring_data.csv"
    df = write_synthetic_csv(csv_path, n=n, seed=seed)
    analysis = run_monitoring_analysis(df)
    write_analysis_outputs(analysis, output)
    artifacts = generate_monitoring_report(analysis, output, include_pdf=pdf)
    console.print(f"[green]Wrote demo CSV:[/green] {csv_path}")
    console.print(f"[green]Wrote HTML report:[/green] {artifacts.html}")
    if artifacts.pdf:
        console.print(f"[green]Wrote PDF report:[/green] {artifacts.pdf}")
    elif artifacts.pdf_error:
        console.print(f"[yellow]PDF skipped; see:[/yellow] {artifacts.pdf_error}")


@app.command()
def compute(
    csv: Annotated[Path, typer.Option("--csv", help="Monitoring CSV to analyze.")],
    output: Annotated[Path, typer.Option(help="Directory for CSV/JSON outputs.")] = Path(
        "outputs/analysis"
    ),
    schema_profile: Annotated[
        str,
        typer.Option(help="Validation profile: public or production."),
    ] = "public",
    monitoring_plan: Annotated[
        Path | None,
        typer.Option(help="Optional monitoring-plan JSON file."),
    ] = None,
    alert_reviews: Annotated[
        Path | None,
        typer.Option(help="Optional alert-review CSV/JSON file."),
    ] = None,
    audit_log: Annotated[
        Path | None,
        typer.Option(help="Optional append-only audit log JSONL path."),
    ] = None,
) -> None:
    """Compute monitoring metrics and write machine-readable outputs."""
    plan = load_monitoring_plan(monitoring_plan) if monitoring_plan else None
    reviews = load_alert_reviews(alert_reviews) if alert_reviews else ()
    df = load_and_validate_csv(csv, profile=schema_profile)
    analysis = run_monitoring_analysis(
        df,
        monitoring_plan=plan,
        schema_profile=schema_profile,
        alert_reviews=reviews,
    )
    outputs = write_analysis_outputs(analysis, output, audit_log=audit_log)
    _print_summary_table(summary_metrics_frame(analysis))
    console.print(f"[green]Wrote outputs to:[/green] {output}")
    for name, path in outputs.items():
        console.print(f"  {name}: {path}")


@app.command("compute-multiclass")
def compute_multiclass_command(
    csv: Annotated[Path, typer.Option("--csv", help="Multi-class monitoring CSV to analyze.")],
    output: Annotated[Path, typer.Option(help="Directory for CSV/JSON outputs.")] = Path(
        "outputs/multiclass"
    ),
    schema_profile: Annotated[
        str,
        typer.Option(help="Validation profile: public or production."),
    ] = "public",
    audit_log: Annotated[
        Path | None,
        typer.Option(help="Optional append-only audit log JSONL path."),
    ] = None,
) -> None:
    """Compute label-based multi-class metrics and write machine-readable outputs."""
    analysis = run_multiclass_monitoring_analysis(
        pd.read_csv(csv),
        schema_profile=schema_profile,
    )
    outputs = write_multiclass_analysis_outputs(analysis, output, audit_log=audit_log)
    _print_summary_table(multiclass_summary_frame(analysis.metrics))
    console.print(f"[green]Wrote multi-class outputs to:[/green] {output}")
    for name, path in outputs.items():
        console.print(f"  {name}: {path}")


@app.command("report")
def report_command(
    csv: Annotated[Path, typer.Option("--csv", help="Monitoring CSV to report on.")],
    output: Annotated[Path, typer.Option(help="Directory for report artifacts.")] = Path(
        "outputs/report"
    ),
    basename: Annotated[str, typer.Option(help="Report filename without extension.")] = (
        "rad_ai_sentinel_report"
    ),
    pdf: Annotated[bool, typer.Option("--pdf/--no-pdf", help="Also attempt PDF export.")] = True,
    schema_profile: Annotated[
        str,
        typer.Option(help="Validation profile: public or production."),
    ] = "public",
    monitoring_plan: Annotated[
        Path | None,
        typer.Option(help="Optional monitoring-plan JSON file."),
    ] = None,
    alert_reviews: Annotated[
        Path | None,
        typer.Option(help="Optional alert-review CSV/JSON file."),
    ] = None,
    audit_log: Annotated[
        Path | None,
        typer.Option(help="Optional append-only audit log JSONL path."),
    ] = None,
) -> None:
    """Generate a downloadable HTML report and optional PDF."""
    plan = load_monitoring_plan(monitoring_plan) if monitoring_plan else None
    reviews = load_alert_reviews(alert_reviews) if alert_reviews else ()
    df = load_and_validate_csv(csv, profile=schema_profile)
    analysis = run_monitoring_analysis(
        df,
        monitoring_plan=plan,
        schema_profile=schema_profile,
        alert_reviews=reviews,
    )
    artifacts = generate_monitoring_report(
        analysis,
        output,
        basename=basename,
        include_pdf=pdf,
        audit_log=audit_log,
    )
    console.print(f"[green]Wrote HTML report:[/green] {artifacts.html}")
    if artifacts.pdf:
        console.print(f"[green]Wrote PDF report:[/green] {artifacts.pdf}")
    elif artifacts.pdf_error:
        console.print(f"[yellow]PDF skipped; see:[/yellow] {artifacts.pdf_error}")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Streamlit server host.")] = "localhost",
    port: Annotated[int, typer.Option(help="Streamlit server port.")] = 8501,
) -> None:
    """Launch the Streamlit dashboard."""
    app_path = Path(__file__).parent / "app" / "main.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    raise typer.Exit(code=subprocess.run(cmd, check=False).returncode)


@app.command("adapt-rsna")
def adapt_rsna_command(
    labels_csv: Annotated[Path, typer.Argument(help="RSNA stage labels CSV.")],
    output_csv: Annotated[Path, typer.Argument(help="Destination monitoring CSV.")],
    predictions_csv: Annotated[
        Path | None,
        typer.Option(help="Optional patientId/prediction CSV from an existing model."),
    ] = None,
    metadata_csv: Annotated[
        Path | None,
        typer.Option(help="Optional patient-level DICOM metadata extract."),
    ] = None,
    threshold: Annotated[float, typer.Option(help="Threshold for y_pred_binary.")] = 0.5,
    model_version: Annotated[str, typer.Option(help="Model version label.")] = "external-model",
) -> None:
    """Adapt public RSNA Pneumonia Detection labels into the monitor schema."""
    df = adapt_rsna_pneumonia_labels(
        labels_csv,
        output_csv,
        predictions_csv=predictions_csv,
        metadata_csv=metadata_csv,
        threshold=threshold,
        model_version=model_version,
    )
    console.print(f"[green]Wrote monitoring CSV:[/green] {output_csv} ({len(df)} rows)")


@app.command("rsna-case-study-template")
def rsna_case_study_template_command(
    output: Annotated[
        Path,
        typer.Argument(help="Destination directory for the RSNA case-study scaffold."),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing scaffold files."),
    ] = False,
) -> None:
    """Write a safe RSNA external-prediction case-study scaffold."""
    try:
        files = write_rsna_case_study_template(output, force=force)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Wrote RSNA case-study template:[/green] {output}")
    for name, path in files.items():
        console.print(f"  {name}: {path}")


@app.command("adapt-export")
def adapt_export_command(
    export_csv: Annotated[Path, typer.Argument(help="De-identified prediction export CSV.")],
    output_csv: Annotated[Path, typer.Argument(help="Destination monitoring CSV.")],
    metadata_csv: Annotated[
        Path | None,
        typer.Option(help="Optional operational metadata CSV."),
    ] = None,
    threshold: Annotated[float, typer.Option(help="Threshold for y_pred_binary.")] = 0.5,
    model_version: Annotated[str, typer.Option(help="Default model version if absent.")] = (
        "external-model"
    ),
    schema_profile: Annotated[
        str,
        typer.Option(help="Validation profile: public or production."),
    ] = "production",
) -> None:
    """Adapt a de-identified PACS/RIS/orchestration export into the monitor schema."""
    df = adapt_prediction_export(
        export_csv,
        output_csv,
        metadata_csv=metadata_csv,
        threshold=threshold,
        model_version=model_version,
        schema_profile=schema_profile,
    )
    console.print(f"[green]Wrote monitoring CSV:[/green] {output_csv} ({len(df)} rows)")


@app.command("connector-templates")
def connector_templates_command(
    output: Annotated[
        Path,
        typer.Argument(help="Destination directory for connector example templates."),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite existing template files."),
    ] = False,
) -> None:
    """Write de-identified connector templates for common operational exports."""
    try:
        files = write_connector_templates(output, force=force)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Wrote connector templates:[/green] {output}")
    for name, path in files.items():
        console.print(f"  {name}: {path}")


@app.command("monitoring-plan-template")
def monitoring_plan_template_command(
    output: Annotated[Path, typer.Argument(help="Destination monitoring-plan JSON.")],
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing file.")] = False,
) -> None:
    """Write an editable monitoring-plan template."""
    try:
        path = write_monitoring_plan_template(output, force=force)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Wrote monitoring plan template:[/green] {path}")


@app.command("inventory-template")
def inventory_template_command(
    output: Annotated[Path, typer.Argument(help="Destination inventory CSV or JSON.")],
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing file.")] = False,
) -> None:
    """Write an editable model-inventory template."""
    try:
        path = write_model_inventory_template(output, force=force)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Wrote model inventory template:[/green] {path}")


@app.command("inventory-validate")
def inventory_validate_command(
    inventory: Annotated[Path, typer.Argument(help="Model inventory CSV or JSON.")],
) -> None:
    """Validate and summarize a model inventory."""
    items = load_model_inventory(inventory)
    console.print(f"[green]Loaded model inventory:[/green] {len(items)} model(s)")
    for item in items:
        console.print(f"  {item.model_id} {item.version}: {item.status}")


@app.command("study-protocol-template")
def study_protocol_template_command(
    output: Annotated[Path, typer.Argument(help="Destination study protocol JSON.")],
    force: Annotated[bool, typer.Option("--force", help="Overwrite existing file.")] = False,
) -> None:
    """Write an editable prospective validation protocol template."""
    try:
        path = write_study_protocol_template(output, force=force)
    except FileExistsError as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Wrote study protocol template:[/green] {path}")


@app.command("study-protocol-validate")
def study_protocol_validate_command(
    protocol: Annotated[Path, typer.Argument(help="Study protocol JSON to validate.")],
) -> None:
    """Validate and summarize a prospective study protocol."""
    loaded = load_study_protocol(protocol)
    console.print(f"[green]Loaded study protocol:[/green] {loaded.study_id}")
    console.print(f"  Primary endpoint: {loaded.primary_endpoint}")
    console.print(f"  Minimum cases: {loaded.minimum_cases}")
    console.print(f"  Drift methods: {', '.join(loaded.drift_methods)}")


def _print_summary_table(df: pd.DataFrame) -> None:
    table = Table(title="rad-ai-sentinel summary")
    for col in df.columns:
        table.add_column(col)
    for row in df.astype(str).itertuples(index=False):
        table.add_row(*row)
    console.print(table)


if __name__ == "__main__":
    app()
