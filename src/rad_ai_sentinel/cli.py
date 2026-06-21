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
from .data import adapt_rsna_pneumonia_labels, write_synthetic_csv
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
) -> None:
    """Compute monitoring metrics and write machine-readable outputs."""
    df = load_and_validate_csv(csv)
    analysis = run_monitoring_analysis(df)
    outputs = write_analysis_outputs(analysis, output)
    _print_summary_table(summary_metrics_frame(analysis))
    console.print(f"[green]Wrote outputs to:[/green] {output}")
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
) -> None:
    """Generate a downloadable HTML report and optional PDF."""
    df = load_and_validate_csv(csv)
    analysis = run_monitoring_analysis(df)
    artifacts = generate_monitoring_report(
        analysis,
        output,
        basename=basename,
        include_pdf=pdf,
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
    raise typer.Exit(subprocess.run(cmd, check=False).returncode)


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


def _print_summary_table(df: pd.DataFrame) -> None:
    table = Table(title="rad-ai-sentinel summary")
    for col in df.columns:
        table.add_column(col)
    for row in df.astype(str).itertuples(index=False):
        table.add_row(*row)
    console.print(table)


if __name__ == "__main__":
    app()
