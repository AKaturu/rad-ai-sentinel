"""Tests for data generation, analysis orchestration, reports, and CLI adapters."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from rad_ai_sentinel.analysis import run_monitoring_analysis, write_analysis_outputs
from rad_ai_sentinel.cli import app
from rad_ai_sentinel.data import (
    adapt_rsna_pneumonia_labels,
    generate_synthetic_monitoring_data,
    write_rsna_case_study_template,
)
from rad_ai_sentinel.exports import build_monitoring_export_payloads
from rad_ai_sentinel.report import generate_monitoring_report, render_report_html
from rad_ai_sentinel.schemas import validate_dataframe


def _case_dir(name: str) -> Path:
    root = Path("outputs") / "test_product_surfaces" / name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_synthetic_monitoring_data_validates() -> None:
    df = generate_synthetic_monitoring_data(n=180, seed=7, include_version_holdout=False)
    validated = validate_dataframe(df)
    assert len(validated) == 180
    assert {"v1.0", "v2.0"} == set(validated["model_version"].unique())


def test_analysis_outputs_are_written() -> None:
    output_dir = _case_dir("analysis")
    df = generate_synthetic_monitoring_data(n=220, seed=11, include_version_holdout=False)
    analysis = run_monitoring_analysis(df, n_resamples=40)
    outputs = write_analysis_outputs(analysis, output_dir)
    assert outputs["summary"].exists()
    assert outputs["stratified"].exists()
    assert outputs["site_calibration"].exists()
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["n"] == len(df)
    assert payload["model_versions"] == ["v1.0", "v2.0"]


def test_html_report_renders() -> None:
    output_dir = _case_dir("report")
    df = generate_synthetic_monitoring_data(n=180, seed=13, include_version_holdout=False)
    analysis = run_monitoring_analysis(df, n_resamples=30)
    html = render_report_html(analysis)
    assert "rad-ai-sentinel Monitoring Report" in html
    artifacts = generate_monitoring_report(analysis, output_dir, include_pdf=False)
    assert artifacts.html.exists()
    assert artifacts.pdf is None


def test_monitoring_export_payloads_share_one_analysis() -> None:
    df = generate_synthetic_monitoring_data(n=180, seed=17, include_version_holdout=False)
    analysis = run_monitoring_analysis(df, n_resamples=30)
    payloads = build_monitoring_export_payloads(analysis)

    expected = {
        "validated_csv",
        "summary_csv",
        "stratified_csv",
        "missing_csv",
        "alerts_csv",
        "drift_csv",
        "site_calibration_csv",
        "versions_csv",
        "metrics_json",
        "report_html",
    }
    assert expected == set(payloads)

    metrics = json.loads(payloads["metrics_json"].data)
    assert metrics["n"] == len(df)
    assert "rad-ai-sentinel Monitoring Report" in payloads["report_html"].data
    assert "model_version" in payloads["validated_csv"].data


def test_rsna_adapter_creates_monitoring_csv() -> None:
    output_dir = _case_dir("rsna_adapter")
    labels = pd.DataFrame(
        {
            "patientId": ["a", "a", "b", "c"],
            "Target": [1, 1, 0, 1],
        }
    )
    labels_path = output_dir / "labels.csv"
    out_path = output_dir / "monitoring.csv"
    labels.to_csv(labels_path, index=False)

    out = adapt_rsna_pneumonia_labels(labels_path, out_path, seed=3)

    assert out_path.exists()
    assert len(out) == 3
    assert out["y_true"].tolist() == [1, 0, 1]
    validate_dataframe(out)


def test_cli_adapt_rsna() -> None:
    output_dir = _case_dir("cli_rsna")
    labels_path = output_dir / "labels.csv"
    output_path = output_dir / "monitoring.csv"
    pd.DataFrame({"patientId": ["p1", "p2"], "Target": [0, 1]}).to_csv(
        labels_path,
        index=False,
    )
    result = CliRunner().invoke(app, ["adapt-rsna", str(labels_path), str(output_path)])
    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_rsna_case_study_template_scaffold() -> None:
    output_dir = _case_dir("rsna_case_study_template")
    files = write_rsna_case_study_template(output_dir)

    assert files["readme"].exists()
    assert files["analysis_plan"].exists()
    predictions = pd.read_csv(files["predictions"])
    metadata = pd.read_csv(files["metadata"])
    assert list(predictions.columns) == ["patientId", "prediction"]
    assert {"patientId", "StudyDate", "PatientSex", "PatientAge"}.issubset(metadata.columns)
    assert "not a clinical validation template" in files["readme"].read_text(encoding="utf-8")


def test_cli_rsna_case_study_template_refuses_overwrite() -> None:
    output_dir = _case_dir("cli_rsna_case_study_template")
    result = CliRunner().invoke(app, ["rsna-case-study-template", str(output_dir)])
    assert result.exit_code == 0, result.output

    second = CliRunner().invoke(app, ["rsna-case-study-template", str(output_dir)])
    assert second.exit_code != 0
    assert "Refusing to overwrite" in second.output
