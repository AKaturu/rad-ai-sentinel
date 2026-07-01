"""Tests for label-based multi-class monitoring."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from rad_ai_sentinel.cli import app
from rad_ai_sentinel.config import (
    COL_MODEL_VERSION,
    COL_PATIENT_ID,
    COL_STUDY_DATE,
    COL_Y_PRED_LABEL,
    COL_Y_TRUE,
)
from rad_ai_sentinel.metrics.multiclass import (
    compute_multiclass_metrics,
    multiclass_confusion_frame,
    multiclass_per_class_frame,
    multiclass_summary_frame,
)
from rad_ai_sentinel.multiclass_analysis import (
    run_multiclass_monitoring_analysis,
    write_multiclass_analysis_outputs,
)
from rad_ai_sentinel.schemas import validate_multiclass_dataframe


def _multiclass_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_PATIENT_ID: [f"p{i:03d}" for i in range(9)],
            COL_STUDY_DATE: pd.date_range("2026-01-01", periods=9, freq="D"),
            COL_MODEL_VERSION: ["triage-v1"] * 9,
            COL_Y_TRUE: [
                "normal",
                "pneumonia",
                "edema",
                "normal",
                "edema",
                "pneumonia",
                "normal",
                "edema",
                "pneumonia",
            ],
            COL_Y_PRED_LABEL: [
                "normal",
                "pneumonia",
                "normal",
                "normal",
                "edema",
                "edema",
                "pneumonia",
                "edema",
                "pneumonia",
            ],
            "site": ["A", "A", "B", "B", "A", "A", "B", "B", "A"],
            "scanner_manufacturer": ["GE"] * 9,
            "modality": ["DX"] * 9,
        }
    )


def test_multiclass_schema_validates_label_contract() -> None:
    out = validate_multiclass_dataframe(_multiclass_df())
    assert out[COL_Y_PRED_LABEL].tolist()[0] == "normal"
    assert pd.api.types.is_datetime64_any_dtype(out[COL_STUDY_DATE])


def test_multiclass_schema_rejects_single_label_space() -> None:
    df = _multiclass_df()
    df[COL_Y_TRUE] = "normal"
    df[COL_Y_PRED_LABEL] = "normal"
    with pytest.raises(ValueError, match="at least two distinct"):
        validate_multiclass_dataframe(df)


def test_compute_multiclass_metrics() -> None:
    df = _multiclass_df()
    metrics = compute_multiclass_metrics(df[COL_Y_TRUE], df[COL_Y_PRED_LABEL])

    assert metrics.labels == ("edema", "normal", "pneumonia")
    assert metrics.n == 9
    assert metrics.accuracy == pytest.approx(6 / 9)
    assert metrics.balanced_accuracy == pytest.approx(2 / 3)
    assert metrics.macro_f1 == pytest.approx(2 / 3)

    normal = next(item for item in metrics.per_class if item.label == "normal")
    assert normal.support == 3
    assert normal.true_positive == 2
    assert normal.false_positive == 1
    assert normal.false_negative == 1
    assert normal.specificity == pytest.approx(5 / 6)

    summary = multiclass_summary_frame(metrics)
    per_class = multiclass_per_class_frame(metrics)
    confusion = multiclass_confusion_frame(metrics)
    assert "Macro F1" in summary["metric"].tolist()
    assert set(per_class["class_label"]) == {"normal", "pneumonia", "edema"}
    assert "predicted_normal" in confusion.columns


def test_multiclass_outputs_and_cli(tmp_path: Path) -> None:
    df = _multiclass_df()
    csv_path = tmp_path / "multiclass.csv"
    out_dir = tmp_path / "outputs"
    audit_log = tmp_path / "audit.jsonl"
    df.to_csv(csv_path, index=False)

    analysis = run_multiclass_monitoring_analysis(df)
    outputs = write_multiclass_analysis_outputs(analysis, out_dir, audit_log=audit_log)
    assert outputs["summary"].exists()
    assert outputs["per_class"].exists()
    assert outputs["confusion"].exists()
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["n"] == 9
    assert payload["model_versions"] == ["triage-v1"]
    assert audit_log.exists()

    cli_out = tmp_path / "cli"
    result = CliRunner().invoke(
        app,
        ["compute-multiclass", "--csv", str(csv_path), "--output", str(cli_out)],
    )
    assert result.exit_code == 0, result.output
    assert (cli_out / "multiclass_metrics_summary.json").exists()


def test_multiclass_json_handles_sparse_label_window(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            COL_PATIENT_ID: ["p001"],
            COL_STUDY_DATE: ["2026-01-01"],
            COL_MODEL_VERSION: ["triage-v1"],
            COL_Y_TRUE: ["normal"],
            COL_Y_PRED_LABEL: ["edema"],
        }
    )
    analysis = run_multiclass_monitoring_analysis(df)
    outputs = write_multiclass_analysis_outputs(analysis, tmp_path)

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["n"] == 1
    assert payload["balanced_accuracy"] == 0.0
    assert any(item["specificity"] is None for item in payload["per_class"])
