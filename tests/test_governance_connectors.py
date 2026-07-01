"""Tests for governance artifacts, audit logging, and connector helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rad_ai_sentinel.analysis import alerts_frame, run_monitoring_analysis, write_analysis_outputs
from rad_ai_sentinel.audit import read_audit_log
from rad_ai_sentinel.config import AlertThresholds
from rad_ai_sentinel.data import adapt_prediction_export, write_connector_templates
from rad_ai_sentinel.governance import (
    ModelInventoryItem,
    load_alert_reviews,
    load_model_inventory,
    load_monitoring_plan,
    save_model_inventory,
    write_model_inventory_template,
    write_monitoring_plan_template,
)
from rad_ai_sentinel.protocol import (
    StudyProtocol,
    load_study_protocol,
    write_study_protocol_template,
)
from rad_ai_sentinel.schemas import validate_dataframe


def test_monitoring_plan_template_loads(tmp_path: Path) -> None:
    path = write_monitoring_plan_template(tmp_path / "monitoring_plan.json")
    plan = load_monitoring_plan(path)
    assert plan.model_id == "example-cxr-triage"
    assert plan.subgroup_min_n > 0
    assert plan.thresholds.min_auroc > 0


def test_model_inventory_csv_round_trip(tmp_path: Path) -> None:
    inventory_path = tmp_path / "inventory.csv"
    item = ModelInventoryItem(
        model_id="cxr-triage",
        model_name="CXR Triage",
        version="v1",
        owner="Radiology",
        intended_use="Prioritize worklist review.",
    )
    save_model_inventory([item], inventory_path)
    loaded = load_model_inventory(inventory_path)
    assert loaded == [item]

    template = write_model_inventory_template(tmp_path / "inventory_template.csv")
    assert pd.read_csv(template)["model_id"].tolist() == ["example-cxr-triage"]


def test_alert_reviews_populate_alert_frame(drift_df: pd.DataFrame, tmp_path: Path) -> None:
    reviews_path = tmp_path / "alert_reviews.csv"
    pd.DataFrame(
        [
            {
                "rule": "psi_major",
                "reviewer": "Dr. Reviewer",
                "disposition": "needs_investigation",
                "follow_up": "Open governance ticket",
                "reviewed_at": "2026-06-30",
                "notes": "Large score-distribution movement.",
            }
        ]
    ).to_csv(reviews_path, index=False)

    reviews = load_alert_reviews(reviews_path)
    analysis = run_monitoring_analysis(
        drift_df,
        thresholds=AlertThresholds(psi_major=0.01, psi_minor=0.001),
        alert_reviews=reviews,
        n_resamples=30,
    )
    frame = alerts_frame(analysis)
    psi_rows = frame[frame["rule"] == "psi_major"]
    assert not psi_rows.empty
    assert psi_rows.iloc[0]["reviewer"] == "Dr. Reviewer"
    assert psi_rows.iloc[0]["disposition"] == "needs_investigation"


def test_audit_log_written_for_analysis_outputs(drift_df: pd.DataFrame, tmp_path: Path) -> None:
    output = tmp_path / "analysis"
    audit_log = tmp_path / "audit.jsonl"
    analysis = run_monitoring_analysis(drift_df, n_resamples=30)
    write_analysis_outputs(analysis, output, audit_log=audit_log, audit_actor="pytest")
    rows = read_audit_log(audit_log)
    assert len(rows) == 1
    assert rows[0]["event_type"] == "analysis_outputs_written"
    assert rows[0]["actor"] == "pytest"
    assert len(rows[0]["artifact_sha256"]) == 64


def test_connector_templates_and_export_adapter(tmp_path: Path) -> None:
    files = write_connector_templates(tmp_path / "templates")
    assert files["pacs_ris"].exists()
    assert "PHI" in files["readme"].read_text(encoding="utf-8")

    export_csv = tmp_path / "export.csv"
    output_csv = tmp_path / "monitoring.csv"
    pd.DataFrame(
        [
            {
                "accession_number_hash": "a1",
                "study_date": "2026-01-01",
                "facility": "Site A",
                "Manufacturer": "GE",
                "Modality": "DX",
                "algorithm_version": "v1",
                "ground_truth": 1,
                "model_score": 0.91,
            },
            {
                "accession_number_hash": "a2",
                "study_date": "2026-01-02",
                "facility": "Site A",
                "Manufacturer": "GE",
                "Modality": "DX",
                "algorithm_version": "v1",
                "ground_truth": 0,
                "model_score": 0.11,
            },
        ]
    ).to_csv(export_csv, index=False)
    out = adapt_prediction_export(export_csv, output_csv, schema_profile="production")
    assert output_csv.exists()
    assert out["y_pred_binary"].tolist() == [1, 0]
    validate_dataframe(out, profile="production")


def test_inventory_json_loads(tmp_path: Path) -> None:
    path = tmp_path / "inventory.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_id": "m1",
                        "model_name": "Model One",
                        "version": "2026.1",
                        "owner": "Radiology",
                        "intended_use": "Monitoring fixture.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_model_inventory(path)[0].model_id == "m1"


def test_study_protocol_template_loads(tmp_path: Path) -> None:
    path = write_study_protocol_template(tmp_path / "study_protocol.json")
    protocol = load_study_protocol(path)
    assert protocol.study_id == "rad-ai-sentinel-rsna-case-study-v1"
    assert protocol.minimum_cases >= 1000
    assert "rolling_auroc" in protocol.drift_methods


def test_study_protocol_rejects_missing_required_fields() -> None:
    try:
        StudyProtocol(
            study_id="",
            title="Missing ID",
            data_source="RSNA",
            prediction_source="frozen predictions",
            primary_endpoint="AUROC",
            minimum_cases=100,
        )
    except ValueError as exc:
        assert "study_id" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("StudyProtocol accepted a missing study_id")


def test_study_protocol_rejects_invalid_case_count() -> None:
    try:
        StudyProtocol(
            study_id="s1",
            title="Invalid case count",
            data_source="RSNA",
            prediction_source="frozen predictions",
            primary_endpoint="AUROC",
            minimum_cases=0,
        )
    except ValueError as exc:
        assert "minimum_cases" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("StudyProtocol accepted zero minimum_cases")
