"""Orchestration for label-based multi-class monitoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import COL_MODEL_VERSION, COL_STUDY_DATE
from .metrics.multiclass import (
    MulticlassMetrics,
    multiclass_confusion_frame,
    multiclass_metrics_from_df,
    multiclass_per_class_frame,
    multiclass_summary_frame,
)
from .schemas import SchemaProfile, validate_multiclass_dataframe


@dataclass(frozen=True)
class MulticlassMonitoringAnalysis:
    """Complete result bundle for a multi-class evaluation window."""

    dataframe: pd.DataFrame
    metrics: MulticlassMetrics
    schema_profile: SchemaProfile = SchemaProfile.PUBLIC

    @property
    def start_date(self) -> pd.Timestamp:
        return pd.to_datetime(self.dataframe[COL_STUDY_DATE]).min()

    @property
    def end_date(self) -> pd.Timestamp:
        return pd.to_datetime(self.dataframe[COL_STUDY_DATE]).max()


def load_and_validate_multiclass_csv(
    path: str | Path,
    *,
    profile: str | SchemaProfile = SchemaProfile.PUBLIC,
) -> pd.DataFrame:
    """Read and validate a multi-class monitoring CSV."""
    return validate_multiclass_dataframe(pd.read_csv(path), profile=profile)


def run_multiclass_monitoring_analysis(
    df: pd.DataFrame,
    *,
    schema_profile: str | SchemaProfile = SchemaProfile.PUBLIC,
) -> MulticlassMonitoringAnalysis:
    """Validate data and compute label-based multi-class metrics."""
    profile = SchemaProfile(schema_profile)
    validated = (
        validate_multiclass_dataframe(df, profile=profile)
        .sort_values(COL_STUDY_DATE)
        .reset_index(drop=True)
    )
    return MulticlassMonitoringAnalysis(
        dataframe=validated,
        metrics=multiclass_metrics_from_df(validated),
        schema_profile=profile,
    )


def write_multiclass_analysis_outputs(
    analysis: MulticlassMonitoringAnalysis,
    output_dir: str | Path,
    *,
    audit_log: str | Path | None = None,
    audit_actor: str = "rad-ai-sentinel",
) -> dict[str, Path]:
    """Persist multi-class analysis tables as CSV plus a compact JSON summary."""
    from .audit import append_audit_event, build_artifact_event

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary": out / "multiclass_summary_metrics.csv",
        "per_class": out / "multiclass_per_class_metrics.csv",
        "confusion": out / "multiclass_confusion_matrix.csv",
        "json": out / "multiclass_metrics_summary.json",
    }
    multiclass_summary_frame(analysis.metrics).to_csv(outputs["summary"], index=False)
    multiclass_per_class_frame(analysis.metrics).to_csv(outputs["per_class"], index=False)
    multiclass_confusion_frame(analysis.metrics).to_csv(outputs["confusion"], index=False)
    outputs["json"].write_text(
        json.dumps(multiclass_analysis_summary_dict(analysis), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if audit_log:
        append_audit_event(
            audit_log,
            build_artifact_event(
                event_type="multiclass_analysis_outputs_written",
                actor=audit_actor,
                artifact=outputs["json"],
                details={
                    "output_dir": str(out),
                    "n": analysis.metrics.n,
                    "classes": list(analysis.metrics.labels),
                },
            ),
        )
    return outputs


def multiclass_analysis_summary_dict(analysis: MulticlassMonitoringAnalysis) -> dict[str, Any]:
    """Small JSON-serializable summary for automation and CI smoke tests."""
    metrics = analysis.metrics.as_dict()
    metrics.update(
        {
            "start_date": analysis.start_date.strftime("%Y-%m-%d"),
            "end_date": analysis.end_date.strftime("%Y-%m-%d"),
            "schema_profile": analysis.schema_profile.value,
            "model_versions": sorted(
                analysis.dataframe[COL_MODEL_VERSION].dropna().unique().tolist()
            ),
        }
    )
    return metrics
