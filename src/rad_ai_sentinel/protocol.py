"""Prospective validation protocol helpers for publication-ready monitoring studies."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import ALL_STRATIFIER_COLUMNS, DEFAULT_SUBGROUP_MIN_N

DEFAULT_DRIFT_METHODS: tuple[str, ...] = (
    "population_stability_index",
    "kolmogorov_smirnov_score_shift",
    "cusum_alerts",
    "rolling_auroc",
)


@dataclass(frozen=True)
class StudyProtocol:
    """Locked analysis plan for one external or institutional monitoring study."""

    study_id: str
    title: str
    data_source: str
    prediction_source: str
    primary_endpoint: str
    minimum_cases: int
    locked_at: str = ""
    secondary_endpoints: tuple[str, ...] = (
        "calibration_error",
        "subgroup_sensitivity_specificity",
        "drift_detection_time",
    )
    drift_methods: tuple[str, ...] = DEFAULT_DRIFT_METHODS
    required_subgroups: tuple[str, ...] = field(default_factory=lambda: ALL_STRATIFIER_COLUMNS)
    subgroup_min_n: int = DEFAULT_SUBGROUP_MIN_N
    alert_threshold_strategy: str = "pre-specified ROC/operating-point analysis"
    reviewer_roles: tuple[str, ...] = ("radiology_ai_practitioner", "data_scientist")
    registration_url: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        required = {
            "study_id": self.study_id,
            "title": self.title,
            "data_source": self.data_source,
            "prediction_source": self.prediction_source,
            "primary_endpoint": self.primary_endpoint,
            "alert_threshold_strategy": self.alert_threshold_strategy,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"Study protocol missing required fields: {', '.join(missing)}")
        if self.minimum_cases < 1:
            raise ValueError("minimum_cases must be positive")
        if self.subgroup_min_n < 1:
            raise ValueError("subgroup_min_n must be positive")
        if not self.drift_methods:
            raise ValueError("at least one drift method is required")
        if not self.required_subgroups:
            raise ValueError("at least one required subgroup is required")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = 1
        return payload


def study_protocol_from_dict(data: dict[str, Any]) -> StudyProtocol:
    """Build a study protocol from a JSON-compatible mapping."""
    return StudyProtocol(
        study_id=str(data.get("study_id", "")),
        title=str(data.get("title", "")),
        data_source=str(data.get("data_source", "")),
        prediction_source=str(data.get("prediction_source", "")),
        primary_endpoint=str(data.get("primary_endpoint", "")),
        minimum_cases=int(data.get("minimum_cases", 0)),
        locked_at=str(data.get("locked_at", "")),
        secondary_endpoints=tuple(data.get("secondary_endpoints", ())),
        drift_methods=tuple(data.get("drift_methods", DEFAULT_DRIFT_METHODS)),
        required_subgroups=tuple(data.get("required_subgroups", ALL_STRATIFIER_COLUMNS)),
        subgroup_min_n=int(data.get("subgroup_min_n", DEFAULT_SUBGROUP_MIN_N)),
        alert_threshold_strategy=str(data.get("alert_threshold_strategy", "")),
        reviewer_roles=tuple(data.get("reviewer_roles", ())),
        registration_url=str(data.get("registration_url", "")),
        notes=str(data.get("notes", "")),
    )


def load_study_protocol(path: str | Path) -> StudyProtocol:
    """Load and validate a study protocol JSON file."""
    return study_protocol_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_study_protocol(protocol: StudyProtocol, path: str | Path) -> Path:
    """Write a study protocol JSON file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(protocol.to_dict(), indent=2) + "\n", encoding="utf-8")
    return destination


def write_study_protocol_template(path: str | Path, *, force: bool = False) -> Path:
    """Write an editable prospective validation protocol template."""
    destination = Path(path)
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing study protocol: {destination}")
    protocol = StudyProtocol(
        study_id="rad-ai-sentinel-rsna-case-study-v1",
        title="External monitoring case study for a radiology AI prediction stream",
        data_source="RSNA Pneumonia Detection Challenge labels or governed institutional export",
        prediction_source="Frozen model predictions generated before outcome analysis",
        primary_endpoint="AUROC change and alert-rule firing compared with the baseline period",
        minimum_cases=1000,
        locked_at="YYYY-MM-DD",
        registration_url="https://osf.io/<placeholder>",
        notes=(
            "Replace placeholders, freeze this file before analysis, and archive its SHA-256 "
            "with the final evidence package."
        ),
    )
    return save_study_protocol(protocol, destination)
