"""Clinical AI governance helpers for monitoring plans and reviews."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .config import (
    ALL_STRATIFIER_COLUMNS,
    DEFAULT_SUBGROUP_MIN_N,
    DEFAULT_THRESHOLDS,
    THRESHOLD_FIELDS,
    AlertThresholds,
)


@dataclass(frozen=True)
class MonitoringPlan:
    """Site-defined monitoring controls for one deployed model."""

    model_id: str
    model_owner: str
    intended_use: str
    review_cadence: str = "monthly"
    operating_threshold: float = 0.5
    thresholds: AlertThresholds = DEFAULT_THRESHOLDS
    subgroup_min_n: int = DEFAULT_SUBGROUP_MIN_N
    required_subgroups: tuple[str, ...] = field(default_factory=lambda: ALL_STRATIFIER_COLUMNS)
    stop_rule_contacts: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("Monitoring plan requires model_id")
        if not self.model_owner.strip():
            raise ValueError("Monitoring plan requires model_owner")
        if not self.intended_use.strip():
            raise ValueError("Monitoring plan requires intended_use")
        if not 0.0 <= self.operating_threshold <= 1.0:
            raise ValueError("operating_threshold must be in [0, 1]")
        if self.subgroup_min_n < 1:
            raise ValueError("subgroup_min_n must be positive")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["schema_version"] = 1
        data["thresholds"] = asdict(self.thresholds)
        return data


@dataclass(frozen=True)
class ModelInventoryItem:
    """One algorithm/version tracked by a local AI governance inventory."""

    model_id: str
    model_name: str
    version: str
    owner: str
    intended_use: str
    status: str = "active"
    monitoring_plan_path: str = ""
    last_reviewed: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        required = {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "owner": self.owner,
            "intended_use": self.intended_use,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"Model inventory item missing required fields: {', '.join(missing)}")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AlertReview:
    """Human review metadata for one fired alert rule."""

    rule: str
    reviewer: str = ""
    disposition: str = "unreviewed"
    follow_up: str = ""
    reviewed_at: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.rule.strip():
            raise ValueError("Alert review requires rule")
        if not self.disposition.strip():
            raise ValueError("Alert review requires disposition")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


INVENTORY_COLUMNS: tuple[str, ...] = (
    "model_id",
    "model_name",
    "version",
    "owner",
    "intended_use",
    "status",
    "monitoring_plan_path",
    "last_reviewed",
    "notes",
)

ALERT_REVIEW_COLUMNS: tuple[str, ...] = (
    "rule",
    "reviewer",
    "disposition",
    "follow_up",
    "reviewed_at",
    "notes",
)


def load_monitoring_plan(path: str | Path) -> MonitoringPlan:
    """Load a monitoring plan from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return monitoring_plan_from_dict(payload)


def save_monitoring_plan(plan: MonitoringPlan, path: str | Path) -> Path:
    """Write a monitoring plan to JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8")
    return destination


def write_monitoring_plan_template(path: str | Path, *, force: bool = False) -> Path:
    """Write an editable monitoring-plan template."""
    destination = Path(path)
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing monitoring plan: {destination}")
    plan = MonitoringPlan(
        model_id="example-cxr-triage",
        model_owner="Radiology AI Governance Committee",
        intended_use="Prioritize chest radiographs for radiologist review.",
        review_cadence="monthly",
        stop_rule_contacts=("ai-governance@example.org",),
    )
    return save_monitoring_plan(plan, destination)


def monitoring_plan_from_dict(data: dict[str, Any]) -> MonitoringPlan:
    """Build a MonitoringPlan from a dict with light alias handling."""
    thresholds = _thresholds_from_dict(data.get("thresholds", {}))
    return MonitoringPlan(
        model_id=str(data.get("model_id", data.get("id", ""))),
        model_owner=str(data.get("model_owner", data.get("owner", ""))),
        intended_use=str(data.get("intended_use", "")),
        review_cadence=str(data.get("review_cadence", "monthly")),
        operating_threshold=float(data.get("operating_threshold", 0.5)),
        thresholds=thresholds,
        subgroup_min_n=int(data.get("subgroup_min_n", DEFAULT_SUBGROUP_MIN_N)),
        required_subgroups=tuple(data.get("required_subgroups", ALL_STRATIFIER_COLUMNS)),
        stop_rule_contacts=tuple(data.get("stop_rule_contacts", ())),
    )


def load_model_inventory(path: str | Path) -> list[ModelInventoryItem]:
    """Load a model inventory from CSV or JSON."""
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = cast(list[dict[str, Any]], payload["models"] if isinstance(payload, dict) else payload)
        return [model_inventory_item_from_dict(row) for row in rows]
    frame = pd.read_csv(source).fillna("")
    rows = cast(list[dict[str, Any]], frame.to_dict(orient="records"))
    return [model_inventory_item_from_dict(row) for row in rows]


def save_model_inventory(items: list[ModelInventoryItem], path: str | Path) -> Path:
    """Write a model inventory to CSV or JSON based on file suffix."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = [item.to_dict() for item in items]
    if destination.suffix.lower() == ".json":
        destination.write_text(
            json.dumps({"schema_version": 1, "models": rows}, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        pd.DataFrame(rows, columns=INVENTORY_COLUMNS).to_csv(destination, index=False)
    return destination


def write_model_inventory_template(path: str | Path, *, force: bool = False) -> Path:
    """Write an editable model-inventory template."""
    destination = Path(path)
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing model inventory: {destination}")
    item = ModelInventoryItem(
        model_id="example-cxr-triage",
        model_name="Example CXR Triage",
        version="v1.0",
        owner="Radiology AI Governance Committee",
        intended_use="Prioritize chest radiographs for radiologist review.",
        monitoring_plan_path="monitoring_plan.json",
    )
    return save_model_inventory([item], destination)


def model_inventory_item_from_dict(data: dict[str, Any]) -> ModelInventoryItem:
    return ModelInventoryItem(
        model_id=str(data.get("model_id", "")),
        model_name=str(data.get("model_name", data.get("name", ""))),
        version=str(data.get("version", "")),
        owner=str(data.get("owner", data.get("model_owner", ""))),
        intended_use=str(data.get("intended_use", "")),
        status=str(data.get("status", "active")),
        monitoring_plan_path=str(data.get("monitoring_plan_path", "")),
        last_reviewed=str(data.get("last_reviewed", "")),
        notes=str(data.get("notes", "")),
    )


def load_alert_reviews(path: str | Path) -> tuple[AlertReview, ...]:
    """Load alert-review metadata from CSV or JSON."""
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        rows = cast(list[dict[str, Any]], payload["reviews"] if isinstance(payload, dict) else payload)
    else:
        rows = cast(list[dict[str, Any]], pd.read_csv(source).fillna("").to_dict(orient="records"))
    return tuple(alert_review_from_dict(row) for row in rows)


def alert_review_from_dict(data: dict[str, Any]) -> AlertReview:
    return AlertReview(
        rule=str(data.get("rule", "")),
        reviewer=str(data.get("reviewer", "")),
        disposition=str(data.get("disposition", "unreviewed")),
        follow_up=str(data.get("follow_up", data.get("follow_up_action", ""))),
        reviewed_at=str(data.get("reviewed_at", "")),
        notes=str(data.get("notes", data.get("review_notes", ""))),
    )


def alert_reviews_by_rule(reviews: tuple[AlertReview, ...]) -> dict[str, AlertReview]:
    """Index reviews by alert rule; later rows intentionally override earlier rows."""
    return {review.rule: review for review in reviews}


def _thresholds_from_dict(values: dict[str, Any]) -> AlertThresholds:
    base = asdict(DEFAULT_THRESHOLDS)
    overrides = {field: values[field] for field in THRESHOLD_FIELDS if field in values}
    base.update(overrides)
    return AlertThresholds(**base)
