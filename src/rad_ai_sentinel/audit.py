"""Append-only audit events for report generation and alert decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    """A single audit event serialized as one JSONL row."""

    event_type: str
    actor: str
    artifact: str
    artifact_sha256: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def append_audit_event(path: str | Path, event: AuditEvent) -> Path:
    """Append one event to a JSONL audit log."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
    return destination


def read_audit_log(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read audit events without mutating the log."""
    source = Path(path)
    if not source.exists():
        return ()
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return tuple(rows)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest for an artifact file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_event(
    *,
    event_type: str,
    actor: str,
    artifact: str | Path,
    details: dict[str, Any] | None = None,
) -> AuditEvent:
    """Create a checksum-backed audit event for an artifact."""
    artifact_path = Path(artifact)
    return AuditEvent(
        event_type=event_type,
        actor=actor,
        artifact=str(artifact_path),
        artifact_sha256=file_sha256(artifact_path),
        details=details or {},
    )
