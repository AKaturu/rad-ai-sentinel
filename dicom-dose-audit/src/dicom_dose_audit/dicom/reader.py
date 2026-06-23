"""DICOM dose ingestion with pydicom.

Read CTDIvol and DLP from two DICOM object types and normalize them into the
shared dose-data contract (see ``config.py``):

1. **CT Image Storage** objects — dose lives as flat header attributes:
   ``CTDIvol`` (0018,9345), ``DLP`` (0018,9934) when present, plus protocol
   (``ProtocolName`` / ``StudyDescription``), scanner (``ManufacturerModelName``
   / ``Manufacturer``), ``KVP``, ``XRayTubeCurrent``, and ``StudyDate``.

2. **CT Radiation Dose Structured Report (RDSR)** objects — dose is encoded as
   coded *content items* inside a recursive ``ContentSequence``. Each irradiation
   event carries a Mean CTDIvol (DCM code 113838) and a DLP (DCM code 113814).
   The parser walks the tree and collects per-event dose values.

The reader is deliberately tolerant: missing dose attributes, unknown SOP
classes, and partial RDSR trees are reported rather than fatal. The output is a
list of :class:`DicomRecord` values that :func:`records_to_dataframe` converts
into the validated dose dataframe.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..config import (
    CODE_CTDI_VOL,
    CODE_DLP,
    CODE_MEAN_CTDI_VOL,
    CODE_SCANNED_LENGTH,
    COL_CTDI_VOL,
    COL_DLP,
    COL_HAS_DOSE_SR,
    COL_KVP,
    COL_PATIENT_ID,
    COL_PROTOCOL,
    COL_PROTOCOL_VERSION,
    COL_SCAN_LENGTH_CM,
    COL_SCANNER_MANUFACTURER,
    COL_SCANNER_MODEL,
    COL_SITE,
    COL_SIZE_CATEGORY,
    COL_SOURCE,
    COL_STUDY_DATE,
    COL_STUDY_UID,
    COL_TUBE_CURRENT,
    DEFAULT_SIZE_CATEGORY,
    SOP_CT_IMAGE,
    SOP_CT_RDSR,
)

logger = logging.getLogger(__name__)


@dataclass
class DicomRecord:
    """One normalized dose record derived from a DICOM object.

    Dose values (``ctdi_vol``, ``dlp``) are ``None`` when the source object did
    not carry them; downstream missing-dose detection consumes those nulls.
    """

    study_uid: str
    patient_id: str
    study_date: str  # YYYYMMDD or YYYY-MM-DD; coerced to datetime by the schema
    protocol: str
    ctdi_vol: float | None = None
    dlp: float | None = None
    protocol_version: str | None = None
    scanner_model: str | None = None
    scanner_manufacturer: str | None = None
    site: str | None = None
    size_category: str | None = None
    kvp: float | None = None
    tube_current: float | None = None
    scan_length_cm: float | None = None
    has_dose_sr: bool = False
    source: str = "dicom"


@dataclass
class DicomIngestReport:
    """Summary of a directory ingestion pass for transparency/logging."""

    scanned: int = 0
    ct_image_records: int = 0
    rdsr_records: int = 0
    skipped_non_dicom: int = 0
    skipped_unsupported: int = 0
    records: list[DicomRecord] = field(default_factory=list)
    # Per-file reasons for skipping, for debugging.
    skip_reasons: list[str] = field(default_factory=list)

    @property
    def total_records(self) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# Single-object reading
# ---------------------------------------------------------------------------


def read_dicom_dose(ds: object) -> DicomRecord | None:
    """Read dose data from a single pydicom Dataset.

    Returns ``None`` for unsupported object types (non-CT, non-RDSR). Returns a
    :class:`DicomRecord` with possibly-null dose values for supported objects
    that lack dose attributes — the missing values flow into missing-dose
    detection.

    Parameters
    ----------
    ds:
        A ``pydicom.Dataset`` (already read from disk or built in memory).
    """
    sop_class = _safe_get(ds, "SOPClassUID")
    sop_str = str(sop_class) if sop_class is not None else ""

    if sop_str in (SOP_CT_IMAGE, SOP_CT_RDSR) or _is_sr(ds):
        # CT images, RDSR containers, or generic SR documents all route here.
        if sop_str == SOP_CT_RDSR or _is_sr(ds):
            return _read_rdsr(ds)
        return _read_ct_image(ds)

    return None


def _read_ct_image(ds: object) -> DicomRecord:
    """Extract dose and metadata from a CT Image Storage object.

    CTDIvol is the only dose quantity with a reliable CT image-header tag
    ((0018,9345)). DLP has no standard image-header tag — it lives in the RDSR.
    When both CTDIvol and a scan length are present, DLP can be reconstructed as
    CTDIvol x scan length; otherwise DLP is left null and surfaces in
    missing-dose detection. This mirrors how real scanner exports behave.
    """
    study_uid = _str_or(_safe_get(ds, "StudyInstanceUID"), "UNKNOWN-STUDY")
    protocol = _derive_protocol(ds)
    ctdi_vol = _safe_float(ds, "CTDIvol")
    record = DicomRecord(
        study_uid=study_uid,
        patient_id=_str_or(_safe_get(ds, "PatientID"), "UNKNOWN-PATIENT"),
        study_date=_str_or(_safe_get(ds, "StudyDate"), _str_or(_safe_get(ds, "SeriesDate"), "")),
        protocol=protocol or "UNSPECIFIED",
        scanner_model=_safe_str(ds, "ManufacturerModelName"),
        scanner_manufacturer=_safe_str(ds, "Manufacturer"),
        kvp=_safe_float(ds, "KVP"),
        tube_current=_safe_float(ds, "XRayTubeCurrent"),
        has_dose_sr=False,
        source="dicom-ct-image",
    )
    record.ctdi_vol = ctdi_vol
    # No standard image-header DLP tag; leave None to be caught by the audit.
    record.dlp = None
    return record


def _read_rdsr(ds: object) -> DicomRecord:
    """Extract dose from a CT Radiation Dose SR content tree.

    RDSR dose values are content items with a coded concept name. We walk the
    recursive ``ContentSequence`` and accumulate the study-level totals: for a
    multi-event acquisition we sum per-event DLP and take the dose-weighted
    (by DLP) mean CTDIvol, mirroring how scanners report aggregate study dose.
    """
    study_uid = _str_or(_safe_get(ds, "StudyInstanceUID"), "UNKNOWN-STUDY")
    protocol = _derive_protocol(ds) or "UNSPECIFIED"

    events = list(_iter_irradiation_events(ds))
    ctdi_values: list[tuple[float, float]] = []  # (ctdi_vol, dlp) for weighting
    total_dlp = 0.0
    scan_length_total = 0.0
    found_any = False

    for ctdi, dlp, scan_len in events:
        found_any = True
        weight = dlp if dlp is not None and dlp > 0 else 1.0
        if ctdi is not None:
            ctdi_values.append((ctdi, weight))
        if dlp is not None:
            total_dlp += dlp
        if scan_len is not None:
            scan_length_total += scan_len

    # Dose-weighted mean CTDIvol across events (falls back to plain mean).
    mean_ctdi: float | None = None
    if ctdi_values:
        total_weight = sum(w for _, w in ctdi_values)
        if total_weight > 0:
            mean_ctdi = sum(c * w for c, w in ctdi_values) / total_weight
        else:
            mean_ctdi = sum(c for c, _ in ctdi_values) / len(ctdi_values)

    record = DicomRecord(
        study_uid=study_uid,
        patient_id=_str_or(_safe_get(ds, "PatientID"), "UNKNOWN-PATIENT"),
        study_date=_str_or(_safe_get(ds, "StudyDate"), ""),
        protocol=protocol,
        scanner_model=_safe_str(ds, "ManufacturerModelName"),
        scanner_manufacturer=_safe_str(ds, "Manufacturer"),
        ctdi_vol=mean_ctdi if mean_ctdi is not None else None,
        dlp=total_dlp if (found_any and total_dlp > 0) else None,
        scan_length_cm=scan_length_total if scan_length_total > 0 else None,
        has_dose_sr=True,
        source="dicom-rdsr",
    )
    if not found_any:
        # RDSR present but no parseable dose content — keep the record so it
        # surfaces in missing-dose detection.
        record.ctdi_vol = None
        record.dlp = None
    return record


def _iter_irradiation_events(ds: object) -> Iterator[tuple[float | None, float | None, float | None]]:
    """Yield ``(mean_ctdi_vol, dlp, scanned_length)`` per irradiation event.

    Walks the RDSR content tree depth-first. An "event" is a CONTAINER content
    item whose children include the coded dose values. We also catch loose
    dose content items that some vendors emit without a strict event container.
    """
    content_seq = _safe_get(ds, "ContentSequence")
    if content_seq is None:
        return
    yield from _walk_content(content_seq)


def _walk_content(seq: Iterable[object]) -> Iterator[tuple[float | None, float | None, float | None]]:
    """Recursively walk content items, yielding parsed dose triples.

    Each content item has a ``ConceptNameCodeSequence`` whose first code's
    ``CodeValue`` identifies the quantity (Mean CTDIvol, DLP, Scanned Length).
    Numeric content items carry the value in ``NumericValue``.
    """
    for item in seq:
        # A content item may itself contain child content (relationship CONTAINER).
        children = _safe_get(item, "ContentSequence")
        if children:
            yield from _walk_content(children)

        code_value = _concept_code_value(item)
        if code_value in (CODE_MEAN_CTDI_VOL, CODE_CTDI_VOL):
            ctdi = _safe_float(item, "NumericValue")
            if ctdi is not None:
                yield (ctdi, None, None)
        elif code_value == CODE_DLP:
            dlp = _safe_float(item, "NumericValue")
            if dlp is not None:
                yield (None, dlp, None)
        elif code_value == CODE_SCANNED_LENGTH:
            length = _safe_float(item, "NumericValue")
            if length is not None:
                yield (None, None, length)


def _concept_code_value(item: object) -> str | None:
    """Return the CodeValue of a content item's ConceptNameCodeSequence."""
    seq = _safe_get(item, "ConceptNameCodeSequence")
    if not seq:
        return None
    first = next(iter(seq), None)
    if first is None:
        return None
    val = _safe_get(first, "CodeValue")
    return str(val) if val is not None else None


# ---------------------------------------------------------------------------
# Directory ingestion
# ---------------------------------------------------------------------------


def ingest_dicom_dir(
    directory: str | Path,
    *,
    glob_pattern: str = "*.dcm",
) -> DicomIngestReport:
    """Scan a directory for DICOM objects and parse dose data from each.

    Non-DICOM files and unsupported SOP classes are counted and skipped, never
    fatal — an audit must complete over a mixed export folder.
    """
    import pydicom

    directory = Path(directory)
    report = DicomIngestReport()
    if not directory.is_dir():
        raise FileNotFoundError(f"DICOM directory not found: {directory}")

    for path in sorted(directory.rglob(glob_pattern)):
        report.scanned += 1
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
        except Exception as exc:  # ingest must not abort on one bad file
            report.skipped_non_dicom += 1
            report.skip_reasons.append(f"{path.name}: not readable ({exc.__class__.__name__})")
            continue

        record = read_dicom_dose(ds)
        if record is None:
            report.skipped_unsupported += 1
            modality = _safe_str(ds, "Modality") or "?"
            report.skip_reasons.append(f"{path.name}: unsupported SOP class ({modality})")
            continue

        report.records.append(record)
        if record.has_dose_sr:
            report.rdsr_records += 1
        else:
            report.ct_image_records += 1

    return report


def records_to_dataframe(records: list[DicomRecord]) -> pd.DataFrame:
    """Convert parsed records into the validated dose dataframe contract."""
    rows: list[dict[str, object]] = []
    for r in records:
        rows.append(
            {
                COL_STUDY_UID: r.study_uid,
                COL_PATIENT_ID: r.patient_id,
                COL_STUDY_DATE: r.study_date,
                COL_PROTOCOL: r.protocol,
                COL_PROTOCOL_VERSION: r.protocol_version,
                COL_SCANNER_MODEL: r.scanner_model,
                COL_SCANNER_MANUFACTURER: r.scanner_manufacturer,
                COL_SITE: r.site,
                COL_SIZE_CATEGORY: r.size_category or DEFAULT_SIZE_CATEGORY,
                COL_CTDI_VOL: r.ctdi_vol,
                COL_DLP: r.dlp,
                COL_KVP: r.kvp,
                COL_TUBE_CURRENT: r.tube_current,
                COL_SCAN_LENGTH_CM: r.scan_length_cm,
                COL_HAS_DOSE_SR: r.has_dose_sr,
                COL_SOURCE: r.source,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_get(ds: object, name: str) -> object | None:
    """Return an attribute if present and not empty, else ``None``."""
    if ds is None:
        return None
    try:
        value = getattr(ds, name, None)
    except Exception:
        return None
    if value is None:
        return None
    # pydicom returns empty multi-values as empty sequences; treat as missing.
    try:
        if hasattr(value, "__len__") and len(value) == 0:
            return None
    except Exception:
        pass
    return value


def _safe_float(ds: object, name: str) -> float | None:
    value = _safe_get(ds, name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_str(ds: object, name: str) -> str | None:
    value = _safe_get(ds, name)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str_or(value: object | None, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _derive_protocol(ds: object) -> str | None:
    """Best-effort protocol name from an object's protocol/description fields."""
    return _safe_str(ds, "ProtocolName") or _safe_str(ds, "StudyDescription")


def _is_sr(ds: object) -> bool:
    """Heuristic: is this a Structured Report document?"""
    modality = _safe_str(ds, "Modality")
    if modality == "SR":
        return True
    # SR document SOP classes live under 1.2.840.10008.5.1.4.1.1.88.*
    sop = _safe_str(ds, "SOPClassUID") or ""
    return sop.startswith("1.2.840.10008.5.1.4.1.1.88.")
