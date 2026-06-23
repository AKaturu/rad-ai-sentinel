"""Synthetic DICOM dataset generation for tests and the demo command.

Builds valid CT Image Storage and CT Radiation Dose SR (RDSR) documents **in
memory with pydicom** and writes them to disk. No real patient data is used.
The generator plants realistic quality issues so the audit has something to
find: missing dose fields, statistical outliers, two protocol versions, and a
mix of patient-size categories.

These datasets round-trip through :mod:`dicom_dose_audit.dicom.reader`, which is
the whole point — the reader is exercised against the same DICOM structure a
real scanner export produces.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as np

# Lazy import so the module can be imported in environments where pydicom is
# present but unused (e.g. type-checking). At runtime pydicom is required.
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    generate_uid,
)

from ..config import (
    CODE_DLP,
    CODE_MEAN_CTDI_VOL,
    CODE_SCANNED_LENGTH,
    SOP_CT_IMAGE,
    SOP_CT_RDSR,
)
from .reader import DicomIngestReport, DicomRecord

# Realistic protocol baseline CTDIvol (mGy) and DLP per cm of scan length,
# approximating published survey medians. These drive plausible synthetic dose.
_PROTOCOL_BASELINES: dict[str, dict[str, float]] = {
    "CT Head": {"ctdi": 52.0, "dlp_per_cm": 52.0},
    "CT Chest": {"ctdi": 8.0, "dlp_per_cm": 8.0},
    "CT Abdomen/Pelvis": {"ctdi": 13.0, "dlp_per_cm": 13.0},
    "CT Angio Chest": {"ctdi": 22.0, "dlp_per_cm": 22.0},
}

_SCANNERS = ["GE Revolution", "Siemens SOMATOM", "Philips Brilliance", "Canon Aquilion"]
_MANUFACTURERS = {"GE Revolution": "GE", "Siemens SOMATOM": "Siemens",
                  "Philips Brilliance": "Philips", "Canon Aquilion": "Canon"}
_SITES = ["North Hospital", "South Hospital", "Community Imaging"]
_SIZES = ["small", "medium", "large", "pediatric"]


def _make_file_meta(sop_class_uid: str) -> FileMetaDataset:
    """Build a minimal valid file meta information block."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class_uid
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()
    return meta


def build_ct_image_dataset(
    *,
    study_uid: str,
    patient_id: str,
    study_date: str,
    protocol: str,
    ctdi_vol: float | None,
    scanner_model: str,
    protocol_version: str | None = None,
    kvp: float = 120.0,
    tube_current: float = 250.0,
    size_category: str | None = None,
    site: str | None = None,
) -> FileDataset:
    """Build a minimal valid CT Image Storage DICOM dataset in memory.

    Only header-level dose/metadata fields are populated — no pixel data — so
    the reader's ``stop_before_pixels=True`` path is exercised realistically.
    Per the DICOM standard, CTDIvol is the only dose quantity carried in CT
    image headers; DLP is not present here (it lives in the RDSR), so image-only
    studies will have a null DLP and surface in missing-dose detection.
    """
    sop_uid = generate_uid()
    file_meta = _make_file_meta(SOP_CT_IMAGE)
    # FileDataset signature: (filename, dataset, preamble, file_meta, ...).
    ds = FileDataset(filename_or_obj="", dataset={}, preamble=b"\0" * 128, file_meta=file_meta)
    ds.SOPClassUID = SOP_CT_IMAGE
    ds.SOPInstanceUID = sop_uid
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.PatientID = patient_id
    ds.PatientName = "SYNTHETIC^DATA"
    ds.StudyDate = study_date.replace("-", "")
    ds.SeriesDate = study_date.replace("-", "")
    ds.ProtocolName = protocol
    ds.StudyDescription = protocol
    ds.ManufacturerModelName = scanner_model
    ds.Manufacturer = _MANUFACTURERS.get(scanner_model, "Unknown")
    ds.KVP = float(kvp)
    ds.XRayTubeCurrent = int(tube_current)
    # CTDIvol is the standard CT image-header dose tag; may be absent.
    if ctdi_vol is not None:
        ds.CTDIvol = float(ctdi_vol)
    # Cosmetic fields so the object looks complete.
    ds.PatientBirthDate = ""
    ds.AccessionNumber = ""
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def _code_item(code_value: str, code_meaning: str, scheme: str = "DCM") -> Dataset:
    """Build a DICOM CodeSequence item (used inside RDSR content items)."""
    item = Dataset()
    item.CodeValue = code_value
    item.CodingSchemeDesignator = scheme
    item.CodeMeaning = code_meaning
    return item


def _content_item(
    name_code: str,
    name_meaning: str,
    *,
    numeric_value: float | None = None,
    text_value: str | None = None,
    children: list[Dataset] | None = None,
    relationship_type: str = "CONTAINS",
) -> Dataset:
    """Build a single RDSR content item (concept name + measured value)."""
    item = Dataset()
    item.RelationshipType = relationship_type
    item.ValueType = "NUM" if numeric_value is not None else ("TEXT" if text_value else "CONTAINER")
    item.ConceptNameCodeSequence = [_code_item(name_code, name_meaning)]
    if numeric_value is not None:
        item.NumericValue = float(numeric_value)
        item.MeasurementUnitsCodeSequence = [_code_item("1", "unit", "UCUM")]
    if text_value is not None:
        item.TextValue = text_value
    if children:
        item.ContentSequence = children
    return item


def build_rdsr_dataset(
    *,
    study_uid: str,
    patient_id: str,
    study_date: str,
    protocol: str,
    events: list[dict[str, float]],
    scanner_model: str,
    protocol_version: str | None = None,
    site: str | None = None,
) -> FileDataset:
    """Build a minimal CT Radiation Dose SR with one or more irradiation events.

    Each event dict has keys ``ctdi_vol``, ``dlp``, ``scan_length`` (all mGy/cm
    units). The dose is encoded as coded content items exactly as a real RDSR
    structures them, so the reader's tree walk is genuinely exercised.
    """
    file_meta = _make_file_meta(SOP_CT_RDSR)
    ds = FileDataset(filename_or_obj="", dataset={}, preamble=b"\0" * 128, file_meta=file_meta)
    ds.SOPClassUID = SOP_CT_RDSR
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "SR"
    ds.PatientID = patient_id
    ds.PatientName = "SYNTHETIC^DATA"
    ds.StudyDate = study_date.replace("-", "")
    ds.StudyDescription = protocol
    ds.ManufacturerModelName = scanner_model
    ds.Manufacturer = _MANUFACTURERS.get(scanner_model, "Unknown")

    # SR document root.
    ds.CompletionFlag = "PARTIAL"
    ds.VerificationFlag = "UNVERIFIED"
    ds.ContentTemplateSequence = []
    ds.ValueType = "CONTAINER"
    ds.ContinuityOfContent = "SEPARATE"
    ds.ConceptNameCodeSequence = [_code_item("113979", "CT Radiation Dose SR")]

    # Build per-event CONTAINER content items with coded dose children.
    event_items: list[Dataset] = []
    for i, ev in enumerate(events):
        children = [
            _content_item(CODE_MEAN_CTDI_VOL, "Mean CTDIvol", numeric_value=ev["ctdi_vol"]),
            _content_item(CODE_DLP, "DLP", numeric_value=ev["dlp"]),
            _content_item(CODE_SCANNED_LENGTH, "Scanned Length", numeric_value=ev["scan_length"]),
        ]
        event = _content_item("113913", f"Irradiation Event {i + 1}", children=children)
        event_items.append(event)

    ds.ContentSequence = event_items
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def generate_synthetic_study_specs(
    n: int = 200,
    *,
    seed: int = 42,
    start_date: str = "2025-10-01",
    outlier_fraction: float = 0.05,
    missing_dose_fraction: float = 0.04,
) -> list[dict[str, object]]:
    """Generate ``n`` synthetic study specifications (not yet DICOM objects).

    Returns a list of dicts describing each study: protocol, scanner, dose
    values, and flags. The caller decides whether to render these as CT images,
    RDSR documents, or CSV rows. Dose is sampled around protocol baselines with
    log-normal noise; a small fraction are planted high outliers and a small
    fraction have missing dose, mirroring real audit data.
    """
    rng = np.random.default_rng(seed)
    protocols = list(_PROTOCOL_BASELINES.keys())
    start = _dt.date.fromisoformat(start_date)
    specs: list[dict[str, object]] = []

    for i in range(n):
        protocol = str(rng.choice(protocols))
        baseline = _PROTOCOL_BASELINES[protocol]
        scanner = str(rng.choice(_SCANNERS))
        site = str(rng.choice(_SITES))
        size = str(rng.choice(_SIZES, p=[0.18, 0.42, 0.32, 0.08]))
        # Pediatric and large patients shift dose somewhat.
        size_factor = {"pediatric": 0.55, "small": 0.80, "medium": 1.0, "large": 1.25}[size]

        scan_length = float(np.clip(rng.normal(28.0, 6.0), 8.0, 60.0))
        is_outlier = rng.random() < outlier_fraction
        is_missing = rng.random() < missing_dose_fraction

        ctdi = baseline["ctdi"] * size_factor * rng.lognormal(0.0, 0.18)
        if is_outlier:
            ctdi *= rng.uniform(2.2, 3.5)  # planted high outlier
        dlp = ctdi * scan_length

        if is_missing:
            ctdi = None
            dlp = None

        # Two protocol versions over time for version comparison.
        phase = i / max(n - 1, 1)
        version = "v1" if phase < 0.5 else "v2"

        date = (start + _dt.timedelta(days=i // 3)).isoformat()

        specs.append(
            {
                "study_uid": str(generate_uid()),
                "patient_id": f"PT{i:05d}",
                "study_date": date,
                "protocol": protocol,
                "protocol_version": version,
                "scanner_model": scanner,
                "site": site,
                "size_category": size,
                "ctdi_vol": round(float(ctdi), 2) if ctdi is not None else None,
                "dlp": round(float(dlp), 2) if dlp is not None else None,
                "scan_length_cm": round(scan_length, 2),
                "kvp": 120.0,
                "tube_current": float(int(rng.integers(150, 350))),
                "is_outlier": bool(is_outlier),
                "is_missing": bool(is_missing),
                "source_kind": str(rng.choice(["ct_image", "rdsr"], p=[0.6, 0.4])),
            }
        )
    return specs


def specs_to_records(specs: list[dict[str, object]]) -> list[DicomRecord]:
    """Render synthetic study specs as normalized :class:`DicomRecord` values.

    Used by the CSV/demo path and by tests that want records without DICOM
    serialization overhead.
    """
    from ..config import DEFAULT_SIZE_CATEGORY

    records: list[DicomRecord] = []
    for s in specs:
        records.append(
            DicomRecord(
                study_uid=str(s["study_uid"]),
                patient_id=str(s["patient_id"]),
                study_date=str(s["study_date"]),
                protocol=str(s["protocol"]),
                ctdi_vol=s["ctdi_vol"] if s["ctdi_vol"] is not None else None,
                dlp=s["dlp"] if s["dlp"] is not None else None,
                protocol_version=str(s["protocol_version"]) if s.get("protocol_version") else None,
                scanner_model=str(s["scanner_model"]) if s.get("scanner_model") else None,
                site=str(s["site"]) if s.get("site") else None,
                size_category=str(s.get("size_category") or DEFAULT_SIZE_CATEGORY),
                kvp=float(s["kvp"]) if s.get("kvp") is not None else None,
                tube_current=float(s["tube_current"]) if s.get("tube_current") is not None else None,
                scan_length_cm=float(s["scan_length_cm"]) if s.get("scan_length_cm") is not None else None,
                has_dose_sr=s.get("source_kind") == "rdsr",
                source="synthetic",
            )
        )
    return records


def write_synthetic_dicom_dir(
    directory: str | Path,
    *,
    n: int = 200,
    seed: int = 42,
    start_date: str = "2025-10-01",
) -> tuple[list[dict[str, object]], DicomIngestReport]:
    """Generate synthetic studies, render as DICOM files, and write to disk.

    Returns the list of study specs (for assertion) and the ingestion report
    obtained by re-reading the written files — proving the round-trip.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    specs = generate_synthetic_study_specs(n=n, seed=seed, start_date=start_date)
    for i, s in enumerate(specs):
        scanner = str(s["scanner_model"])
        if s["source_kind"] == "rdsr":
            events = [
                {
                    "ctdi_vol": float(s["ctdi_vol"]) if s["ctdi_vol"] is not None else 0.0,
                    "dlp": float(s["dlp"]) if s["dlp"] is not None else 0.0,
                    "scan_length": float(s["scan_length_cm"]) if s["scan_length_cm"] else 28.0,
                }
            ]
            ds = build_rdsr_dataset(
                study_uid=str(s["study_uid"]),
                patient_id=str(s["patient_id"]),
                study_date=str(s["study_date"]),
                protocol=str(s["protocol"]),
                events=events,
                scanner_model=scanner,
            )
        else:
            ds = build_ct_image_dataset(
                study_uid=str(s["study_uid"]),
                patient_id=str(s["patient_id"]),
                study_date=str(s["study_date"]),
                protocol=str(s["protocol"]),
                ctdi_vol=s["ctdi_vol"],
                scanner_model=scanner,
            )
        ds.save_as(str(directory / f"study_{i:05d}.dcm"), write_like_original=False)

    # Re-read to prove the round-trip and return the report.
    from .reader import ingest_dicom_dir

    report = ingest_dicom_dir(directory)
    return specs, report
