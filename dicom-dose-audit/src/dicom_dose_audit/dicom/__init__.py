"""DICOM ingestion: read dose data from CT image headers and RDSR documents."""

from __future__ import annotations

from .reader import (
    DicomIngestReport,
    DicomRecord,
    ingest_dicom_dir,
    read_dicom_dose,
    records_to_dataframe,
)
from .synthetic import (
    build_ct_image_dataset,
    build_rdsr_dataset,
    generate_synthetic_study_specs,
    specs_to_records,
    write_synthetic_dicom_dir,
)

__all__ = [
    "DicomIngestReport",
    "DicomRecord",
    "build_ct_image_dataset",
    "build_rdsr_dataset",
    "generate_synthetic_study_specs",
    "ingest_dicom_dir",
    "read_dicom_dose",
    "records_to_dataframe",
    "specs_to_records",
    "write_synthetic_dicom_dir",
]
