"""Central configuration: column names, units, code dictionaries, audit defaults.

A single source of truth for the dose-data contract and the statistical audit
parameters so the schema, DICOM reader, CLI, dashboard, and report never drift
out of sync. Radiation dose is reported in milli-gray (mGy) for CTDIvol and
milli-gray-centimeters (mGy*cm) for DLP, the standard CT dose units.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Dose-data column contract ---------------------------------------------
# These names are the API between a user's data (CSV or parsed DICOM) and the
# analytics engine. See README "Dose CSV format".

# Required identifiers / keys.
COL_STUDY_UID = "study_uid"
COL_PATIENT_ID = "patient_id"
COL_STUDY_DATE = "study_date"

# Stratifiers used for grouping. All may be missing/empty.
COL_PROTOCOL = "protocol"
COL_PROTOCOL_VERSION = "protocol_version"
COL_SCANNER_MODEL = "scanner_model"
COL_SCANNER_MANUFACTURER = "scanner_manufacturer"
COL_SITE = "site"
COL_SIZE_CATEGORY = "size_category"

# Dose quantities (always mGy and mGy*cm respectively, after normalization).
COL_CTDI_VOL = "ctdi_vol_mgy"
COL_DLP = "dlp_mgy_cm"

# Acquisition parameters (optional, useful context).
COL_KVP = "kvp"
COL_TUBE_CURRENT = "tube_current_ma"
COL_SCAN_LENGTH_CM = "scan_length_cm"

# Provenance.
COL_HAS_DOSE_SR = "has_dose_sr"
COL_SOURCE = "source"

REQUIRED_COLUMNS: tuple[str, ...] = (
    COL_STUDY_UID,
    COL_PATIENT_ID,
    COL_STUDY_DATE,
    COL_PROTOCOL,
    COL_CTDI_VOL,
    COL_DLP,
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    COL_PROTOCOL_VERSION,
    COL_SCANNER_MODEL,
    COL_SCANNER_MANUFACTURER,
    COL_SITE,
    COL_SIZE_CATEGORY,
    COL_KVP,
    COL_TUBE_CURRENT,
    COL_SCAN_LENGTH_CM,
    COL_HAS_DOSE_SR,
    COL_SOURCE,
)

STRATIFIER_COLUMNS: tuple[str, ...] = (
    COL_PROTOCOL,
    COL_SCANNER_MODEL,
    COL_SCANNER_MANUFACTURER,
    COL_SITE,
    COL_SIZE_CATEGORY,
)

# Patient-size categories used for stratification. These are labels, not
# measurements; a site maps its own water-equivalent-diameter / age bands onto
# them. "pediatric" is kept separate from the adult small/medium/large bands.
SIZE_CATEGORIES: tuple[str, ...] = ("pediatric", "small", "medium", "large", "unknown")
DEFAULT_SIZE_CATEGORY = "unknown"

# Canonical units (for display and validation messages).
UNIT_CTDI_VOL = "mGy"
UNIT_DLP = "mGy*cm"

# --- DICOM constants --------------------------------------------------------

# SOP Class UIDs we recognize when reading DICOM objects.
SOP_CT_IMAGE = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
SOP_CT_RDSR = "1.2.840.10008.5.1.4.1.1.88.70"  # Enhanced SR / CT Radiation Dose SR container

# DICOM tag numbers (group, element) for CT image header dose fields.
TAG_CTDI_VOL = (0x0018, 0x9345)  # CTDIvol
TAG_DLP = (0x0018, 0x9934)  # DLP appears in some CT image/enhanced objects
TAG_KVP = (0x0018, 0x0060)
TAG_TUBE_CURRENT = (0x0018, 0x1150)
TAG_PROTOCOL_NAME = (0x0018, 0x1030)
TAG_STUDY_DESCRIPTION = (0x0008, 0x1030)
TAG_MANUFACTURER = (0x0008, 0x0070)
TAG_MANUFACTURER_MODEL = (0x0008, 0x1090)
TAG_STUDY_DATE = (0x0008, 0x0020)
TAG_STUDY_UID = (0x0020, 0x000D)
TAG_PATIENT_ID = (0x0010, 0x0020)

# DCM (DICOM Modality) code values used inside RDSR content items.
CODE_CTDI_VOL = "113813"  # "CT Dose Information" container often holds CTDIvol
CODE_MEAN_CTDI_VOL = "113838"  # "Mean CTDIvol"
CODE_DLP = "113814"  # "CT Dose Length Product" / "DLP"
CODE_SCANNED_LENGTH = "113829"  # "Scanned Length"
CODE_CT_ACQUISITION_TYPE = "113816"
CODE_PROTOCOL = "113919"  # "CT Acquisition Type" / protocol context


# --- Statistical audit defaults --------------------------------------------

DEFAULT_CONFIDENCE_LEVEL: float = 0.95
DEFAULT_BOOTSTRAP_N: int = 1000  # modest for snappy CLI/dashboard use
DEFAULT_RANDOM_SEED: int = 42

# Outlier detection defaults (robust methods).
# IQR multiplier of 1.5 is the standard Tukey fence; 3.0 is "far" outlier.
DEFAULT_IQR_MULTIPLIER: float = 1.5
DEFAULT_IQR_MULTIPLIER_FAR: float = 3.0
# Minimum number of studies within a protocol group before outliers are flagged.
# Small groups are statistically unreliable for fence methods.
DEFAULT_MIN_GROUP_SIZE: int = 8
# Modified z-score (MAD-based) threshold; >=3.5 is a conventional "outlier".
DEFAULT_MAD_Z_THRESHOLD: float = 3.5


@dataclass(frozen=True)
class OutlierConfig:
    """Configurable parameters for statistical outlier detection.

    The audit flags *statistical* outliers using robust methods. A statistical
    outlier is a dose value unusually far from the protocol's typical spread —
    it is NOT a determination that a study was clinically unsafe. Clinical
    interpretation requires institutional review against validated benchmarks.
    """

    # Tukey IQR fence multiplier; values outside Q1 - k*IQR or Q3 + k*IQR flagged.
    iqr_multiplier: float = DEFAULT_IQR_MULTIPLIER
    # Stricter multiplier for "far" outliers.
    iqr_multiplier_far: float = DEFAULT_IQR_MULTIPLIER_FAR
    # Minimum studies per protocol group before flagging any outlier.
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE
    # Modified z-score (median absolute deviation) threshold for cross-check.
    mad_z_threshold: float = DEFAULT_MAD_Z_THRESHOLD


DEFAULT_OUTLIER_CONFIG: OutlierConfig = OutlierConfig()


@dataclass(frozen=True)
class DRLBenchmark:
    """A user-supplied Diagnostic Reference Level benchmark for one protocol.

    DRLs are typically the 75th percentile of observed dose at a national or
    local level. They are advisory optimization targets, NOT safety limits.
    Benchmarks are entirely optional; when absent, the audit relies purely on
    the within-cohort statistical-outlier analysis. A site must set these from
    its own validated institutional/national survey data.
    """

    protocol: str
    ctdi_vol_drl_mgy: float | None = None
    dlp_drl_mgy_cm: float | None = None
    # Human-readable source citation, e.g. "ACR 2023 national survey".
    source: str = "user-supplied"


# --- Grouping / display defaults -------------------------------------------

# Dose metric columns the audit summarizes and plots.
DOSE_METRIC_COLUMNS: tuple[str, ...] = (COL_CTDI_VOL, COL_DLP)

# Summary statistics computed per group.
SUMMARY_STATISTICS: tuple[str, ...] = (
    "n",
    "missing",
    "mean",
    "median",
    "p25",
    "p75",
    "p90",
    "std",
    "min",
    "max",
)

# Safety disclaimer carried through all user-facing surfaces.
SAFETY_DISCLAIMER = (
    "This tool identifies STATISTICAL outliers and summarizes dose distributions "
    "for quality-improvement review. It does NOT determine that any study was "
    "clinically unsafe, and it is not a medical device. Diagnostic reference "
    "levels and clinical thresholds require institutional review, validated "
    "benchmarks, and qualified medical-physics oversight before any action."
)
