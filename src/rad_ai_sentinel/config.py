"""Central configuration: column names, alert thresholds, defaults.

A single source of truth for the CSV contract and surveillance parameters so
the schema, CLI, dashboard, and report never drift out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- CSV column contract ---------------------------------------------------
# These names are the API between a user's data and the tool. See README.

# Required columns: a prediction cannot be monitored without these.
COL_PATIENT_ID = "patient_id"
COL_STUDY_DATE = "study_date"
COL_MODEL_VERSION = "model_version"
COL_Y_TRUE = "y_true"
COL_Y_PRED_PROBA = "y_pred_proba"
COL_Y_PRED_BINARY = "y_pred_binary"
COL_Y_PRED_LABEL = "y_pred_label"

# Optional metadata columns used for stratification. May be missing/empty.
COL_SITE = "site"
COL_SCANNER_MANUFACTURER = "scanner_manufacturer"
COL_MODALITY = "modality"
COL_AGE_GROUP = "age_group"
COL_SEX = "sex"
COL_RACE_ETHNICITY = "race_ethnicity"

REQUIRED_COLUMNS: tuple[str, ...] = (
    COL_PATIENT_ID,
    COL_STUDY_DATE,
    COL_MODEL_VERSION,
    COL_Y_TRUE,
    COL_Y_PRED_PROBA,
    COL_Y_PRED_BINARY,
)

MULTICLASS_REQUIRED_COLUMNS: tuple[str, ...] = (
    COL_PATIENT_ID,
    COL_STUDY_DATE,
    COL_MODEL_VERSION,
    COL_Y_TRUE,
    COL_Y_PRED_LABEL,
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    COL_SITE,
    COL_SCANNER_MANUFACTURER,
    COL_MODALITY,
    COL_AGE_GROUP,
    COL_SEX,
    COL_RACE_ETHNICITY,
)

# Demographic / equity stratifiers (handled with extra sensitivity).
DEMOGRAPHIC_COLUMNS: tuple[str, ...] = (COL_AGE_GROUP, COL_SEX, COL_RACE_ETHNICITY)
# Operational stratifiers.
OPERATIONAL_COLUMNS: tuple[str, ...] = (COL_SITE, COL_SCANNER_MANUFACTURER, COL_MODALITY)

ALL_STRATIFIER_COLUMNS: tuple[str, ...] = DEMOGRAPHIC_COLUMNS + OPERATIONAL_COLUMNS


# --- Statistical defaults --------------------------------------------------

DEFAULT_CONFIDENCE_LEVEL: float = 0.95
DEFAULT_BOOTSTRAP_N: int = 1000  # keep modest for snappy CLI/dashboard use
DEFAULT_RANDOM_SEED: int = 42
DEFAULT_SUBGROUP_MIN_N: int = 10

# Calibration reliability-diagram bins.
DEFAULT_CALIBRATION_BINS: int = 10

# Drift detection defaults.
DEFAULT_ROLLING_WINDOW_DAYS: int = 30
DEFAULT_DRIFT_PSI_MINOR: float = 0.1  # widely cited heuristic thresholds
DEFAULT_DRIFT_PSI_MAJOR: float = 0.25


@dataclass(frozen=True)
class AlertThresholds:
    """Configurable ACR-style 'stop rule' thresholds.

    When observed performance crosses these on the *current* evaluation window,
    an alert fires. Defaults are deliberately conservative; a site should tune
    them to its own risk tolerance and prevalence.
    """

    # Absolute minimum acceptable AUROC; below this the model is unsafe to trust.
    min_auroc: float = 0.80
    # Maximum allowed relative drop in AUROC vs. the baseline (reference) window.
    max_auroc_drop_relative: float = 0.05
    # Minimum acceptable sensitivity at the operating threshold.
    min_sensitivity: float = 0.80
    # Minimum acceptable specificity at the operating threshold.
    min_specificity: float = 0.80
    # PSI value at/above which drift is considered 'major' (triggers an alert).
    psi_major: float = DEFAULT_DRIFT_PSI_MAJOR
    # PSI value at/above which drift is considered 'minor' (a warning).
    psi_minor: float = DEFAULT_DRIFT_PSI_MINOR
    # Subgroup disparity: max allowed absolute gap in sensitivity between
    # any two subgroups within a stratifier.
    max_subgroup_sens_gap: float = 0.05


DEFAULT_THRESHOLDS: AlertThresholds = AlertThresholds()

# Convenience bundle for passing thresholds around without importing the class.
THRESHOLD_FIELDS: tuple[str, ...] = tuple(name for name in AlertThresholds.__dataclass_fields__)
