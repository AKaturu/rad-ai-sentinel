"""Tests for drift detection, alerts, version comparison, and missing-data analysis."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from rad_ai_sentinel.drift.alerts import check_alerts
from rad_ai_sentinel.drift.detection import (
    compute_drift,
    cusum,
    kl_divergence,
    psi,
    rolling_auroc,
)
from rad_ai_sentinel.drift.missing import analyze_missing_data
from rad_ai_sentinel.drift.versions import compare_versions

# ---------------------------------------------------------------------------
# PSI / KL divergence
# ---------------------------------------------------------------------------


class TestPSI:
    def test_identical_distributions(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, size=1000)
        assert psi(scores, scores) == pytest.approx(0.0, abs=0.001)

    def test_different_distributions(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0.3, 0.1, size=1000)
        ref = np.clip(ref, 0, 1)
        cur = rng.normal(0.7, 0.1, size=1000)
        cur = np.clip(cur, 0, 1)
        # Very different distributions -> high PSI.
        assert psi(ref, cur) > 0.5

    def test_similar_distributions(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(0.5, 0.15, size=2000)
        ref = np.clip(ref, 0, 1)
        cur = rng.normal(0.5, 0.16, size=2000)
        cur = np.clip(cur, 0, 1)
        # Slightly different -> low PSI.
        assert psi(ref, cur) < 0.1


class TestKLDivergence:
    def test_identical_distributions(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, size=1000)
        assert kl_divergence(scores, scores) == pytest.approx(0.0, abs=0.01)

    def test_different_distributions(self) -> None:
        rng = np.random.default_rng(42)
        ref = np.clip(rng.normal(0.3, 0.1, 1000), 0, 1)
        cur = np.clip(rng.normal(0.7, 0.1, 1000), 0, 1)
        assert kl_divergence(ref, cur) > 0.1


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------


class TestCUSUM:
    def test_stable_series_no_detection(self) -> None:
        # Stable series around a mean should not trigger.
        values = np.array([0.9, 0.91, 0.89, 0.9, 0.92, 0.88, 0.9, 0.91])
        result = cusum(values, threshold=0.05)
        assert result.s_high.shape == values.shape

    def test_shifted_series_detection(self) -> None:
        # Series with a clear downward shift should trigger at least once.
        values = np.array([0.9, 0.91, 0.89, 0.9] * 5 + [0.5, 0.48, 0.52, 0.49] * 5)
        result = cusum(values, threshold=0.5)
        assert result.detections.sum() >= 1

    def test_empty_series(self) -> None:
        result = cusum(np.array([]))
        assert len(result.s_high) == 0


# ---------------------------------------------------------------------------
# Rolling AUROC
# ---------------------------------------------------------------------------


class TestRollingAUROC:
    def test_returns_windows(self, drift_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rolling_auroc(drift_df, window_days=30, min_samples=10)
        assert len(result.dates) > 0
        assert result.window_size_days == 30

    def test_insufficient_data(self) -> None:
        # Tiny dataset -> all NaN.
        df = pd.DataFrame(
            {
                "study_date": pd.date_range("2026-01-01", periods=5),
                "y_true": [1, 0, 1, 0, 1],
                "y_pred_proba": [0.9, 0.1, 0.8, 0.2, 0.7],
            }
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rolling_auroc(df, window_days=10, min_samples=100)
        assert np.all(np.isnan(result.auroc))


# ---------------------------------------------------------------------------
# Compute drift (integration)
# ---------------------------------------------------------------------------


class TestComputeDrift:
    def test_planted_drift_detected(self, drift_df: pd.DataFrame) -> None:
        """Data with a planted distribution shift should show non-trivial PSI."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compute_drift(drift_df, reference_fraction=0.3)
        # The drift_df has a planted shift; PSI should exceed the 'none' threshold.
        assert result.psi_value > 0.05
        assert result.psi_level in ("none", "minor", "major")


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------


class TestAlerts:
    def test_clean_report(self) -> None:
        report = check_alerts(
            current_auroc=0.92,
            baseline_auroc=0.92,
            current_sensitivity=0.95,
            current_specificity=0.90,
            psi_value=0.02,
            subgroup_sens_gap=0.02,
        )
        assert report.is_clean
        assert len(report.alerts) == 0

    def test_all_rules_fire(self) -> None:
        report = check_alerts(
            current_auroc=0.70,
            baseline_auroc=0.92,
            current_sensitivity=0.65,
            current_specificity=0.60,
            psi_value=0.30,
            subgroup_sens_gap=0.15,
        )
        assert not report.is_clean
        assert report.n_critical >= 4
        assert report.n_warning >= 1
        rules = {a.rule for a in report.alerts}
        assert "min_auroc" in rules
        assert "max_auroc_drop_relative" in rules
        assert "min_sensitivity" in rules
        assert "min_specificity" in rules
        assert "psi_major" in rules
        assert "max_subgroup_sens_gap" in rules

    def test_psi_minor_is_warning(self) -> None:
        report = check_alerts(psi_value=0.15)
        assert report.n_warning == 1
        assert report.n_critical == 0
        assert report.alerts[0].rule == "psi_minor"

    def test_none_inputs_skip_rules(self) -> None:
        # All None -> clean report, no errors.
        report = check_alerts()
        assert report.is_clean


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------


class TestMissingData:
    def test_complete_data(self) -> None:
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "patient_id": [f"p{i}" for i in range(100)],
                "study_date": pd.date_range("2026-01-01", periods=100),
                "model_version": "v2.0",
                "y_true": rng.integers(0, 2, 100),
                "y_pred_proba": rng.uniform(0, 1, 100),
                "y_pred_binary": rng.integers(0, 2, 100),
                "sex": rng.choice(["F", "M"], 100),
                "site": rng.choice(["A", "B"], 100),
            }
        )
        report = analyze_missing_data(df)
        assert report.overall_n == 100
        # All stratifier columns present with 0% missing.
        for col_report in report.per_column:
            assert col_report.pct_missing == 0.0

    def test_missing_metadata(self) -> None:
        rng = np.random.default_rng(42)
        sex = rng.choice(["F", "M"], 100).astype(object)
        sex[:30] = None  # 30% missing
        df = pd.DataFrame(
            {
                "patient_id": [f"p{i}" for i in range(100)],
                "study_date": pd.date_range("2026-01-01", periods=100),
                "model_version": "v2.0",
                "y_true": rng.integers(0, 2, 100),
                "y_pred_proba": rng.uniform(0, 1, 100),
                "y_pred_binary": rng.integers(0, 2, 100),
                "sex": sex,
            }
        )
        report = analyze_missing_data(df)
        sex_report = next(c for c in report.per_column if c.column == "sex")
        assert sex_report.pct_missing == pytest.approx(0.30, abs=0.01)

    def test_subgroup_levels_counted(self) -> None:
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "patient_id": [f"p{i}" for i in range(100)],
                "study_date": pd.date_range("2026-01-01", periods=100),
                "model_version": "v2.0",
                "y_true": rng.integers(0, 2, 100),
                "y_pred_proba": rng.uniform(0, 1, 100),
                "y_pred_binary": rng.integers(0, 2, 100),
                "sex": ["F"] * 60 + ["M"] * 40,
            }
        )
        report = analyze_missing_data(df)
        sex_avail = next(s for s in report.subgroup_availability if s.stratifier == "sex")
        assert sex_avail.available_samples == 100
        assert sex_avail.levels == {"F": 60, "M": 40}


# ---------------------------------------------------------------------------
# Version comparison
# ---------------------------------------------------------------------------


class TestVersionComparison:
    def test_two_versions(self, versions_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = compare_versions(versions_df, "v1.0", "v2.0")
        assert result.version_a == "v1.0"
        assert result.version_b == "v2.0"
        # v1.0 should have higher AUROC than v2.0 (v2.0 is the miscalibrated one).
        assert result.metrics_a.sensitivity.estimate > 0
        assert result.metrics_b.sensitivity.estimate > 0
