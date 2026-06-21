"""Tests for the metrics engine: binary, curves, calibration, CI, stratified."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from rad_ai_sentinel.metrics.binary import (
    BinaryMetrics,
    ConfusionCounts,
    compute_binary_metrics,
    confusion_counts,
)
from rad_ai_sentinel.metrics.calibration import compute_calibration
from rad_ai_sentinel.metrics.ci import CI, bootstrap_ci, proportion_ci, wilson_ci
from rad_ai_sentinel.metrics.delong import delong_test
from rad_ai_sentinel.metrics.stratified import (
    StratifiedResult,
    max_subgroup_gap,
    stratify,
    stratify_all,
)

# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------


class TestWilsonCI:
    """Wilson score interval should have correct coverage at known values."""

    def test_perfect_sensitivity(self) -> None:
        """100% sensitivity in 10/10: Wilson upper bound < 1.0 but close."""
        lo, hi = wilson_ci(10, 10, confidence=0.95)
        assert lo < 1.0
        assert hi <= 1.0

    def test_zero_sensitivity(self) -> None:
        """0% sensitivity in 0/10: Wilson lower bound is 0 (exact at boundary)."""
        lo, hi = wilson_ci(0, 10, confidence=0.95)
        assert lo >= 0.0  # Wilson interval touches 0 at p=0
        assert hi < 1.0

    def test_typical_sensitivity(self) -> None:
        """75% sensitivity in 6/8."""
        lo, hi = wilson_ci(6, 8, confidence=0.95)
        # Point estimate is 0.75; CI should contain it.
        assert lo < 0.75 < hi

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError):
            wilson_ci(5, 0)  # n <= 0
        with pytest.raises(ValueError):
            wilson_ci(-1, 10)  # successes < 0
        with pytest.raises(ValueError):
            wilson_ci(11, 10)  # successes > n

    def test_proportion_ci_is_wrapper(self) -> None:
        ci = proportion_ci(6, 8)
        assert isinstance(ci, CI)
        assert ci.estimate == pytest.approx(0.75)


class TestBootstrapCI:
    """BCa bootstrap CI should contain the point estimate for reasonable data."""

    def test_mean_ci(self) -> None:
        data = np.random.default_rng(42).standard_normal(200)
        ci = bootstrap_ci(data, np.mean, n_resamples=200)
        assert ci.lower < ci.estimate < ci.upper

    def test_auroc_ci(self) -> None:
        """Bootstrap CI on AUROC should contain the point estimate."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200)
        y_score = y_true.astype(float) * 0.6 + rng.normal(0, 0.3, size=200)
        y_score = np.clip(y_score, 0, 1)

        def auc_stat(paired):
            from sklearn.metrics import roc_auc_score

            return roc_auc_score(paired[:, 0], paired[:, 1])

        paired = np.column_stack([y_true, y_score])
        ci = bootstrap_ci(paired, auc_stat, paired=True, n_resamples=200)
        assert ci.lower < ci.estimate < ci.upper


# ---------------------------------------------------------------------------
# Binary metrics
# ---------------------------------------------------------------------------


class TestConfusionCounts:
    def test_perfect_classifier(self, tiny_df: pd.DataFrame) -> None:
        cc = confusion_counts(tiny_df["y_true"], tiny_df["y_pred_binary"])
        assert cc.tp == 2 and cc.fp == 0 and cc.tn == 2 and cc.fn == 0

    def test_mixed_classifier(self, mixed_df: pd.DataFrame) -> None:
        cc = confusion_counts(mixed_df["y_true"], mixed_df["y_pred_binary"])
        # TP=3 FN=1 FP=1 TN=3
        assert cc.tp == 3 and cc.fn == 1 and cc.fp == 1 and cc.tn == 3

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            confusion_counts([1, 0], [1, 0, 0])


class TestBinaryMetrics:
    def test_perfect_classifier_metrics(self, tiny_df: pd.DataFrame) -> None:
        m = compute_binary_metrics(tiny_df["y_true"], tiny_df["y_pred_binary"])
        assert m.sensitivity.estimate == pytest.approx(1.0)
        assert m.specificity.estimate == pytest.approx(1.0)
        assert m.ppv.estimate == pytest.approx(1.0)
        assert m.npv.estimate == pytest.approx(1.0)
        assert m.accuracy.estimate == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)
        assert m.prevalence == pytest.approx(0.5)

    def test_mixed_classifier_metrics(self, mixed_df: pd.DataFrame) -> None:
        m = compute_binary_metrics(mixed_df["y_true"], mixed_df["y_pred_binary"])
        # TP=3 FN=1 FP=1 TN=3 => sens=3/4=0.75, spec=3/4=0.75
        assert m.sensitivity.estimate == pytest.approx(0.75)
        assert m.specificity.estimate == pytest.approx(0.75)
        assert m.ppv.estimate == pytest.approx(0.75)
        assert m.npv.estimate == pytest.approx(0.75)
        assert m.counts.n == 8

    def test_wilson_ci_present(self, mixed_df: pd.DataFrame) -> None:
        m = compute_binary_metrics(mixed_df["y_true"], mixed_df["y_pred_binary"])
        # CI should be non-NaN and contain point estimate
        assert m.sensitivity.lower < 0.75 < m.sensitivity.upper

    def test_empty_input(self) -> None:
        m = compute_binary_metrics([], [])
        assert m.counts.n == 0
        assert m.sensitivity.estimate != m.sensitivity.estimate  # NaN

    def test_as_dict_keys(self, mixed_df: pd.DataFrame) -> None:
        m = compute_binary_metrics(mixed_df["y_true"], mixed_df["y_pred_binary"])
        d = m.as_dict()
        for key in [
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "accuracy",
            "f1",
            "prevalence",
            "tp",
            "fp",
            "tn",
            "fn",
        ]:
            assert key in d


# ---------------------------------------------------------------------------
# Curves (ROC / PR)
# ---------------------------------------------------------------------------


class TestCurves:
    def test_roc_perfect(self, tiny_df: pd.DataFrame) -> None:
        """Perfect classifier: AUROC = 1.0."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from rad_ai_sentinel.metrics.curves import compute_roc

            result = compute_roc(tiny_df["y_true"], tiny_df["y_pred_proba"])
        assert result.area.estimate == pytest.approx(1.0, abs=0.01)

    def test_pr_perfect(self, tiny_df: pd.DataFrame) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from rad_ai_sentinel.metrics.curves import compute_pr

            result = compute_pr(tiny_df["y_true"], tiny_df["y_pred_proba"])
        assert result.area.estimate == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_perfect_calibration(self) -> None:
        """If predictions == labels, Brier should be near 0."""
        y_true = np.array([1, 1, 0, 0], dtype=float)
        y_score = np.array([1.0, 1.0, 0.0, 0.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = compute_calibration(y_true, y_score)
        assert cal.brier.estimate == pytest.approx(0.0, abs=0.01)
        assert cal.ece == pytest.approx(0.0, abs=0.05)

    def test_random_predictions(self) -> None:
        """Random (0.5) predictions on balanced data: Brier ~0.25."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200).astype(float)
        y_score = np.full(200, 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = compute_calibration(y_true, y_score)
        assert cal.brier.estimate == pytest.approx(0.25, abs=0.05)

    def test_brier_ci_contains_point(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200).astype(float)
        # Add noise so bootstrap resamples are not all identical.
        y_score = np.clip(y_true * 0.7 + (1 - y_true) * 0.3 + rng.normal(0, 0.05, 200), 0, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = compute_calibration(y_true, y_score)
        assert cal.brier.lower < cal.brier.estimate < cal.brier.upper


# ---------------------------------------------------------------------------
# Stratified analysis
# ---------------------------------------------------------------------------


class TestStratified:
    def test_stratify_by_sex(self) -> None:
        """When split by sex, both subgroups should have metrics."""
        rng = np.random.default_rng(42)
        rows = []
        for i in range(100):
            sex = "F" if i < 50 else "M"
            y_true = 1 if rng.random() < 0.3 else 0
            y_proba = y_true + rng.normal(0, 0.15)
            y_proba = np.clip(y_proba, 0, 1)
            rows.append(
                {
                    "y_true": y_true,
                    "y_pred_proba": y_proba,
                    "y_pred_binary": int(round(y_proba)),  # noqa: RUF046
                    "sex": sex,
                }
            )
        df = pd.DataFrame(rows)
        results = stratify(df, "sex")
        assert len(results) == 2
        for r in results:
            assert r.stratifier == "sex"
            assert r.n > 0
            assert not np.isnan(r.metrics.sensitivity.estimate)

    def test_stratify_missing_column(self) -> None:
        df = pd.DataFrame({"y_true": [1, 0], "y_pred_proba": [0.9, 0.1], "y_pred_binary": [1, 0]})
        assert stratify(df, "nonexistent_column") == []

    def test_stratify_all(self) -> None:
        rng = np.random.default_rng(42)
        rows = []
        for i in range(200):
            y_true = 1 if rng.random() < 0.3 else 0
            y_proba = np.clip(y_true + rng.normal(0, 0.15), 0, 1)
            rows.append(
                {
                    "y_true": y_true,
                    "y_pred_proba": y_proba,
                    "y_pred_binary": int(round(y_proba)),  # noqa: RUF046
                    "sex": "F" if i % 2 == 0 else "M",
                    "site": "A" if i % 3 == 0 else "B",
                }
            )
        df = pd.DataFrame(rows)
        results = stratify_all(df)
        assert "sex" in results
        assert "site" in results

    def test_max_subgroup_gap(self) -> None:
        """With different sensitivities across groups, gap should be > 0."""
        results = [
            StratifiedResult(
                "s",
                "A",
                50,
                BinaryMetrics(
                    sensitivity=CI(0.9, 0.8, 1.0),
                    specificity=CI(0.8, 0.7, 0.9),
                    ppv=CI(0.8, 0.7, 0.9),
                    npv=CI(0.9, 0.8, 1.0),
                    accuracy=CI(0.85, 0.8, 0.9),
                    f1=0.85,
                    prevalence=0.5,
                    counts=ConfusionCounts(tp=45, fp=5, tn=40, fn=10),
                ),
            ),
            StratifiedResult(
                "s",
                "B",
                50,
                BinaryMetrics(
                    sensitivity=CI(0.7, 0.6, 0.8),
                    specificity=CI(0.85, 0.75, 0.95),
                    ppv=CI(0.82, 0.72, 0.92),
                    npv=CI(0.74, 0.64, 0.84),
                    accuracy=CI(0.775, 0.7, 0.85),
                    f1=0.75,
                    prevalence=0.5,
                    counts=ConfusionCounts(tp=35, fp=10, tn=50, fn=15),
                ),
            ),
        ]
        gap = max_subgroup_gap(results, "sensitivity")
        assert gap == pytest.approx(0.2, abs=0.01)

    def test_single_level_returns_zero_gap(self) -> None:
        results = [
            StratifiedResult(
                "s",
                "A",
                50,
                BinaryMetrics(
                    sensitivity=CI(0.8, 0.7, 0.9),
                    specificity=CI(0.8, 0.7, 0.9),
                    ppv=CI(0.8, 0.7, 0.9),
                    npv=CI(0.8, 0.7, 0.9),
                    accuracy=CI(0.8, 0.7, 0.9),
                    f1=0.8,
                    prevalence=0.5,
                    counts=ConfusionCounts(tp=40, fp=10, tn=40, fn=10),
                ),
            ),
        ]
        assert max_subgroup_gap(results, "sensitivity") == 0.0


# ---------------------------------------------------------------------------
# DeLong test
# ---------------------------------------------------------------------------


class TestDeLong:
    def test_identical_models(self) -> None:
        """Same scores should yield no significant difference."""
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=300).astype(int)
        scores = y.astype(float) * 0.6 + rng.normal(0, 0.3, size=300)
        scores = np.clip(scores, 0, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = delong_test(y, scores, scores)
        assert result.auc_diff == pytest.approx(0.0, abs=0.01)
        assert result.p_value > 0.05

    def test_different_models(self) -> None:
        """Model that adds noise should have lower AUROC than the true model."""
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=500).astype(int)
        scores_good = y.astype(float) * 0.7 + rng.normal(0, 0.2, size=500)
        scores_bad = y.astype(float) * 0.55 + rng.normal(0, 0.4, size=500)
        scores_good = np.clip(scores_good, 0, 1)
        scores_bad = np.clip(scores_bad, 0, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = delong_test(y, scores_good, scores_bad)
        assert result.auc1 > result.auc2
        assert result.p_value < 0.05  # significant difference
