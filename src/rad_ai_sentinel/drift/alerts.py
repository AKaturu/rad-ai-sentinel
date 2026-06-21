"""ACR-style stop-rule alerts.

The ACR-SIIM Practice Parameter for Imaging AI (2026) requires sites to
"define stop rules" for AI tools whose real-world performance degrades. This
module operationalizes that: given current metrics, a baseline, and configurable
thresholds, it fires alerts when any rule is breached.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import DEFAULT_THRESHOLDS, AlertThresholds


@dataclass(frozen=True)
class Alert:
    """A single fired alert."""

    rule: str  # e.g. "min_auroc", "max_auroc_drop", "psi_major"
    severity: str  # "critical" or "warning"
    message: str
    observed: float
    threshold: float


@dataclass(frozen=True)
class AlertReport:
    """Collection of fired alerts plus a summary."""

    alerts: tuple[Alert, ...] = ()
    is_clean: bool = field(default=True)

    def __post_init__(self) -> None:
        # is_clean is False if any alerts exist.
        if self.alerts:
            object.__setattr__(self, "is_clean", False)

    @property
    def n_critical(self) -> int:
        return sum(1 for a in self.alerts if a.severity == "critical")

    @property
    def n_warning(self) -> int:
        return sum(1 for a in self.alerts if a.severity == "warning")


def check_alerts(
    current_auroc: float | None = None,
    baseline_auroc: float | None = None,
    current_sensitivity: float | None = None,
    current_specificity: float | None = None,
    psi_value: float | None = None,
    subgroup_sens_gap: float | None = None,
    thresholds: AlertThresholds | None = None,
) -> AlertReport:
    """Evaluate all stop rules and return fired alerts.

    Each rule is evaluated independently. If any metric is None (not computed /
    not applicable), that rule is skipped.

    Parameters
    ----------
    current_auroc, baseline_auroc:
        Current and baseline AUROC for the drop-check.
    current_sensitivity, current_specificity:
        Current 2x2 metrics.
    psi_value:
        Population Stability Index from the drift module.
    subgroup_sens_gap:
        Maximum absolute sensitivity gap across subgroups.
    thresholds:
        Configurable thresholds; defaults to ``DEFAULT_THRESHOLDS``.

    Returns
    -------
    AlertReport
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    alerts: list[Alert] = []

    # --- AUROC floor ---
    if current_auroc is not None and current_auroc < thresholds.min_auroc:
        alerts.append(
            Alert(
                rule="min_auroc",
                severity="critical",
                message=f"AUROC ({current_auroc:.4f}) below minimum ({thresholds.min_auroc:.2f})",
                observed=current_auroc,
                threshold=thresholds.min_auroc,
            )
        )

    # --- AUROC relative drop ---
    if (
        baseline_auroc is not None
        and current_auroc is not None
        and baseline_auroc > 0
        and (baseline_auroc - current_auroc) / baseline_auroc > thresholds.max_auroc_drop_relative
    ):
        rel_drop = (baseline_auroc - current_auroc) / baseline_auroc
        alerts.append(
            Alert(
                rule="max_auroc_drop_relative",
                severity="critical",
                message=(
                    f"AUROC dropped {rel_drop:.1%} from baseline "
                    f"({baseline_auroc:.4f} → {current_auroc:.4f}); "
                    f"threshold: {thresholds.max_auroc_drop_relative:.1%}"
                ),
                observed=rel_drop,
                threshold=thresholds.max_auroc_drop_relative,
            )
        )

    # --- Sensitivity floor ---
    if current_sensitivity is not None and current_sensitivity < thresholds.min_sensitivity:
        alerts.append(
            Alert(
                rule="min_sensitivity",
                severity="critical",
                message=(
                    f"Sensitivity ({current_sensitivity:.4f}) below minimum "
                    f"({thresholds.min_sensitivity:.2f})"
                ),
                observed=current_sensitivity,
                threshold=thresholds.min_sensitivity,
            )
        )

    # --- Specificity floor ---
    if current_specificity is not None and current_specificity < thresholds.min_specificity:
        alerts.append(
            Alert(
                rule="min_specificity",
                severity="critical",
                message=(
                    f"Specificity ({current_specificity:.4f}) below minimum "
                    f"({thresholds.min_specificity:.2f})"
                ),
                observed=current_specificity,
                threshold=thresholds.min_specificity,
            )
        )

    # --- PSI drift ---
    if psi_value is not None:
        if psi_value >= thresholds.psi_major:
            alerts.append(
                Alert(
                    rule="psi_major",
                    severity="critical",
                    message=(
                        f"PSI ({psi_value:.4f}) indicates major drift "
                        f"(threshold: {thresholds.psi_major:.2f})"
                    ),
                    observed=psi_value,
                    threshold=thresholds.psi_major,
                )
            )
        elif psi_value >= thresholds.psi_minor:
            alerts.append(
                Alert(
                    rule="psi_minor",
                    severity="warning",
                    message=(
                        f"PSI ({psi_value:.4f}) indicates minor drift "
                        f"(threshold: {thresholds.psi_minor:.2f})"
                    ),
                    observed=psi_value,
                    threshold=thresholds.psi_minor,
                )
            )

    # --- Subgroup disparity ---
    if subgroup_sens_gap is not None and subgroup_sens_gap > thresholds.max_subgroup_sens_gap:
        alerts.append(
            Alert(
                rule="max_subgroup_sens_gap",
                severity="warning",
                message=(
                    f"Subgroup sensitivity gap ({subgroup_sens_gap:.4f}) exceeds "
                    f"threshold ({thresholds.max_subgroup_sens_gap:.2f})"
                ),
                observed=subgroup_sens_gap,
                threshold=thresholds.max_subgroup_sens_gap,
            )
        )

    return AlertReport(alerts=tuple(alerts))
