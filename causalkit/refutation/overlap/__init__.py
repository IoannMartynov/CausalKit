from __future__ import annotations

# Re-export public API from diagnostics module
from .overlap_validation import (
    DEFAULT_THRESHOLDS,
    positivity_overlap_checks,
    overlap_report_from_result,
    run_overlap_diagnostics,
    att_overlap_tests,
    edge_mass,
    ks_distance,
    auc_for_m,
    ess_per_group,
    att_weight_sum_identity,
)
from .overlap_validation import (
    calibration_report_m,
    ece_binary,
)

__all__ = [
    "DEFAULT_THRESHOLDS",
    "positivity_overlap_checks",
    "overlap_report_from_result",
    "run_overlap_diagnostics",
    "att_overlap_tests",
    "edge_mass",
    "ks_distance",
    "auc_for_m",
    "ess_per_group",
    "att_weight_sum_identity",
    "calibration_report_m",
    "ece_binary",
]
