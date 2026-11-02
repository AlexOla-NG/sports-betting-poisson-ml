"""Evaluation and calibration package for sports betting pipeline."""

from src.evaluation.calibration import (
    CLASS_LABELS,
    PROBABILITY_COLUMNS,
    classification_accuracy,
    compute_chronological_brier_trend,
    compute_per_class_calibration_summary,
    compute_reliability_table,
    expected_calibration_error,
    multiclass_brier_score,
    multiclass_log_loss,
    validate_multiclass_probabilities,
    validate_multiclass_target,
    validate_selected_ensemble_frame,
)

__all__ = [
    "CLASS_LABELS",
    "PROBABILITY_COLUMNS",
    "classification_accuracy",
    "compute_chronological_brier_trend",
    "compute_per_class_calibration_summary",
    "compute_reliability_table",
    "expected_calibration_error",
    "multiclass_brier_score",
    "multiclass_log_loss",
    "validate_multiclass_probabilities",
    "validate_multiclass_target",
    "validate_selected_ensemble_frame",
]
