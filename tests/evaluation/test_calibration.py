import numpy as np
import pandas as pd
import pytest

from src.evaluation.calibration import (
    PROBABILITY_COLUMNS,
    build_calibration_report,
    classification_accuracy,
    compute_chronological_brier_trend,
    compute_per_class_calibration_summary,
    compute_reliability_table,
    expected_calibration_error,
    multiclass_brier_score,
    multiclass_log_loss,
    save_calibration_artifacts,
    validate_multiclass_probabilities,
    validate_multiclass_target,
)


def probabilities():
    return np.array([[0.8, 0.1, 0.1], [0.1, 0.7, 0.2], [0.2, 0.2, 0.6], [0.4, 0.3, 0.3]])


def test_probability_validation_rejects_invalid_inputs():
    assert validate_multiclass_probabilities(probabilities()).dtype == np.float64
    for invalid in [np.empty((0, 3)), np.ones((2, 2)), np.array([[np.nan, 0.5, 0.5]]), np.array([[np.inf, 0.0, 1.0]]), np.array([[-0.1, 0.5, 0.6]]), np.array([[0.0, 0.0, 0.0]]), np.array([[0.5, 0.5, 0.2]])]:
        with pytest.raises(ValueError):
            validate_multiclass_probabilities(invalid)


def test_target_validation_and_metric_ordering():
    target = np.array([0, 1, 2, 0])
    assert validate_multiclass_target(target).dtype.kind == "i"
    with pytest.raises(ValueError):
        validate_multiclass_target(np.array([]))
    with pytest.raises(ValueError):
        validate_multiclass_target(np.array([0, 3]))
    with pytest.raises(ValueError):
        validate_multiclass_target(np.array([0.5, 1.0]))
    assert multiclass_brier_score(np.eye(3), np.array([0, 1, 2])) == pytest.approx(0.0)
    assert multiclass_log_loss(np.eye(3), np.array([0, 1, 2])) < 1e-10
    assert classification_accuracy(np.eye(3), np.array([0, 1, 2])) == 1.0


def test_reliability_bins_and_ece():
    matrix = np.array([[0.05, 0.05, 0.90], [0.15, 0.75, 0.10], [0.95, 0.03, 0.02]])
    target = np.array([2, 1, 0])
    table = compute_reliability_table(matrix, target, class_index=0, n_bins=10)
    assert len(table) == 10
    assert table.loc[9, "fixture_count"] == 1
    assert expected_calibration_error(table) >= 0
    summary = compute_per_class_calibration_summary(matrix, target)
    assert summary["class_label"].tolist() == ["home_win", "draw", "away_win"]


def test_chronological_trend_sorts_and_preserves_counts():
    frame = pd.DataFrame({
        "fixture_id": ["b", "a", "c"],
        "date": ["2025-02-01", "2025-01-01", "2025-02-15"],
        "target_encoded": [1, 0, 2],
        **{column: probabilities()[:3, index] for index, column in enumerate(PROBABILITY_COLUMNS)},
    })
    trend = compute_chronological_brier_trend(frame)
    assert trend["period"].is_monotonic_increasing
    assert trend["fixture_count"].sum() == 3


def selected_frame(n=560):
    p = np.column_stack([
        np.linspace(0.35, 0.45, n),
        np.full(n, 0.3),
        np.linspace(0.35, 0.25, n),
    ])
    return pd.DataFrame({
        "fixture_id": [f"f{i}" for i in range(n)],
        "date": pd.date_range("2025-01-01", periods=n),
        "target_encoded": np.arange(n) % 3,
        PROBABILITY_COLUMNS[0]: p[:, 0],
        PROBABILITY_COLUMNS[1]: p[:, 1],
        PROBABILITY_COLUMNS[2]: p[:, 2],
        "selected_candidate": ["blend_context"] * n,
        "selected_xgb_weight": [0.8] * n,
        "selected_mc_weight": [0.2] * n,
    })


def test_report_builder_and_artifact_writer(tmp_path):
    frame = selected_frame()
    overall, per_class, reliability = build_calibration_report(frame)
    trend = compute_chronological_brier_trend(frame)
    assert len(overall) == 1
    assert len(per_class) == 3
    assert len(reliability) == 30
    paths = save_calibration_artifacts(overall, per_class, reliability, trend, tmp_path)
    assert all(path.exists() for path in paths.values())
    assert len(pd.read_parquet(paths["reliability_diagram_data"])) == 30
