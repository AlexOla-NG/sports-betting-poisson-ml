import numpy as np
import pandas as pd
import pytest

from src.ml.ensemble_blend import (
    align_oof_inputs,
    blend_probabilities,
    evaluate_probability_predictions,
    select_best_candidate,
    sweep_ensemble_weights,
    validate_probability_matrix,
)


def test_validate_probability_matrix_normalizes_and_rejects_invalid_values():
    probabilities = validate_probability_matrix(np.array([[2.0, 1.0, 1.0]]))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)
    with pytest.raises(ValueError):
        validate_probability_matrix(np.array([[np.nan, 0.5, 0.5]]))
    with pytest.raises(ValueError):
        validate_probability_matrix(np.array([[1.0, -0.1, 0.1]]))
    with pytest.raises(ValueError):
        validate_probability_matrix(np.array([[0.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        validate_probability_matrix(np.array([[1.0, 0.0]]))


def test_blending_endpoints_and_midpoint():
    mc = np.array([[0.6, 0.2, 0.2]])
    xgb = np.array([[0.2, 0.3, 0.5]])
    np.testing.assert_allclose(blend_probabilities(mc, xgb, 0.0), mc)
    np.testing.assert_allclose(blend_probabilities(mc, xgb, 1.0), xgb)
    np.testing.assert_allclose(blend_probabilities(mc, xgb, 0.5), [[0.4, 0.25, 0.35]])
    with pytest.raises(ValueError):
        blend_probabilities(mc, xgb, 1.1)


def test_metrics_use_home_draw_away_class_order():
    metrics = evaluate_probability_predictions(
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.array([0, 1]),
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["brier_score"] == 0.0
    assert metrics["log_loss"] < 1e-10


def test_sweep_and_selection_apply_metric_ties_and_simplicity():
    mc = np.tile([[0.6, 0.2, 0.2]], (6, 1))
    xgb = np.tile([[0.2, 0.3, 0.5]], (6, 1))
    target = np.array([0, 0, 0, 1, 1, 1])
    result = sweep_ensemble_weights(mc, xgb, target, [0.0, 0.5, 1.0])
    assert result["xgb_weight"].tolist() == [0.0, 0.5, 1.0]

    candidates = pd.DataFrame(
        [
            {"candidate": "blend_full", "xgb_weight": 0.5, "brier_score": 0.4, "log_loss": 1.0},
            {"candidate": "context_only", "xgb_weight": 1.0, "brier_score": 0.4, "log_loss": 1.0},
            {"candidate": "mc_only", "xgb_weight": 0.0, "brier_score": 0.4, "log_loss": 1.0},
        ]
    )
    selected = select_best_candidate(candidates, "brier_score", "log_loss", 1e-12)
    assert selected["candidate"] == "mc_only"


def test_alignment_excludes_non_oof_rows_and_rejects_conflicting_targets():
    feature_table = pd.DataFrame(
        {"fixture_id": ["a", "b", "c"], "date": pd.date_range("2025-01-01", periods=3), "target_encoded": [0, 1, 2]}
    )
    mc = pd.DataFrame(
        {"fixture_id": ["a", "b", "c"], "mc_prob_home_win": [0.5, 0.3, 0.2], "mc_prob_draw": [0.3, 0.4, 0.3], "mc_prob_away_win": [0.2, 0.3, 0.5]}
    )
    xgb = pd.DataFrame(
        {"fixture_id": ["a", "b", "c"], "xgb_prob_home_win": [np.nan, 0.3, 0.2], "xgb_prob_draw": [np.nan, 0.4, 0.3], "xgb_prob_away_win": [np.nan, 0.3, 0.5]}
    )
    aligned = align_oof_inputs(mc, xgb, feature_table)
    assert aligned["fixture_id"].tolist() == ["b", "c"]
    assert len(aligned) == 2
