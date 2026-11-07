"""Tests for the versioned FBref-only fallback evaluation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.loader import load_config
from src.ml.xgboost_model import load_model_artifact, save_model_artifact
from src.ml.ensemble_blend import evaluate_probability_predictions
from src.prediction.fbref_fallback import (
    FBREF_FALLBACK_FEATURE_COLUMNS,
    assign_fixture_identity,
    build_fbref_fallback_features,
    run_fbref_fallback_evaluation,
)
from src.simulation.monte_carlo import simulate_fixture_trials


def _config() -> dict:
    config = load_config()
    config["prediction"]["fbref_fallback"]["training_cutoff"] = "2024-01-06"
    config["ml"]["xgboost"]["n_estimators"] = 4
    config["monte_carlo"]["num_trials"] = 200
    return config


def _fixtures() -> pd.DataFrame:
    rows = [
        ("a", "2024-01-01", 1, "Alpha", "Beta", 1, 0),
        ("b", "2024-01-01", 1, "Gamma", "Delta", 0, 0),
        ("c", "2024-01-02", 2, "Alpha", "Gamma", 2, 1),
        ("d", "2024-01-02", 2, "Beta", "Delta", 1, 2),
        ("e", "2024-01-03", 3, "Alpha", "Delta", 0, 1),
        ("f", "2024-01-03", 3, "Beta", "Gamma", 2, 2),
        ("g", "2024-01-04", 4, "Alpha", "Beta", 1, 1),
        ("h", "2024-01-04", 4, "Gamma", "Delta", 2, 0),
        ("i", "2024-01-05", 5, "Alpha", "Gamma", 1, 0),
        ("j", "2024-01-05", 5, "Beta", "Delta", 0, 2),
        ("k", "2024-01-06", 6, "Alpha", "Delta", 2, 1),
        ("l", "2024-01-06", 6, "Beta", "Gamma", 0, 1),
        ("m", "2024-01-07", 7, "Alpha", "Beta", np.nan, np.nan),
        ("n", "2024-01-07", 7, "Gamma", "Delta", np.nan, np.nan),
    ]
    return pd.DataFrame(
        rows,
        columns=["fixture_id", "date", "week", "home_team", "away_team", "home_goals", "away_goals"],
    ).assign(league="EPL", season="2425")


def test_schema_excludes_xg_absence_and_current_result_features() -> None:
    features = build_fbref_fallback_features(_fixtures(), _config())
    assert list(features.columns[-len(FBREF_FALLBACK_FEATURE_COLUMNS):]) == FBREF_FALLBACK_FEATURE_COLUMNS
    forbidden = {"xg", "absence", "appearance", "elo", "poisson", "lambda", "mc"}
    assert not any(any(token in column.lower() for token in forbidden) for column in FBREF_FALLBACK_FEATURE_COLUMNS)
    assert "target_encoded" not in FBREF_FALLBACK_FEATURE_COLUMNS


def test_same_date_and_own_score_cannot_change_features() -> None:
    config = _config()
    first = build_fbref_fallback_features(_fixtures(), config).set_index("fixture_id")
    changed = _fixtures().copy()
    changed.loc[changed["fixture_id"] == "m", ["home_goals", "away_goals"]] = [5, 0]
    second = build_fbref_fallback_features(changed, config).set_index("fixture_id")
    pd.testing.assert_frame_equal(
        first.loc[["m", "n"], FBREF_FALLBACK_FEATURE_COLUMNS],
        second.loc[["m", "n"], FBREF_FALLBACK_FEATURE_COLUMNS],
    )
    assert first.loc["m", "target_encoded"] != second.loc["m", "target_encoded"]


def test_previous_completed_date_updates_later_history() -> None:
    features = build_fbref_fallback_features(_fixtures(), _config()).set_index("fixture_id")
    assert features.loc["g", "h2h_match_count"] != features.loc["m", "h2h_match_count"]


def test_identity_is_stable_when_postponed_date_changes_and_collisions_raise() -> None:
    original = _fixtures().iloc[[12]].copy()
    postponed = original.copy()
    postponed["date"] = "2024-02-01"
    assert assign_fixture_identity(original).fixture_id.iloc[0] == assign_fixture_identity(postponed).fixture_id.iloc[0]
    collision = pd.concat([original, original], ignore_index=True)
    with pytest.raises(ValueError, match="collision"):
        assign_fixture_identity(collision)


def test_insufficient_history_uses_explicit_cold_start_values() -> None:
    features = build_fbref_fallback_features(_fixtures().iloc[[0]], _config())
    row = features.iloc[0]
    assert row["home_points_form"] == 1.0
    assert row["home_goals_for_form"] == 1.35
    assert row["h2h_home_win_rate"] == 0.5
    assert row["h2h_match_count"] == 0


def test_model_save_load_prediction_parity(tmp_path: Path) -> None:
    config = _config()
    features = build_fbref_fallback_features(_fixtures(), config)
    train = features[features.target_encoded.notna()]
    from src.prediction.fbref_fallback import _fit_model

    model = _fit_model(train, train.target_encoded, config)
    path = tmp_path / "model.json"
    save_model_artifact(model, path)
    restored = load_model_artifact(path)
    matrix = train[FBREF_FALLBACK_FEATURE_COLUMNS]
    np.testing.assert_allclose(model.predict_proba(matrix), restored.predict_proba(matrix))


def test_mc_reproducibility_and_probability_integrity() -> None:
    first = simulate_fixture_trials(1.3, 1.1, num_trials=500, random_seed=42)
    second = simulate_fixture_trials(1.3, 1.1, num_trials=500, random_seed=42)
    assert first == second
    probabilities = np.array([first["mc_prob_home_win"], first["mc_prob_draw"], first["mc_prob_away_win"]])
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert probabilities.sum() == pytest.approx(1.0)


def test_evaluation_candidates_share_population_and_saved_metrics_reproduce(tmp_path: Path) -> None:
    probe = _fixtures().iloc[12:].copy()
    probe[["home_goals", "away_goals"]] = [[2, 1], [0, 0]]
    result = run_fbref_fallback_evaluation(
        _fixtures().iloc[:12],
        probe,
        _config(),
        tmp_path,
    )
    metrics = result["metrics"]
    predictions = result["predictions"]
    assert metrics.fixture_count.nunique() == 1
    assert metrics.fixture_count.iloc[0] == len(predictions) == 2
    assert set(metrics.candidate) == {
        "fallback_xgboost_only",
        "safe_base_lambda_mc_only",
        "fixed_80_20_blend",
    }
    for candidate, columns in {
        "fallback_xgboost_only": ["xgb_prob_home_win", "xgb_prob_draw", "xgb_prob_away_win"],
        "safe_base_lambda_mc_only": ["mc_prob_home_win", "mc_prob_draw", "mc_prob_away_win"],
        "fixed_80_20_blend": [
            "blend_80_20_prob_home_win",
            "blend_80_20_prob_draw",
            "blend_80_20_prob_away_win",
        ],
    }.items():
        saved = predictions[columns].to_numpy()
        reported = metrics.loc[metrics.candidate == candidate].iloc[0]
        assert evaluate_probability_predictions(saved, predictions.actual_outcome.to_numpy())["log_loss"] == pytest.approx(
            reported.log_loss
        )
    assert (tmp_path / "metadata.json").exists()
