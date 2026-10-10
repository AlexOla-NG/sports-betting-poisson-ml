"""Unit tests for src/ml/xgboost_model.py.

Verifies chronological walk-forward splitting, multi-class Brier scoring,
feature filtering, native NaN handling, and model artifact persistence.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.xgboost_model import (
    compute_multiclass_brier_score,
    compute_oof_permutation_importance,
    evaluate_feature_ablations,
    generate_walk_forward_splits,
    get_feature_columns,
    load_model_artifact,
    save_model_artifact,
    train_xgboost,
)


def _validation_config(initial_train_size: int = 45, step_size: int = 15) -> dict:
    return {
        "ml": {"xgboost": {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 20, "subsample": 1.0, "colsample_bytree": 1.0}},
        "evaluation": {"initial_train_size": initial_train_size, "step_size": step_size},
        "monte_carlo": {"random_seed": 42},
    }


def _synthetic_validation_data(n_samples: int = 90) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    target = pd.Series(np.arange(n_samples) % 3)
    predictive = target.to_numpy(dtype=float) + rng.normal(0, 0.05, n_samples)
    features = pd.DataFrame(
        {
            "predictive_feature": predictive,
            "noise_feature": rng.normal(size=n_samples),
            "poisson_rating_diff": predictive,
            "adjusted_lambda_diff": predictive + rng.normal(0, 0.1, n_samples),
            "xg_for_form_diff": rng.normal(size=n_samples),
            "xg_against_form_diff": rng.normal(size=n_samples),
            "h2h_home_win_rate": rng.random(n_samples),
            "rest_days_diff": rng.normal(size=n_samples),
            "week": np.arange(n_samples),
        }
    )
    return features, target


def test_generate_walk_forward_splits_chronological_and_non_overlapping():
    """Verify that walk-forward splits are strictly chronological, expanding, and non-overlapping."""
    n_samples = 110
    initial_train_size = 50
    step_size = 20

    splits = generate_walk_forward_splits(
        n_samples=n_samples,
        initial_train_size=initial_train_size,
        step_size=step_size,
    )

    # Expected folds:
    # Fold 1: train 0:50, test 50:70
    # Fold 2: train 0:70, test 70:90
    # Fold 3: train 0:90, test 90:110
    assert len(splits) == 3

    tested_indices = []
    for i, (train_idx, test_idx) in enumerate(splits):
        # Training window must always start at 0 (expanding window)
        assert train_idx[0] == 0
        assert len(train_idx) == initial_train_size + i * step_size
        # Test window must immediately follow training window
        assert test_idx[0] == len(train_idx)
        # All training indices must be strictly before all test indices
        assert np.max(train_idx) < np.min(test_idx)
        tested_indices.extend(test_idx.tolist())

    # Ensure test indices form a continuous, non-overlapping partition of [initial_train_size : n_samples]
    assert tested_indices == list(range(initial_train_size, n_samples))


def test_generate_walk_forward_splits_invalid_inputs():
    """Verify ValueError is raised on invalid split parameters."""
    with pytest.raises(ValueError):
        generate_walk_forward_splits(n_samples=100, initial_train_size=100)

    with pytest.raises(ValueError):
        generate_walk_forward_splits(n_samples=100, initial_train_size=-5)

    with pytest.raises(ValueError):
        generate_walk_forward_splits(n_samples=100, step_size=0)


def test_compute_multiclass_brier_score():
    """Verify multi-class Brier score matches expected mathematical values."""
    # Case 1: Perfect predictions
    y_true_perfect = np.array([0, 1, 2])
    y_prob_perfect = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert compute_multiclass_brier_score(y_true_perfect, y_prob_perfect) == pytest.approx(0.0)

    # Case 2: Uniform predictions across 3 classes
    # Brier for one sample: (1/3 - 1)^2 + (1/3 - 0)^2 + (1/3 - 0)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 2/3 ≈ 0.6667
    y_true_uniform = np.array([0, 1])
    y_prob_uniform = np.array(
        [
            [1 / 3, 1 / 3, 1 / 3],
            [1 / 3, 1 / 3, 1 / 3],
        ]
    )
    assert compute_multiclass_brier_score(y_true_uniform, y_prob_uniform) == pytest.approx(2 / 3)


def test_get_feature_columns_filters_metadata_and_targets():
    """Verify default exclude columns are excluded from modeling features."""
    df = pd.DataFrame(
        {
            "fixture_id": ["fix1"],
            "league": ["ENG"],
            "date": ["2024-08-16"],
            "home_goals": [2],
            "target": ["H"],
            "target_encoded": [0],
            "points_diff": [1.5],
            "xg_diff": [0.8],
        }
    )

    feature_cols = get_feature_columns(df)
    assert "fixture_id" not in feature_cols
    assert "target_encoded" not in feature_cols
    assert "points_diff" in feature_cols
    assert "xg_diff" in feature_cols


def test_train_xgboost_synthetic_walk_forward_and_nan_handling(tmp_path: Path):
    """Verify train_xgboost runs walk-forward validation and handles NaNs natively without row drops."""
    np.random.seed(42)
    n_samples = 60
    initial_train_size = 30
    step_size = 15

    # Generate synthetic features with intentional NaNs
    feat_1 = np.random.randn(n_samples)
    feat_2 = np.random.randn(n_samples)
    # Inject NaNs into feat_2
    feat_2[0:5] = np.nan  # Initial cold start NaNs
    feat_2[40:43] = np.nan  # Evaluation window NaNs (e.g. promoted team/missing elo)

    df_features = pd.DataFrame(
        {
            "feat_1": feat_1,
            "feat_2": feat_2,
            "bool_feat": np.random.choice([True, False], size=n_samples),
        }
    )
    # 3-class target
    target = pd.Series(np.random.choice([0, 1, 2], size=n_samples))

    synthetic_config = {
        "ml": {
            "xgboost": {
                "max_depth": 2,
                "learning_rate": 0.1,
                "n_estimators": 10,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
            }
        },
        "evaluation": {
            "initial_train_size": initial_train_size,
            "step_size": step_size,
        },
        "monte_carlo": {
            "random_seed": 42,
        },
    }

    results = train_xgboost(
        features=df_features,
        target=target,
        config=synthetic_config,
    )

    # 1. Verify fold metrics
    fold_metrics = results["fold_metrics"]
    assert len(fold_metrics) == 2  # [30:45], [45:60]
    assert list(fold_metrics["fold"]) == [1, 2]
    assert all(fold_metrics["accuracy"] >= 0.0)
    assert all(fold_metrics["log_loss"] > 0.0)

    # 2. Verify out-of-fold probabilities
    oof_probs = results["oof_probabilities"]
    assert oof_probs.shape == (n_samples, 3)
    # Rows in initial train set [0:initial_train_size] were never in test folds
    assert np.isnan(oof_probs[:initial_train_size]).all()
    # Evaluated rows [initial_train_size:n_samples] must be fully populated probabilities summing to ~1
    assert not np.isnan(oof_probs[initial_train_size:]).any()
    prob_sums = np.sum(oof_probs[initial_train_size:], axis=1)
    np.testing.assert_allclose(prob_sums, 1.0, atol=1e-5)

    # 3. Verify feature importance output
    importance = results["feature_importance"]
    assert isinstance(importance, pd.Series)
    assert set(importance.index) == {"feat_1", "feat_2", "bool_feat"}

    # 4. Verify save and load artifact
    model_path = tmp_path / "model.json"
    save_model_artifact(results["final_model"], model_path)
    assert model_path.exists()

    loaded_model = load_model_artifact(model_path)
    loaded_preds = loaded_model.predict_proba(df_features[["feat_1", "feat_2", "bool_feat"]].astype(float))
    original_preds = results["final_model"].predict_proba(df_features[["feat_1", "feat_2", "bool_feat"]].astype(float))
    np.testing.assert_allclose(loaded_preds, original_preds, atol=1e-5)


def test_oof_permutation_importance_uses_held_out_rows_and_positive_signal():
    features, target = _synthetic_validation_data()
    result = compute_oof_permutation_importance(
        features,
        target,
        _validation_config(),
        feature_names=["predictive_feature", "noise_feature"],
        n_repeats=3,
    )

    assert set(result["feature"]) == {"predictive_feature", "noise_feature"}
    assert {"mean_delta_log_loss", "std_delta_log_loss", "mean_delta_brier", "std_delta_brier", "number_of_folds", "n_repeats"}.issubset(result.columns)
    predictive = result.loc[result["feature"] == "predictive_feature"].iloc[0]
    assert predictive["mean_delta_log_loss"] > 0
    assert predictive["mean_delta_brier"] > 0
    assert result.attrs["evaluated_rows"] == 45
    assert result.attrs["test_indices"].min() == 45


def test_feature_ablations_keep_features_and_chronological_splits():
    features, target = _synthetic_validation_data()
    result = evaluate_feature_ablations(features, target, _validation_config())

    assert set(result["variant"]) == {
        "full_model",
        "without_poisson_rating_diff",
        "without_poisson_mc_family",
        "xg_form_context_only",
    }
    counts = result.set_index("variant")["number_of_features"]
    assert counts["without_poisson_rating_diff"] < counts["full_model"]
    assert counts["without_poisson_mc_family"] < counts["full_model"]
    assert counts["xg_form_context_only"] < counts["full_model"]
    assert all(column not in result.attrs["feature_sets"]["full_model"] for column in ["target", "fixture_id"])
    for train_idx, test_idx in result.attrs["splits"]:
        assert train_idx.max() < test_idx.min()


def test_oof_validation_probabilities_sum_to_one():
    features, target = _synthetic_validation_data()
    result = evaluate_feature_ablations(features, target, _validation_config())
    assert np.isfinite(result[["oof_log_loss", "oof_brier_score"]].to_numpy()).all()
    np.testing.assert_allclose(result.attrs["probability_sums"], 1.0, atol=1e-12)
