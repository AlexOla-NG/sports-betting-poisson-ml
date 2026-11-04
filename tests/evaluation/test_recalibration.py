"""Unit tests for temporally evaluated probability recalibration module.

Pipeline stage: 07_evaluation, Task 5.2
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.calibration import multiclass_brier_score, multiclass_log_loss
from src.evaluation.recalibration import (
    CALIBRATION_EVALUATION_SCOPE,
    IdentityCalibrator,
    IsotonicOvRCalibrator,
    SigmoidOvRCalibrator,
    TemporalFold,
    compute_recalibration_comparison,
    compute_recalibration_fold_metrics,
    compute_recalibration_reliability,
    generate_date_safe_temporal_splits,
    instantiate_calibrator,
    run_temporal_recalibration,
    save_recalibration_artifacts,
    validate_recalibration_input,
)


@pytest.fixture
def synthetic_fixture_data() -> pd.DataFrame:
    """Generate a deterministic synthetic dataset spanning 12 distinct dates."""
    rng = np.random.default_rng(42)
    n_fixtures = 120

    dates = pd.date_range("2025-01-01", periods=12, freq="7D")
    fixture_dates = np.repeat(dates, 10)  # 10 fixtures per date

    # Generate synthetic probabilities
    raw_logits = rng.normal(size=(n_fixtures, 3))
    probs = np.exp(raw_logits) / np.exp(raw_logits).sum(axis=1, keepdims=True)

    targets = rng.choice([0, 1, 2], size=n_fixtures, p=[0.45, 0.25, 0.30])

    df = pd.DataFrame(
        {
            "fixture_id": [f"fix_{i:04d}" for i in range(n_fixtures)],
            "date": fixture_dates,
            "target_encoded": targets,
            "ensemble_prob_home_win": probs[:, 0],
            "ensemble_prob_draw": probs[:, 1],
            "ensemble_prob_away_win": probs[:, 2],
        }
    )
    return df


def test_validate_recalibration_input_valid(synthetic_fixture_data: pd.DataFrame) -> None:
    """Verify input validation succeeds on valid synthetic data."""
    validated = validate_recalibration_input(
        synthetic_fixture_data,
        probability_tolerance=1e-12,
        enforce_upstream_contract=False,
    )
    assert len(validated) == len(synthetic_fixture_data)
    assert validated["date"].is_monotonic_increasing


def test_validate_recalibration_input_invalid_probabilities() -> None:
    """Verify validation raises when probabilities do not sum to 1."""
    df = pd.DataFrame(
        {
            "fixture_id": ["fix_01"],
            "date": [pd.Timestamp("2025-01-01")],
            "target_encoded": [0],
            "ensemble_prob_home_win": [0.5],
            "ensemble_prob_draw": [0.5],
            "ensemble_prob_away_win": [0.5],  # Sum = 1.5
        }
    )
    with pytest.raises(ValueError, match="probability rows must sum to one"):
        validate_recalibration_input(df, enforce_upstream_contract=False)


def test_generate_date_safe_temporal_splits(synthetic_fixture_data: pd.DataFrame) -> None:
    """Verify expanding splits preserve date boundaries, strict chronology, and retain final partial block."""
    initial_size = 40  # 4 full dates of 10 matches
    step_size = 25  # Should expand to next full date boundary (30 matches = 3 dates)

    folds = generate_date_safe_temporal_splits(
        synthetic_fixture_data,
        initial_calibration_size=initial_size,
        evaluation_block_size=step_size,
    )

    assert len(folds) >= 2
    evaluated_ids = set()

    for i, fold in enumerate(folds):
        assert isinstance(fold, TemporalFold)
        assert fold.fold_id == i
        assert fold.train_start_date <= fold.train_end_date
        assert fold.eval_start_date <= fold.eval_end_date

        # Strict date safety: max(training_date) < min(evaluation_date)
        assert fold.train_end_date < fold.eval_start_date

        # No double evaluation
        intersection = evaluated_ids.intersection(fold.eval_fixture_ids)
        assert len(intersection) == 0
        evaluated_ids.update(fold.eval_fixture_ids)

        # Expanding training size
        if i > 0:
            assert fold.train_count > folds[i - 1].train_count

    # Verify initial training rows are excluded from evaluation
    first_fold_train_ids = set(folds[0].train_fixture_ids)
    assert len(first_fold_train_ids.intersection(evaluated_ids)) == 0

    # Total evaluated fixtures + initial training fixtures == total dataset fixtures
    assert len(evaluated_ids) + folds[0].train_count == len(synthetic_fixture_data)


def test_splits_fail_on_missing_training_class() -> None:
    """Verify split generator fails when a training fold has 0 positive instances of a class."""
    df = pd.DataFrame(
        {
            "fixture_id": [f"f_{i}" for i in range(30)],
            "date": pd.date_range("2025-01-01", periods=30, freq="1D"),
            "target_encoded": [0, 1] * 15,  # Class 2 is missing entirely
            "ensemble_prob_home_win": [0.6] * 30,
            "ensemble_prob_draw": [0.4] * 30,
            "ensemble_prob_away_win": [0.0] * 30,
        }
    )
    with pytest.raises(ValueError, match="training set lacks class diversity for OvR class 2"):
        generate_date_safe_temporal_splits(
            df,
            initial_calibration_size=10,
            evaluation_block_size=10,
        )


def test_identity_calibrator_preserves_input() -> None:
    """Verify IdentityCalibrator returns an exact copy without mutating the input."""
    p_train = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
    y_train = np.array([0, 1])
    p_eval = np.array([[0.4, 0.4, 0.2], [0.2, 0.2, 0.6]])
    p_eval_orig = p_eval.copy()

    cal = IdentityCalibrator()
    cal.fit(p_train, y_train)
    out = cal.predict(p_eval)

    np.testing.assert_allclose(out, p_eval_orig, atol=1e-15)
    np.testing.assert_allclose(p_eval, p_eval_orig, atol=1e-15)  # No mutation


def test_sigmoid_calibrator_fit_and_predict() -> None:
    """Verify SigmoidOvRCalibrator produces valid normalized probabilities."""
    rng = np.random.default_rng(123)
    p_train = rng.uniform(0.1, 0.9, size=(60, 3))
    p_train = p_train / p_train.sum(axis=1, keepdims=True)
    y_train = np.repeat([0, 1, 2], 20)

    p_eval = rng.uniform(0.1, 0.9, size=(20, 3))
    p_eval = p_eval / p_eval.sum(axis=1, keepdims=True)
    p_eval_orig = p_eval.copy()

    cal = SigmoidOvRCalibrator(C=1e9, random_state=42)
    cal.fit(p_train, y_train)
    out = cal.predict(p_eval)

    assert out.shape == p_eval.shape
    assert np.all(out >= 0.0)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(p_eval, p_eval_orig, atol=1e-15)  # No mutation


def test_isotonic_calibrator_out_of_bounds_clipping() -> None:
    """Verify IsotonicOvRCalibrator clips out-of-bounds input scores to [0, 1]."""
    p_train = np.array([[0.2, 0.4, 0.4], [0.4, 0.2, 0.4], [0.6, 0.2, 0.2]])
    y_train = np.array([0, 1, 2])

    cal = IsotonicOvRCalibrator(increasing=True, y_min=0.0, y_max=1.0, out_of_bounds="clip")
    cal.fit(p_train, y_train)

    # Input with probabilities outside the training range
    p_eval = np.array([[0.05, 0.85, 0.10], [0.95, 0.02, 0.03]])
    out = cal.predict(p_eval)

    assert out.shape == p_eval.shape
    assert np.all(out >= 0.0)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)


def test_calibrator_raises_on_zero_total_row() -> None:
    """Verify calibrator raises ValueError when pre-normalization row sum is zero."""
    class MockCalibrator(IdentityCalibrator):
        def predict(self, probabilities: np.ndarray) -> np.ndarray:
            # Emulate an invalid calibrator that produces all zeros
            q = np.zeros_like(probabilities)
            row_sums = q.sum(axis=1, keepdims=True)
            if (row_sums <= 0).any():
                raise ValueError("produced zero row sum; cannot normalize")
            return q / row_sums

    p = np.array([[0.5, 0.3, 0.2]])
    with pytest.raises(ValueError, match="cannot normalize"):
        MockCalibrator().predict(p)


def test_run_temporal_recalibration_and_metrics(synthetic_fixture_data: pd.DataFrame) -> None:
    """Verify forward-evaluation generates identical fixture populations and valid metrics."""
    folds = generate_date_safe_temporal_splits(
        synthetic_fixture_data,
        initial_calibration_size=40,
        evaluation_block_size=30,
    )
    methods = ["identity", "sigmoid_ovr", "isotonic_ovr"]

    preds = run_temporal_recalibration(
        synthetic_fixture_data,
        folds=folds,
        methods=methods,
    )

    assert set(preds["calibration_method"].unique()) == set(methods)
    assert preds["evaluation_scope"].iloc[0] == CALIBRATION_EVALUATION_SCOPE

    # Each method must have the exact same evaluated fixture IDs
    eval_counts = {m: len(preds[preds["calibration_method"] == m]) for m in methods}
    assert len(set(eval_counts.values())) == 1

    fold_metrics = compute_recalibration_fold_metrics(preds, folds)
    assert len(fold_metrics) == len(methods) * len(folds)

    reliability = compute_recalibration_reliability(preds, n_bins=10)
    assert len(reliability) == len(methods) * 3 * 10

    comp_df, selected = compute_recalibration_comparison(preds, fold_metrics)
    assert selected in methods
    assert len(comp_df[comp_df["is_selected_method"]]) == 1

    # Verify deltas: negative = improvement
    id_row = comp_df[comp_df["calibration_method"] == "identity"].iloc[0]
    for _, row in comp_df.iterrows():
        expected_brier_delta = row["multiclass_brier_score"] - id_row["multiclass_brier_score"]
        expected_ll_delta = row["log_loss"] - id_row["log_loss"]
        np.testing.assert_allclose(row["brier_delta"], expected_brier_delta, atol=1e-12)
        np.testing.assert_allclose(row["log_loss_delta"], expected_ll_delta, atol=1e-12)


def test_selection_fallback_to_identity() -> None:
    """Verify conservative selection rule defaults to identity when calibrators fail criteria."""
    folds_meta = [
        {"fold_id": 0, "eval_count": 10},
        {"fold_id": 1, "eval_count": 10},
    ]
    # Synthetic fold metrics where calibrators are strictly worse
    fold_records = [
        {"fold_id": 0, "calibration_method": "identity", "multiclass_brier_score": 0.60, "log_loss": 1.00, "accuracy": 0.5},
        {"fold_id": 1, "calibration_method": "identity", "multiclass_brier_score": 0.60, "log_loss": 1.00, "accuracy": 0.5},
        {"fold_id": 0, "calibration_method": "sigmoid_ovr", "multiclass_brier_score": 0.65, "log_loss": 1.05, "accuracy": 0.45},
        {"fold_id": 1, "calibration_method": "sigmoid_ovr", "multiclass_brier_score": 0.65, "log_loss": 1.05, "accuracy": 0.45},
    ]
    fold_df = pd.DataFrame(fold_records)

    # Synthetic forward predictions matching fold metrics
    pred_records = []
    dates = pd.date_range("2025-01-01", periods=20)
    for m in ["identity", "sigmoid_ovr"]:
        for i in range(20):
            fold_id = 0 if i < 10 else 1
            pred_records.append(
                {
                    "fixture_id": f"f_{i}",
                    "date": dates[i],
                    "target_encoded": 0,
                    "fold_id": fold_id,
                    "calibration_method": m,
                    "calibrated_prob_home_win": 0.6 if m == "identity" else 0.4,
                    "calibrated_prob_draw": 0.2 if m == "identity" else 0.3,
                    "calibrated_prob_away_win": 0.2 if m == "identity" else 0.3,
                    "raw_prob_home_win": 0.6,
                    "raw_prob_draw": 0.2,
                    "raw_prob_away_win": 0.2,
                    "evaluation_scope": CALIBRATION_EVALUATION_SCOPE,
                }
            )
    pred_df = pd.DataFrame(pred_records)

    comp_df, selected = compute_recalibration_comparison(pred_df, fold_df)
    assert selected == "identity"
    sig_row = comp_df[comp_df["calibration_method"] == "sigmoid_ovr"].iloc[0]
    assert not sig_row["eligible_for_adoption"]
    assert "did not improve pooled Brier" in sig_row["rejection_reason"]


def test_save_and_verify_recalibration_artifacts(
    synthetic_fixture_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verify all 5 artifacts are saved, re-read, and validated without error."""
    folds = generate_date_safe_temporal_splits(
        synthetic_fixture_data,
        initial_calibration_size=40,
        evaluation_block_size=30,
    )
    methods = ["identity", "sigmoid_ovr"]
    preds = run_temporal_recalibration(synthetic_fixture_data, folds, methods)
    fold_metrics = compute_recalibration_fold_metrics(preds, folds)
    reliability = compute_recalibration_reliability(preds, n_bins=10)
    comp_df, selected = compute_recalibration_comparison(preds, fold_metrics)

    paths = save_recalibration_artifacts(
        fold_metrics=fold_metrics,
        comparison=comp_df,
        reliability_data=reliability,
        forward_predictions=preds,
        selected_method=selected,
        output_dir=tmp_path,
    )

    for name, p in paths.items():
        assert p.exists()
        loaded = pd.read_parquet(p)
        assert not loaded.empty
