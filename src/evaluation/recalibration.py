"""Temporally evaluated probability recalibration.

Pipeline Stage: 07_evaluation, Task 5.2
This module provides date-safe expanding temporal splits, multi-class
calibrators (identity, sigmoid_ovr, isotonic_ovr), forward-evaluation
orchestration, paired metric comparison against identity, and conservative
provisional adoption selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.evaluation.calibration import (
    CLASS_LABELS,
    PROBABILITY_COLUMNS,
    classification_accuracy,
    compute_reliability_table,
    expected_calibration_error,
    multiclass_brier_score,
    multiclass_log_loss,
    validate_multiclass_probabilities,
    validate_multiclass_target,
    validate_selected_ensemble_frame,
)

CALIBRATION_EVALUATION_SCOPE = "temporal_recalibration_conditional_on_selected_blend"


@dataclass(frozen=True)
class TemporalFold:
    """Metadata and indices for a single date-safe temporal evaluation fold."""

    fold_id: int
    train_indices: np.ndarray
    eval_indices: np.ndarray
    train_fixture_ids: list[str]
    eval_fixture_ids: list[str]
    train_count: int
    eval_count: int
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    eval_start_date: pd.Timestamp
    eval_end_date: pd.Timestamp


def validate_recalibration_input(
    df: pd.DataFrame,
    probability_tolerance: float = 1e-12,
    enforce_upstream_contract: bool = True,
) -> pd.DataFrame:
    """Validate input fixture DataFrame for temporal recalibration.

    Pipeline stage: 07_evaluation, Task 5.2.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing OOF ensemble predictions.
    probability_tolerance : float
        Tolerance for row sum = 1.0 check.
    enforce_upstream_contract : bool
        If True, enforces the 560-row, blend_context (0.80/0.20) upstream contract.
        If False, validates general schema and types (usable with synthetic test data).

    Returns
    -------
    pd.DataFrame
        Validated DataFrame sorted chronologically by [date, fixture_id].
    """
    if enforce_upstream_contract:
        ordered = validate_selected_ensemble_frame(df)
        matrix = ordered[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
        validate_multiclass_probabilities(matrix, tolerance=probability_tolerance)
        return ordered

    # Generic validation for synthetic/test data
    required = {"fixture_id", "date", "target_encoded", *PROBABILITY_COLUMNS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"input DataFrame missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("input DataFrame cannot be empty")
    if df["fixture_id"].isna().any() or df["fixture_id"].duplicated().any():
        raise ValueError("fixture_id must be present, non-null, and unique")

    result = df.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("input DataFrame contains invalid dates")

    validate_multiclass_target(result["target_encoded"].to_numpy())
    matrix = result[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    validate_multiclass_probabilities(matrix, tolerance=probability_tolerance)

    return result.sort_values(["date", "fixture_id"]).reset_index(drop=True)


def generate_date_safe_temporal_splits(
    df: pd.DataFrame,
    initial_calibration_size: int = 200,
    evaluation_block_size: int = 50,
) -> list[TemporalFold]:
    """Generate non-overlapping date-safe expanding temporal evaluation splits.

    Protocol:
    1. Start with at least initial_calibration_size earliest observations.
    2. Extend initial fitting population to include its entire final date.
    3. Evaluate on approximately evaluation_block_size fixtures.
    4. Extend evaluation block to include its entire final date.
    5. Expand fitting window to include all preceding observations.
    6. Repeat until the final fixture, including the final partial block.
    7. Guarantee max(training_date) < min(evaluation_date) for every fold.

    Pipeline stage: 07_evaluation, Task 5.2.
    """
    if df.empty:
        raise ValueError("cannot generate splits from empty DataFrame")
    if initial_calibration_size <= 0 or evaluation_block_size <= 0:
        raise ValueError("initial_calibration_size and evaluation_block_size must be positive")

    ordered = df.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    n_rows = len(ordered)
    if n_rows <= initial_calibration_size:
        raise ValueError(
            f"dataset size ({n_rows}) must be strictly greater than initial_calibration_size ({initial_calibration_size})"
        )

    # Initial training cutoff extended to full date
    init_last_date = ordered.iloc[initial_calibration_size - 1]["date"]
    current_train_end = ordered[ordered["date"] <= init_last_date].index[-1] + 1

    if current_train_end >= n_rows:
        raise ValueError("initial training cutoff extended to date boundary consumed entire dataset")

    folds: list[TemporalFold] = []
    fold_id = 0
    evaluated_indices_set: set[int] = set()

    while current_train_end < n_rows:
        target_eval_end = min(current_train_end + evaluation_block_size, n_rows)
        eval_last_date = ordered.iloc[target_eval_end - 1]["date"]

        eval_end_idx = ordered[ordered["date"] <= eval_last_date].index[-1] + 1
        eval_end_idx = min(eval_end_idx, n_rows)

        train_indices = np.arange(0, current_train_end)
        eval_indices = np.arange(current_train_end, eval_end_idx)

        train_subset = ordered.iloc[train_indices]
        eval_subset = ordered.iloc[eval_indices]

        # Verify date safety: max(train_date) < min(eval_date)
        max_train_date = train_subset["date"].max()
        min_eval_date = eval_subset["date"].min()
        if not (max_train_date < min_eval_date):
            raise ValueError(
                f"Fold {fold_id} date leakage: max training date ({max_train_date}) "
                f"is not strictly before min evaluation date ({min_eval_date})"
            )

        # Verify no double evaluation
        overlap = evaluated_indices_set.intersection(eval_indices)
        if overlap:
            raise ValueError(f"Fold {fold_id} re-evaluates fixtures: indices {overlap}")
        evaluated_indices_set.update(eval_indices)

        # Validate that training fold contains at least 1 positive and 1 negative for every OvR class
        y_train = train_subset["target_encoded"].to_numpy()
        for c in (0, 1, 2):
            pos_count = int(np.sum(y_train == c))
            neg_count = len(y_train) - pos_count
            if pos_count == 0 or neg_count == 0:
                raise ValueError(
                    f"Fold {fold_id} training set lacks class diversity for OvR class {c}: "
                    f"positives={pos_count}, negatives={neg_count}"
                )

        fold = TemporalFold(
            fold_id=fold_id,
            train_indices=train_indices,
            eval_indices=eval_indices,
            train_fixture_ids=train_subset["fixture_id"].tolist(),
            eval_fixture_ids=eval_subset["fixture_id"].tolist(),
            train_count=len(train_indices),
            eval_count=len(eval_indices),
            train_start_date=train_subset["date"].min(),
            train_end_date=max_train_date,
            eval_start_date=min_eval_date,
            eval_end_date=eval_subset["date"].max(),
        )
        folds.append(fold)
        fold_id += 1
        current_train_end = eval_end_idx

    return folds


class BaseCalibrator:
    """Base class for multi-class calibration models."""

    def fit(self, probabilities: np.ndarray, target: np.ndarray) -> BaseCalibrator:
        raise NotImplementedError

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IdentityCalibrator(BaseCalibrator):
    """Identity calibrator returning unchanged validated input probabilities."""

    def fit(self, probabilities: np.ndarray, target: np.ndarray) -> IdentityCalibrator:
        validate_multiclass_probabilities(probabilities)
        validate_multiclass_target(target)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        matrix = validate_multiclass_probabilities(probabilities)
        return matrix.copy()


class SigmoidOvRCalibrator(BaseCalibrator):
    """One-vs-Rest Platt scaling (sigmoid calibration) fitted directly on raw probabilities.

    q_k = sigmoid(a_k * p_k + b_k)
    calibrated_p_k = q_k / sum(q)
    """

    def __init__(
        self,
        solver: str = "lbfgs",
        C: float = 1e9,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        random_state: int | None = 42,
        tolerance: float = 1e-12,
    ) -> None:
        self.solver = solver
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.random_state = random_state
        self.tolerance = tolerance
        self.models: list[LogisticRegression] = []

    def fit(self, probabilities: np.ndarray, target: np.ndarray) -> SigmoidOvRCalibrator:
        matrix = validate_multiclass_probabilities(probabilities, tolerance=self.tolerance)
        labels = validate_multiclass_target(target)
        if len(matrix) != len(labels):
            raise ValueError("probabilities and target lengths must match")

        self.models = []
        for k in range(3):
            binary_target = (labels == k).astype(int)
            if np.sum(binary_target == 1) == 0 or np.sum(binary_target == 0) == 0:
                raise ValueError(f"training target for class {k} has zero variance")

            lr = LogisticRegression(
                solver=self.solver,
                C=self.C,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            lr.fit(matrix[:, k : k + 1], binary_target)
            self.models.append(lr)

        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if not self.models or len(self.models) != 3:
            raise RuntimeError("SigmoidOvRCalibrator must be fitted before predict")

        matrix = validate_multiclass_probabilities(probabilities, tolerance=self.tolerance)
        n_samples = len(matrix)
        q = np.zeros((n_samples, 3), dtype=np.float64)

        for k in range(3):
            # Extract probability of positive binary class
            q[:, k] = self.models[k].predict_proba(matrix[:, k : k + 1])[:, 1]

        if not np.isfinite(q).all() or (q < 0).any():
            raise ValueError("sigmoid output contains non-finite or negative values")

        row_sums = q.sum(axis=1, keepdims=True)
        if (row_sums <= 0).any():
            raise ValueError("sigmoid produced zero or negative row sum; cannot normalize")

        normalized = q / row_sums
        return validate_multiclass_probabilities(normalized, tolerance=self.tolerance)


class IsotonicOvRCalibrator(BaseCalibrator):
    """One-vs-Rest Isotonic Regression calibration fitted directly on raw probabilities.

    Uses increasing=True, output bounds [0, 1], and out_of_bounds='clip'.
    calibrated_p_k = q_k / sum(q)
    """

    def __init__(
        self,
        increasing: bool = True,
        y_min: float = 0.0,
        y_max: float = 1.0,
        out_of_bounds: str = "clip",
        tolerance: float = 1e-12,
    ) -> None:
        self.increasing = increasing
        self.y_min = y_min
        self.y_max = y_max
        self.out_of_bounds = out_of_bounds
        self.tolerance = tolerance
        self.models: list[IsotonicRegression] = []

    def fit(self, probabilities: np.ndarray, target: np.ndarray) -> IsotonicOvRCalibrator:
        matrix = validate_multiclass_probabilities(probabilities, tolerance=self.tolerance)
        labels = validate_multiclass_target(target)
        if len(matrix) != len(labels):
            raise ValueError("probabilities and target lengths must match")

        self.models = []
        for k in range(3):
            binary_target = (labels == k).astype(int)
            if np.sum(binary_target == 1) == 0 or np.sum(binary_target == 0) == 0:
                raise ValueError(f"training target for class {k} has zero variance")

            iso = IsotonicRegression(
                increasing=self.increasing,
                y_min=self.y_min,
                y_max=self.y_max,
                out_of_bounds=self.out_of_bounds,
            )
            iso.fit(matrix[:, k], binary_target)
            self.models.append(iso)

        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if not self.models or len(self.models) != 3:
            raise RuntimeError("IsotonicOvRCalibrator must be fitted before predict")

        matrix = validate_multiclass_probabilities(probabilities, tolerance=self.tolerance)
        n_samples = len(matrix)
        q = np.zeros((n_samples, 3), dtype=np.float64)

        for k in range(3):
            q[:, k] = self.models[k].predict(matrix[:, k])

        if not np.isfinite(q).all() or (q < 0).any():
            raise ValueError("isotonic output contains non-finite or negative values")

        row_sums = q.sum(axis=1, keepdims=True)
        if (row_sums <= 0).any():
            raise ValueError("isotonic produced zero or negative row sum; cannot normalize")

        normalized = q / row_sums
        return validate_multiclass_probabilities(normalized, tolerance=self.tolerance)


def instantiate_calibrator(
    method_name: str,
    config: dict[str, Any] | None = None,
) -> BaseCalibrator:
    """Instantiate a fresh calibrator object based on method name and config.

    Pipeline stage: 07_evaluation, Task 5.2.
    """
    recal_cfg = config.get("recalibration", {}) if config else {}
    tol = recal_cfg.get("probability_tolerance", 1e-12)

    if method_name == "identity":
        return IdentityCalibrator()

    if method_name == "sigmoid_ovr":
        sig_cfg = recal_cfg.get("sigmoid", {})
        return SigmoidOvRCalibrator(
            solver=sig_cfg.get("solver", "lbfgs"),
            C=float(sig_cfg.get("C", 1e9)),
            fit_intercept=bool(sig_cfg.get("fit_intercept", True)),
            max_iter=int(sig_cfg.get("max_iter", 1000)),
            random_state=sig_cfg.get("random_state", 42),
            tolerance=tol,
        )

    if method_name == "isotonic_ovr":
        iso_cfg = recal_cfg.get("isotonic", {})
        return IsotonicOvRCalibrator(
            increasing=bool(iso_cfg.get("increasing", True)),
            y_min=float(iso_cfg.get("y_min", 0.0)),
            y_max=float(iso_cfg.get("y_max", 1.0)),
            out_of_bounds=iso_cfg.get("out_of_bounds", "clip"),
            tolerance=tol,
        )

    raise ValueError(f"unrecognized calibration method: {method_name}")


def run_temporal_recalibration(
    df: pd.DataFrame,
    folds: list[TemporalFold],
    methods: list[str] = ("identity", "sigmoid_ovr", "isotonic_ovr"),
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Fit and evaluate candidate calibrators across date-safe temporal folds.

    Generates a long-format DataFrame containing only forward-evaluation fixtures.

    Pipeline stage: 07_evaluation, Task 5.2.
    """
    ordered = df.sort_values(["date", "fixture_id"]).reset_index(drop=True)
    prob_cols = list(PROBABILITY_COLUMNS)

    prediction_records: list[dict[str, Any]] = []

    for fold in folds:
        train_df = ordered.iloc[fold.train_indices]
        eval_df = ordered.iloc[fold.eval_indices]

        p_train = train_df[prob_cols].to_numpy(dtype=np.float64)
        y_train = train_df["target_encoded"].to_numpy(dtype=int)

        p_eval = eval_df[prob_cols].to_numpy(dtype=np.float64)

        for method in methods:
            calibrator = instantiate_calibrator(method, config)
            calibrator.fit(p_train, y_train)
            p_calibrated = calibrator.predict(p_eval)

            for i, eval_idx in enumerate(fold.eval_indices):
                row = ordered.iloc[eval_idx]
                prediction_records.append(
                    {
                        "fixture_id": str(row["fixture_id"]),
                        "date": row["date"],
                        "target_encoded": int(row["target_encoded"]),
                        "fold_id": fold.fold_id,
                        "calibration_method": method,
                        "calibration_train_start_date": fold.train_start_date,
                        "calibration_train_end_date": fold.train_end_date,
                        "calibration_train_count": fold.train_count,
                        "raw_prob_home_win": float(p_eval[i, 0]),
                        "raw_prob_draw": float(p_eval[i, 1]),
                        "raw_prob_away_win": float(p_eval[i, 2]),
                        "calibrated_prob_home_win": float(p_calibrated[i, 0]),
                        "calibrated_prob_draw": float(p_calibrated[i, 1]),
                        "calibrated_prob_away_win": float(p_calibrated[i, 2]),
                        "evaluation_scope": CALIBRATION_EVALUATION_SCOPE,
                    }
                )

    predictions = pd.DataFrame(prediction_records)
    predictions["date"] = pd.to_datetime(predictions["date"])
    return predictions.sort_values(["calibration_method", "date", "fixture_id"]).reset_index(drop=True)


def compute_recalibration_fold_metrics(
    forward_predictions: pd.DataFrame,
    folds: list[TemporalFold],
) -> pd.DataFrame:
    """Compute per-fold validation metrics for each calibration candidate.

    Pipeline stage: 07_evaluation, Task 5.2.
    """
    fold_map = {f.fold_id: f for f in folds}
    rows: list[dict[str, Any]] = []

    cal_prob_cols = ["calibrated_prob_home_win", "calibrated_prob_draw", "calibrated_prob_away_win"]

    for (method, fold_id), group in forward_predictions.groupby(
        ["calibration_method", "fold_id"], sort=True
    ):
        fold = fold_map[fold_id]
        probs = group[cal_prob_cols].to_numpy(dtype=np.float64)
        target = group["target_encoded"].to_numpy(dtype=int)

        brier = multiclass_brier_score(probs, target)
        ll = multiclass_log_loss(probs, target)
        acc = classification_accuracy(probs, target)

        one_hot = np.eye(3, dtype=np.float64)[target]
        binary_brier_home = float(np.mean((probs[:, 0] - one_hot[:, 0]) ** 2))
        binary_brier_draw = float(np.mean((probs[:, 1] - one_hot[:, 1]) ** 2))
        binary_brier_away = float(np.mean((probs[:, 2] - one_hot[:, 2]) ** 2))

        rows.append(
            {
                "fold_id": fold_id,
                "calibration_method": method,
                "train_count": fold.train_count,
                "eval_count": fold.eval_count,
                "train_start_date": fold.train_start_date,
                "train_end_date": fold.train_end_date,
                "eval_start_date": fold.eval_start_date,
                "eval_end_date": fold.eval_end_date,
                "multiclass_brier_score": brier,
                "log_loss": ll,
                "accuracy": acc,
                "binary_brier_home": binary_brier_home,
                "binary_brier_draw": binary_brier_draw,
                "binary_brier_away": binary_brier_away,
                "evaluation_scope": CALIBRATION_EVALUATION_SCOPE,
            }
        )

    return pd.DataFrame(rows).sort_values(["calibration_method", "fold_id"]).reset_index(drop=True)


def compute_recalibration_reliability(
    forward_predictions: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute 10-bin one-vs-rest reliability table across pooled forward predictions.

    Pipeline stage: 07_evaluation, Task 5.2.
    """
    cal_prob_cols = ["calibrated_prob_home_win", "calibrated_prob_draw", "calibrated_prob_away_win"]
    reliability_frames: list[pd.DataFrame] = []

    for method, group in forward_predictions.groupby("calibration_method", sort=True):
        probs = group[cal_prob_cols].to_numpy(dtype=np.float64)
        target = group["target_encoded"].to_numpy(dtype=int)

        for class_idx in (0, 1, 2):
            table = compute_reliability_table(probs, target, class_idx, n_bins=n_bins)
            table["calibration_method"] = method
            table["bins_per_class"] = n_bins
            table["evaluation_scope"] = CALIBRATION_EVALUATION_SCOPE
            reliability_frames.append(table)

    return pd.concat(reliability_frames, ignore_index=True)


def compute_recalibration_comparison(
    forward_predictions: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    numerical_tolerance: float = 1e-12,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, str]:
    """Compute pooled metrics, paired deltas, and select method via conservative rule.

    Conservative adoption conditions for a calibrated candidate vs identity:
    1. pooled_brier_delta < -numerical_tolerance
    2. pooled_log_loss_delta <= numerical_tolerance
    3. fold_brier_wins > n_folds / 2.0

    Pipeline stage: 07_evaluation, Task 5.2.
    """
    cal_prob_cols = ["calibrated_prob_home_win", "calibrated_prob_draw", "calibrated_prob_away_win"]

    methods = sorted(forward_predictions["calibration_method"].unique())
    if "identity" not in methods:
        raise ValueError("identity candidate must be present in forward predictions")

    # Pooled metrics directly from pooled rows
    pooled_rows: dict[str, dict[str, Any]] = {}
    for method in methods:
        m_df = forward_predictions[forward_predictions["calibration_method"] == method]
        probs = m_df[cal_prob_cols].to_numpy(dtype=np.float64)
        target = m_df["target_encoded"].to_numpy(dtype=int)

        brier = multiclass_brier_score(probs, target)
        ll = multiclass_log_loss(probs, target)
        acc = classification_accuracy(probs, target)

        eces = []
        for c in (0, 1, 2):
            rt = compute_reliability_table(probs, target, c, n_bins=n_bins)
            eces.append(expected_calibration_error(rt))

        pooled_rows[method] = {
            "calibration_method": method,
            "forward_eval_count": len(m_df),
            "eval_start_date": m_df["date"].min(),
            "eval_end_date": m_df["date"].max(),
            "multiclass_brier_score": brier,
            "log_loss": ll,
            "accuracy": acc,
            "macro_ece": float(np.mean(eces)),
            "home_win_ece": eces[0],
            "draw_ece": eces[1],
            "away_win_ece": eces[2],
        }

    # Paired comparisons against identity
    identity_brier = pooled_rows["identity"]["multiclass_brier_score"]
    identity_ll = pooled_rows["identity"]["log_loss"]

    identity_folds = fold_metrics[fold_metrics["calibration_method"] == "identity"].set_index("fold_id")
    n_folds = len(identity_folds)

    comparison_records: list[dict[str, Any]] = []

    for method in methods:
        p_data = pooled_rows[method]
        b_delta = float(p_data["multiclass_brier_score"] - identity_brier)
        ll_delta = float(p_data["log_loss"] - identity_ll)

        m_folds = fold_metrics[fold_metrics["calibration_method"] == method].set_index("fold_id")
        fold_brier_diffs = m_folds["multiclass_brier_score"] - identity_folds["multiclass_brier_score"]
        # Improvement means diff < -tolerance
        fold_wins = int((fold_brier_diffs < -numerical_tolerance).sum())
        largest_improvement = float(fold_brier_diffs.min())  # most negative = biggest improvement
        largest_deterioration = float(fold_brier_diffs.max())  # most positive = biggest worsening

        # Check eligibility
        if method == "identity":
            eligible = True
            rejection_reason = "baseline_method"
        else:
            reasons = []
            if not (b_delta < -numerical_tolerance):
                reasons.append(f"did not improve pooled Brier ({b_delta:+.6f} >= -{numerical_tolerance})")
            if not (ll_delta <= numerical_tolerance):
                reasons.append(f"worsened pooled log-loss ({ll_delta:+.6f} > {numerical_tolerance})")
            if not (fold_wins > n_folds / 2.0):
                reasons.append(f"won only {fold_wins}/{n_folds} folds (<= 50%)")

            eligible = len(reasons) == 0
            rejection_reason = "; ".join(reasons) if reasons else "meets_all_adoption_criteria"

        comparison_records.append(
            {
                **p_data,
                "brier_delta": b_delta,
                "log_loss_delta": ll_delta,
                "fold_count": n_folds,
                "fold_brier_win_count": fold_wins,
                "largest_fold_improvement": largest_improvement,
                "largest_fold_deterioration": largest_deterioration,
                "eligible_for_adoption": eligible,
                "rejection_reason": rejection_reason,
            }
        )

    comp_df = pd.DataFrame(comparison_records)

    # Sort numerically by Brier (ascending), log loss (ascending)
    comp_df = comp_df.sort_values(
        ["multiclass_brier_score", "log_loss"], ascending=[True, True]
    ).reset_index(drop=True)
    comp_df["numerical_rank"] = np.arange(1, len(comp_df) + 1)
    numerical_winner = comp_df.iloc[0]["calibration_method"]

    # Conservative selection rule
    # Filter eligible candidates
    eligible_candidates = comp_df[comp_df["eligible_for_adoption"]].copy()

    # Rule: If neither calibrated method is eligible, retain identity
    # If multiple eligible candidates tie, apply tie-breaking rules:
    # 1. Lower log loss
    # 2. Prefer sigmoid over isotonic
    # 3. Prefer identity when tied with calibrated candidate
    if len(eligible_candidates) == 1:
        selected_method = eligible_candidates.iloc[0]["calibration_method"]
    else:
        # Check if any calibrated candidate is eligible
        calibrated_eligible = eligible_candidates[eligible_candidates["calibration_method"] != "identity"]
        if calibrated_eligible.empty:
            selected_method = "identity"
        else:
            best_cal = calibrated_eligible.iloc[0]
            # Check if best calibrated candidate ties identity in Brier
            if abs(best_cal["multiclass_brier_score"] - identity_brier) <= numerical_tolerance:
                selected_method = "identity"
            else:
                selected_method = best_cal["calibration_method"]

    comp_df["is_selected_method"] = comp_df["calibration_method"] == selected_method
    comp_df["evaluation_scope"] = CALIBRATION_EVALUATION_SCOPE

    return comp_df, selected_method


def save_recalibration_artifacts(
    fold_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    reliability_data: pd.DataFrame,
    forward_predictions: pd.DataFrame,
    selected_method: str,
    output_dir: Path,
) -> dict[str, Path]:
    """Persist and verify all 5 Task 5.2 recalibration Parquet artifacts.

    Pipeline stage: 07_evaluation, Task 5.2.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "fold_metrics": output_dir / "recalibration_fold_metrics.parquet",
        "comparison": output_dir / "recalibration_comparison.parquet",
        "reliability_data": output_dir / "recalibration_reliability_data.parquet",
        "forward_predictions": output_dir / "recalibration_forward_predictions.parquet",
        "recalibrated_probabilities": output_dir / "recalibrated_probabilities.parquet",
    }

    selected_preds = forward_predictions[
        forward_predictions["calibration_method"] == selected_method
    ].copy()

    fold_metrics.to_parquet(paths["fold_metrics"], index=False)
    comparison.to_parquet(paths["comparison"], index=False)
    reliability_data.to_parquet(paths["reliability_data"], index=False)
    forward_predictions.to_parquet(paths["forward_predictions"], index=False)
    selected_preds.to_parquet(paths["recalibrated_probabilities"], index=False)

    # Verification read
    reread_fold = pd.read_parquet(paths["fold_metrics"])
    reread_comp = pd.read_parquet(paths["comparison"])
    reread_rel = pd.read_parquet(paths["reliability_data"])
    reread_all_preds = pd.read_parquet(paths["forward_predictions"])
    reread_sel_preds = pd.read_parquet(paths["recalibrated_probabilities"])

    if len(reread_comp[reread_comp["is_selected_method"]]) != 1:
        raise ValueError("comparison table must have exactly one selected method")

    if reread_sel_preds["fixture_id"].duplicated().any():
        raise ValueError("recalibrated probabilities fixture IDs must be unique")

    n_sel = len(reread_sel_preds)
    expected_all_preds = n_sel * len(reread_comp)
    if len(reread_all_preds) != expected_all_preds:
        raise ValueError(
            f"forward predictions count mismatch: expected {expected_all_preds}, got {len(reread_all_preds)}"
        )

    cal_cols = ["calibrated_prob_home_win", "calibrated_prob_draw", "calibrated_prob_away_win"]
    validate_multiclass_probabilities(reread_sel_preds[cal_cols].to_numpy(dtype=np.float64))

    return paths
