"""OOF ensemble evaluation for Monte Carlo and XGBoost probabilities.

Pipeline Stage: 06_ml, Task 4.3
Inputs: OOF Monte Carlo/XGBoost probability tables and target outcomes.
Outputs: OOF ensemble sweep and selected ensemble probability tables.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from src.ml.xgboost_model import compute_multiclass_brier_score


PROBABILITY_COLUMNS = ["home_win", "draw", "away_win"]


def validate_probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    """Validate and normalize a three-class probability matrix.

    Pipeline stage: 06_ml, Task 4.3.
    """
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("probabilities must have shape (n_rows, 3)")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = matrix.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("probability rows must have a positive sum")
    matrix = matrix / row_sums
    matrix[:, -1] = 1.0 - matrix[:, :-1].sum(axis=1)
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("probability rows do not sum to one")
    return matrix


def blend_probabilities(
    mc_probabilities: np.ndarray,
    xgb_probabilities: np.ndarray,
    xgb_weight: float,
) -> np.ndarray:
    """Blend normalized MC and XGBoost 1X2 probabilities.

    Pipeline stage: 06_ml, Task 4.3.
    """
    if not 0.0 <= xgb_weight <= 1.0:
        raise ValueError("xgb_weight must be between 0 and 1")
    mc = validate_probability_matrix(mc_probabilities)
    xgb = validate_probability_matrix(xgb_probabilities)
    if mc.shape != xgb.shape:
        raise ValueError("MC and XGBoost probability matrices must have equal shape")
    return validate_probability_matrix((1.0 - xgb_weight) * mc + xgb_weight * xgb)


def evaluate_probability_predictions(
    probabilities: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    """Evaluate accuracy, log loss, and summed multiclass Brier score.

    Pipeline stage: 06_ml, Task 4.3.
    """
    normalized = validate_probability_matrix(probabilities)
    target_array = np.asarray(target, dtype=int)
    if len(target_array) != len(normalized):
        raise ValueError("target and probability row counts must match")
    predictions = np.argmax(normalized, axis=1)
    return {
        "accuracy": float(accuracy_score(target_array, predictions)),
        "log_loss": float(log_loss(target_array, normalized, labels=[0, 1, 2])),
        "brier_score": compute_multiclass_brier_score(target_array, normalized),
    }


def sweep_ensemble_weights(
    mc_probabilities: np.ndarray,
    xgb_probabilities: np.ndarray,
    target: np.ndarray,
    weights: list[float],
    candidate: str = "ensemble",
) -> pd.DataFrame:
    """Evaluate configured XGBoost weights on one common OOF population.

    Pipeline stage: 06_ml, Task 4.3.
    """
    records = []
    for weight in weights:
        probabilities = blend_probabilities(mc_probabilities, xgb_probabilities, float(weight))
        metrics = evaluate_probability_predictions(probabilities, target)
        records.append(
            {
                "candidate": candidate,
                "xgb_weight": float(weight),
                "mc_weight": 1.0 - float(weight),
                "oof_fixture_count": len(target),
                **metrics,
            }
        )
    return pd.DataFrame(records)


def select_best_candidate(
    candidate_results: pd.DataFrame,
    primary_metric: str,
    tie_break_metric: str,
    tolerance: float,
) -> pd.Series:
    """Select a candidate using metric, tie-break, and simplicity rules.

    Pipeline stage: 06_ml, Task 4.3.
    """
    required = {"candidate", "xgb_weight", primary_metric, tie_break_metric}
    missing = required.difference(candidate_results.columns)
    if missing:
        raise ValueError(f"candidate_results missing columns: {sorted(missing)}")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    best_primary = float(candidate_results[primary_metric].min())
    tied = candidate_results[candidate_results[primary_metric] <= best_primary + tolerance].copy()
    best_tie = float(tied[tie_break_metric].min())
    tied = tied[tied[tie_break_metric] <= best_tie + tolerance].copy()

    def simplicity(row: pd.Series) -> tuple[int, int]:
        candidate = str(row["candidate"])
        priority = {
            "mc_only": 0,
            "context_only": 1,
            "full_xgb_only": 2,
        }.get(candidate, 3)
        return priority, 0

    tied["_simplicity"] = tied.apply(simplicity, axis=1)
    return tied.sort_values("_simplicity", kind="stable").iloc[0].drop(labels=["_simplicity"])


def align_oof_inputs(
    mc_probabilities: pd.DataFrame,
    xgb_probabilities: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> pd.DataFrame:
    """Align MC/XGBoost OOF probabilities with date and target validation.

    Pipeline stage: 06_ml, Task 4.3. Rows without valid OOF XGBoost
    probabilities are excluded explicitly; conflicting IDs or targets raise.
    """
    mc_columns = ["fixture_id", "mc_prob_home_win", "mc_prob_draw", "mc_prob_away_win"]
    xgb_columns = ["fixture_id", "xgb_prob_home_win", "xgb_prob_draw", "xgb_prob_away_win"]
    feature_columns = ["fixture_id", "date", "target_encoded"]
    for frame, columns, name in [
        (mc_probabilities, mc_columns, "MC"),
        (xgb_probabilities, xgb_columns, "XGBoost"),
        (feature_table, feature_columns, "feature table"),
    ]:
        missing = set(columns).difference(frame.columns)
        if missing:
            raise ValueError(f"{name} input missing columns: {sorted(missing)}")

    feature_ids = set(feature_table["fixture_id"])
    mc_ids = set(mc_probabilities["fixture_id"])
    xgb_ids = set(xgb_probabilities["fixture_id"])
    unexpected_xgb_ids = sorted(xgb_ids.difference(feature_ids | mc_ids))
    if unexpected_xgb_ids:
        raise ValueError(f"XGBoost fixture IDs are not present in the other inputs: {unexpected_xgb_ids[:5]}")
    if "target_encoded" in xgb_probabilities.columns:
        target_check = feature_table[["fixture_id", "target_encoded"]].merge(
            xgb_probabilities[["fixture_id", "target_encoded"]],
            on="fixture_id",
            how="inner",
            suffixes=("_feature", "_xgb"),
        )
        conflicting = target_check[
            target_check["target_encoded_xgb"].notna()
            & (target_check["target_encoded_feature"] != target_check["target_encoded_xgb"])
        ]
        if not conflicting.empty:
            raise ValueError("Feature table and XGBoost targets disagree for aligned fixtures")

    aligned = feature_table[feature_columns].merge(mc_probabilities[mc_columns], on="fixture_id", how="inner", validate="one_to_one")
    aligned = aligned.merge(xgb_probabilities[xgb_columns], on="fixture_id", how="inner", validate="one_to_one")
    valid = aligned[["xgb_prob_home_win", "xgb_prob_draw", "xgb_prob_away_win"]].notna().all(axis=1)
    aligned = aligned.loc[valid].sort_values(["date", "fixture_id"]).reset_index(drop=True)
    if aligned.empty:
        raise ValueError("No common OOF probability rows remain after alignment")
    for prefix in ("mc", "xgb"):
        probabilities = aligned[[f"{prefix}_prob_home_win", f"{prefix}_prob_draw", f"{prefix}_prob_away_win"]].to_numpy()
        validate_probability_matrix(probabilities)
    return aligned
