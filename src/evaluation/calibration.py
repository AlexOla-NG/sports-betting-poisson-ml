"""Calibration evaluation for the selected Task 4.3 OOF ensemble.

Pipeline Stage: 07_evaluation, Task 5.1
This module evaluates persisted probabilities without recalibration or model
refitting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

CLASS_LABELS = {0: "home_win", 1: "draw", 2: "away_win"}
PROBABILITY_COLUMNS = (
    "ensemble_prob_home_win",
    "ensemble_prob_draw",
    "ensemble_prob_away_win",
)


def validate_multiclass_probabilities(
    probabilities: np.ndarray,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Validate unchanged three-class probability rows.

    Pipeline stage: 07_evaluation, Task 5.1.
    """
    matrix = np.asarray(probabilities, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError("probabilities must have shape (n_samples, 3)")
    if matrix.shape[0] == 0:
        raise ValueError("probabilities cannot be empty")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    row_sums = matrix.sum(axis=1)
    if not np.all(np.abs(row_sums - 1.0) <= tolerance):
        raise ValueError("probability rows must sum to one within tolerance")
    return matrix


def validate_multiclass_target(target: np.ndarray) -> np.ndarray:
    """Validate and return integer Home/Draw/Away target classes.

    Pipeline stage: 07_evaluation, Task 5.1.
    """
    values = np.asarray(target)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("target must be a non-empty one-dimensional array")
    if pd.isna(values).any():
        raise ValueError("target contains null values")
    numeric = pd.to_numeric(values, errors="coerce")
    if pd.isna(numeric).any() or not np.all(np.equal(numeric, np.floor(numeric))):
        raise ValueError("target values must be integer-like")
    integer_target = numeric.astype(int)
    if not np.isin(integer_target, list(CLASS_LABELS)).all():
        raise ValueError("target values must be in {0, 1, 2}")
    return integer_target


def _paired_inputs(probabilities: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired probability and target arrays for calibration metrics."""
    matrix = validate_multiclass_probabilities(probabilities)
    labels = validate_multiclass_target(target)
    if len(matrix) != len(labels):
        raise ValueError("probability and target lengths must match")
    return matrix, labels


def multiclass_brier_score(probabilities: np.ndarray, target: np.ndarray) -> float:
    """Compute the project's summed multiclass Brier score.

    Pipeline stage: 07_evaluation, Task 5.1.
    """
    matrix, labels = _paired_inputs(probabilities, target)
    one_hot = np.eye(3, dtype=np.float64)[labels]
    return float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1)))


def multiclass_log_loss(
    probabilities: np.ndarray,
    target: np.ndarray,
    epsilon: float = 1e-15,
) -> float:
    """Compute multiclass log loss with clipping only inside the logarithm.

    Pipeline stage: 07_evaluation, Task 5.1.
    """
    matrix, labels = _paired_inputs(probabilities, target)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return float(-np.mean(np.log(np.clip(matrix[np.arange(len(labels)), labels], epsilon, 1.0))))


def classification_accuracy(probabilities: np.ndarray, target: np.ndarray) -> float:
    """Compute argmax accuracy under Home=0, Draw=1, Away=2 ordering.

    Pipeline stage: 07_evaluation, Task 5.1.
    """
    matrix, labels = _paired_inputs(probabilities, target)
    return float(np.mean(np.argmax(matrix, axis=1) == labels))


def compute_reliability_table(
    probabilities: np.ndarray,
    target: np.ndarray,
    class_index: int,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute fixed-width one-vs-rest reliability bins for one class.

    Pipeline stage: 07_evaluation, Task 5.1.
    """
    matrix, labels = _paired_inputs(probabilities, target)
    if class_index not in CLASS_LABELS:
        raise ValueError("class_index must be 0, 1, or 2")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    predicted = matrix[:, class_index]
    bin_indices = np.minimum((predicted * n_bins).astype(int), n_bins - 1)
    rows = []
    for bin_index in range(n_bins):
        selected = bin_indices == bin_index
        count = int(selected.sum())
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        if count:
            mean_probability = float(predicted[selected].mean())
            observed = float((labels[selected] == class_index).mean())
            gap = abs(observed - mean_probability)
        else:
            mean_probability = np.nan
            observed = np.nan
            gap = np.nan
        rows.append(
            {
                "class_index": class_index,
                "class_label": CLASS_LABELS[class_index],
                "bin_index": bin_index,
                "bin_lower": lower,
                "bin_upper": upper,
                "fixture_count": count,
                "mean_predicted_probability": mean_probability,
                "observed_frequency": observed,
                "absolute_calibration_gap": gap,
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(reliability_table: pd.DataFrame) -> float:
    """Compute ECE over non-empty bins as a diagnostic, not a scoring rule.

    Pipeline stage: 07_evaluation, Task 5.1.
    """
    required = {"fixture_count", "absolute_calibration_gap"}
    missing = required.difference(reliability_table.columns)
    if missing:
        raise ValueError(f"reliability table missing columns: {sorted(missing)}")
    non_empty = reliability_table[reliability_table["fixture_count"] > 0].copy()
    if non_empty.empty:
        raise ValueError("reliability table has no non-empty bins")
    if not np.isfinite(non_empty["absolute_calibration_gap"]).all():
        raise ValueError("non-empty reliability bins must have finite gaps")
    total = non_empty["fixture_count"].sum()
    return float((non_empty["fixture_count"] * non_empty["absolute_calibration_gap"]).sum() / total)


def compute_per_class_calibration_summary(
    probabilities: np.ndarray,
    target: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute one calibration summary row for each outcome class.

    Pipeline stage: 07_evaluation, Task 5.1.
    """
    matrix, labels = _paired_inputs(probabilities, target)
    rows = []
    for class_index, class_label in CLASS_LABELS.items():
        table = compute_reliability_table(matrix, labels, class_index, n_bins)
        non_empty = table[table["fixture_count"] > 0]
        rows.append(
            {
                "class_index": class_index,
                "class_label": class_label,
                "fixture_count": len(labels),
                "actual_event_rate": float((labels == class_index).mean()),
                "mean_predicted_probability": float(matrix[:, class_index].mean()),
                "binary_brier_score": float(np.mean((matrix[:, class_index] - (labels == class_index)) ** 2)),
                "expected_calibration_error": expected_calibration_error(table),
                "max_absolute_calibration_gap": float(non_empty["absolute_calibration_gap"].max()),
                "non_empty_bin_count": int(len(non_empty)),
                "empty_bin_count": int(n_bins - len(non_empty)),
            }
        )
    return pd.DataFrame(rows)


def compute_chronological_brier_trend(
    predictions: pd.DataFrame,
    date_column: str = "date",
    target_column: str = "target_encoded",
    probability_columns: tuple[str, str, str] = PROBABILITY_COLUMNS,
    frequency: str = "MS",
) -> pd.DataFrame:
    """Compute non-forward-looking calendar-month probability-quality metrics.

    Pipeline stage: 07_evaluation, Task 5.1.
    """
    required = {date_column, target_column, *probability_columns}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")
    ordered = predictions.copy()
    ordered[date_column] = pd.to_datetime(ordered[date_column], errors="coerce")
    if ordered[date_column].isna().any():
        raise ValueError("predictions contain invalid dates")
    ordered = ordered.sort_values([date_column, "fixture_id"] if "fixture_id" in ordered else [date_column])
    rows = []
    for period, group in ordered.groupby(ordered[date_column].dt.to_period("M"), sort=True):
        matrix = group[list(probability_columns)].to_numpy(dtype=np.float64)
        labels = group[target_column].to_numpy()
        metrics = {
            "period": period.to_timestamp(),
            "fixture_count": len(group),
            "start_date": group[date_column].min(),
            "end_date": group[date_column].max(),
            "multiclass_brier_score": multiclass_brier_score(matrix, labels),
            "log_loss": multiclass_log_loss(matrix, labels),
            "accuracy": classification_accuracy(matrix, labels),
        }
        rows.append(metrics)
    result = pd.DataFrame(rows)
    if result.empty or int(result["fixture_count"].sum()) != len(ordered):
        raise ValueError("chronological trend fixture counts do not match input")
    return result


def validate_selected_ensemble_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate the persisted Task 4.3 selected OOF ensemble input.

    Pipeline stage: 07_evaluation, Task 5.1.
    """
    required = {"fixture_id", "date", "target_encoded", *PROBABILITY_COLUMNS, "selected_candidate", "selected_xgb_weight", "selected_mc_weight"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"selected ensemble is missing columns: {sorted(missing)}")
    if len(predictions) != 560:
        raise ValueError(f"expected 560 selected OOF rows, got {len(predictions)}")
    if predictions["fixture_id"].isna().any() or predictions["fixture_id"].duplicated().any():
        raise ValueError("fixture_id must be present and unique")
    if predictions["selected_candidate"].nunique() != 1 or predictions["selected_candidate"].iloc[0] != "blend_context":
        raise ValueError("selected candidate must be constant blend_context")
    if not np.isclose(predictions["selected_xgb_weight"], 0.8, atol=1e-12).all() or not np.isclose(predictions["selected_mc_weight"], 0.2, atol=1e-12).all():
        raise ValueError("selected weights must be constant 0.80 XGBoost and 0.20 MC")
    matrix = predictions[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    validate_multiclass_probabilities(matrix)
    validate_multiclass_target(predictions["target_encoded"].to_numpy())
    if np.allclose(matrix, matrix[0], atol=1e-15):
        raise ValueError("selected probability vectors are degenerate")
    result = predictions.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    return result.sort_values(["date", "fixture_id"]).reset_index(drop=True)


def build_calibration_report(
    predictions: pd.DataFrame,
    n_bins: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build overall, per-class, and reliability calibration tables.

    Pipeline stage: 07_evaluation, Task 5.1.
    """
    ordered = validate_selected_ensemble_frame(predictions)
    matrix = ordered[list(PROBABILITY_COLUMNS)].to_numpy(dtype=np.float64)
    target = validate_multiclass_target(ordered["target_encoded"].to_numpy())
    reliability = pd.concat(
        [compute_reliability_table(matrix, target, class_index, n_bins) for class_index in CLASS_LABELS],
        ignore_index=True,
    )
    per_class = compute_per_class_calibration_summary(matrix, target, n_bins)
    overall = pd.DataFrame([{
        "selected_candidate": ordered["selected_candidate"].iloc[0],
        "selected_xgb_weight": ordered["selected_xgb_weight"].iloc[0],
        "selected_mc_weight": ordered["selected_mc_weight"].iloc[0],
        "oof_fixture_count": len(ordered),
        "start_date": ordered["date"].min(),
        "end_date": ordered["date"].max(),
        "multiclass_brier_score": multiclass_brier_score(matrix, target),
        "log_loss": multiclass_log_loss(matrix, target),
        "accuracy": classification_accuracy(matrix, target),
        "macro_expected_calibration_error": per_class["expected_calibration_error"].mean(),
        "home_win_ece": per_class.loc[per_class["class_label"] == "home_win", "expected_calibration_error"].iloc[0],
        "draw_ece": per_class.loc[per_class["class_label"] == "draw", "expected_calibration_error"].iloc[0],
        "away_win_ece": per_class.loc[per_class["class_label"] == "away_win", "expected_calibration_error"].iloc[0],
        "calibration_bin_count": len(reliability),
        "evaluation_scope": "oof_pre_calibration",
    }])
    metadata = ordered[["selected_candidate", "selected_xgb_weight", "selected_mc_weight"]].iloc[0].to_dict()
    per_class = per_class.assign(**metadata, evaluation_scope="oof_pre_calibration")
    reliability = reliability.assign(**metadata, evaluation_scope="oof_pre_calibration")
    return overall, per_class, reliability


def save_calibration_artifacts(
    overall_metrics: pd.DataFrame,
    per_class_summary: pd.DataFrame,
    reliability_data: pd.DataFrame,
    chronological_trend: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write and re-read the three Task 5.1 calibration artifacts.

    Pipeline stage: 07_evaluation, Task 5.1.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat(
        [overall_metrics.assign(metric_scope="overall"), per_class_summary.assign(metric_scope="per_class")],
        ignore_index=True,
        sort=False,
    )
    paths = {
        "calibration_metrics": output_dir / "calibration_metrics.parquet",
        "reliability_diagram_data": output_dir / "reliability_diagram_data.parquet",
        "brier_score_trend": output_dir / "brier_score_trend.parquet",
    }
    metrics.to_parquet(paths["calibration_metrics"], index=False)
    reliability_data.to_parquet(paths["reliability_diagram_data"], index=False)
    chronological_trend.to_parquet(paths["brier_score_trend"], index=False)
    reread_metrics = pd.read_parquet(paths["calibration_metrics"])
    reread_reliability = pd.read_parquet(paths["reliability_diagram_data"])
    reread_trend = pd.read_parquet(paths["brier_score_trend"])
    if len(reread_metrics[reread_metrics["metric_scope"] == "overall"]) != 1 or len(reread_metrics[reread_metrics["metric_scope"] == "per_class"]) != 3:
        raise ValueError("calibration metrics artifact has unexpected row counts")
    if len(reread_reliability) != 30 or reread_trend.empty or int(reread_trend["fixture_count"].sum()) != 560:
        raise ValueError("calibration artifact row counts are invalid")
    return paths
