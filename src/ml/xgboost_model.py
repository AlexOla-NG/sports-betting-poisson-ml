"""Pipeline Stage: Phase 4 — ML Layer (Task 4.2: XGBoost Model)

This module provides walk-forward validation and training for an XGBoost multi-class
classifier predicting football match outcomes (Home Win, Draw, Away Win).
It strictly enforces chronological walk-forward splits, reads all hyperparameters from
config.yaml, supports native NaN handling, and exports trained model artifacts alongside
gain-based feature importances.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

from src.config.loader import get_walk_forward_params, get_xgboost_params


DEFAULT_EXCLUDE_COLUMNS = frozenset(
    {
        "fixture_id",
        "league",
        "season",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "target",
        "target_encoded",
    }
)

POISSON_MC_FEATURES = frozenset(
    {
        "poisson_rating_diff",
        "poisson_attack_diff",
        "poisson_defense_diff",
        "home_attack_rating",
        "home_defense_rating",
        "away_attack_rating",
        "away_defense_rating",
        "expected_home_goals",
        "expected_away_goals",
        "base_home_lambda",
        "base_away_lambda",
        "adjusted_home_lambda",
        "adjusted_away_lambda",
        "adjusted_lambda_diff",
        "mc_prob_home_win",
        "mc_prob_draw",
        "mc_prob_away_win",
        "mc_prob_over_2_5",
        "mc_prob_under_2_5",
        "mc_prob_btts",
    }
)


def generate_walk_forward_splits(
    n_samples: int, initial_train_size: int = 200, step_size: int = 38
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate chronological, non-overlapping expanding-window walk-forward splits.

    Pipeline Stage: Phase 4 — ML Layer

    Parameters
    ----------
    n_samples : int
        Total number of chronological match rows in the dataset.
    initial_train_size : int, optional
        Number of matches in the initial training fold, by default 200.
    step_size : int, optional
        Step size (number of matches in each test fold), by default 38.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        List of (train_indices, test_indices) tuples. Each test split is strictly
        non-overlapping and ordered chronologically after the expanding train split.
    """
    if initial_train_size <= 0:
        raise ValueError("initial_train_size must be positive.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if initial_train_size >= n_samples:
        raise ValueError(
            f"initial_train_size ({initial_train_size}) must be less than n_samples ({n_samples})."
        )

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    test_start = initial_train_size

    while test_start < n_samples:
        train_idx = np.arange(0, test_start)
        test_end = min(test_start + step_size, n_samples)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))
        test_start = test_end

    return splits


def compute_multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute the multi-class Brier score across predicted class probability vectors.

    Pipeline Stage: Phase 4 — ML Layer

    Parameters
    ----------
    y_true : np.ndarray
        Array of true class indices of shape (N,).
    y_prob : np.ndarray
        Array of predicted probabilities of shape (N, K).

    Returns
    -------
    float
        Mean multi-class Brier score: (1/N) * sum_i sum_k (p_ik - y_ik)^2.
    """
    n_samples, n_classes = y_prob.shape
    y_one_hot = np.zeros((n_samples, n_classes), dtype=float)
    for c in range(n_classes):
        y_one_hot[:, c] = (y_true == c).astype(float)
    return float(np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1)))


def _normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Normalize float64 class probabilities row-wise for ML validation."""
    normalized = np.asarray(probabilities, dtype=np.float64)
    row_sums = normalized.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(normalized)) or np.any(row_sums <= 0):
        raise ValueError("Predicted probabilities contain invalid rows")
    normalized = normalized / row_sums
    normalized[:, -1] = 1.0 - normalized[:, :-1].sum(axis=1)
    if not np.allclose(normalized.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("Probability rows do not sum to one after normalization")
    return normalized


def _model_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Build the fixed configured XGBoost model arguments for OOF validation."""
    kwargs = {
        **get_xgboost_params(config),
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
    }
    kwargs.setdefault("random_state", config.get("monte_carlo", {}).get("random_seed", 42))
    return kwargs


def _prepare_xgboost_features(features: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean feature columns to numeric values for XGBoost."""
    prepared = features.copy()
    for column in prepared.select_dtypes(include="bool").columns:
        prepared[column] = prepared[column].astype(float)
    return prepared


def compute_oof_permutation_importance(
    features: pd.DataFrame,
    target: pd.Series,
    config: dict[str, Any],
    feature_names: list[str] | None = None,
    n_repeats: int = 10,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Compute held-out walk-forward permutation importance.

    Each fold fits only on its chronological training rows and permutes one
    feature only within that fold's held-out test rows. Positive metric deltas
    mean shuffling degraded predictive performance.
    """
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")
    prepared = _prepare_xgboost_features(features)
    names = feature_names or list(prepared.columns)
    missing = set(names).difference(prepared.columns)
    if missing:
        raise ValueError(f"Permutation features missing from input: {sorted(missing)}")

    initial_train_size, step_size = get_walk_forward_params(config)
    splits = generate_walk_forward_splits(len(prepared), initial_train_size, step_size)
    seed = random_seed if random_seed is not None else config.get("monte_carlo", {}).get("random_seed", 42)
    records: list[dict[str, Any]] = []
    probability_sums: list[float] = []

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        model = xgb.XGBClassifier(**_model_kwargs(config))
        model.fit(prepared.iloc[train_idx], target.iloc[train_idx].to_numpy())
        test_features = prepared.iloc[test_idx].copy()
        test_target = target.iloc[test_idx].to_numpy()
        baseline_probabilities = _normalize_probabilities(model.predict_proba(test_features))
        baseline_log_loss = log_loss(test_target, baseline_probabilities, labels=[0, 1, 2])
        baseline_brier = compute_multiclass_brier_score(test_target, baseline_probabilities)

        for feature_number, feature in enumerate(names):
            for repeat in range(n_repeats):
                rng = np.random.default_rng(seed + fold_number * 100_003 + feature_number * 1_009 + repeat)
                permuted = test_features.copy()
                permuted[feature] = rng.permutation(permuted[feature].to_numpy())
                permuted_probabilities = _normalize_probabilities(model.predict_proba(permuted))
                records.append(
                    {
                        "feature": feature,
                        "fold": fold_number,
                        "repeat": repeat,
                        "delta_log_loss": log_loss(test_target, permuted_probabilities, labels=[0, 1, 2]) - baseline_log_loss,
                        "delta_brier": compute_multiclass_brier_score(test_target, permuted_probabilities) - baseline_brier,
                    }
                )

    detail = pd.DataFrame(records)
    summary = (
        detail.groupby("feature", as_index=False)
        .agg(
            mean_delta_log_loss=("delta_log_loss", "mean"),
            std_delta_log_loss=("delta_log_loss", "std"),
            mean_delta_brier=("delta_brier", "mean"),
            std_delta_brier=("delta_brier", "std"),
            number_of_folds=("fold", "nunique"),
            n_repeats=("repeat", "nunique"),
        )
    )
    summary["std_delta_log_loss"] = summary["std_delta_log_loss"].fillna(0.0)
    summary["std_delta_brier"] = summary["std_delta_brier"].fillna(0.0)
    summary.attrs["detail"] = detail
    summary.attrs["evaluated_rows"] = int(sum(len(test_idx) for _, test_idx in splits))
    summary.attrs["test_indices"] = np.concatenate([test_idx for _, test_idx in splits])
    return summary


def evaluate_feature_ablations(
    features: pd.DataFrame,
    target: pd.Series,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Evaluate fixed-feature ablations using identical OOF walk-forward folds."""
    all_features = list(features.columns)
    family_missing = sorted(POISSON_MC_FEATURES.difference(all_features))
    context_features = [
        column for column in all_features
        if column not in POISSON_MC_FEATURES
    ]
    variants = {
        "full_model": all_features,
        "without_poisson_rating_diff": [c for c in all_features if c != "poisson_rating_diff"],
        "without_poisson_mc_family": [c for c in all_features if c not in POISSON_MC_FEATURES],
        "xg_form_context_only": context_features,
    }
    initial_train_size, step_size = get_walk_forward_params(config)
    splits = generate_walk_forward_splits(len(features), initial_train_size, step_size)
    prepared = _prepare_xgboost_features(features)
    records: list[dict[str, Any]] = []
    probability_sums: list[float] = []

    for variant, columns in variants.items():
        model_kwargs = _model_kwargs(config)
        probabilities = np.full((len(prepared), 3), np.nan, dtype=np.float64)
        predictions = np.full(len(prepared), np.nan, dtype=np.float64)
        for train_idx, test_idx in splits:
            model = xgb.XGBClassifier(**model_kwargs)
            model.fit(prepared.iloc[train_idx][columns], target.iloc[train_idx].to_numpy())
            fold_probabilities = _normalize_probabilities(model.predict_proba(prepared.iloc[test_idx][columns]))
            probability_sums.extend(fold_probabilities.sum(axis=1).tolist())
            probabilities[test_idx] = fold_probabilities
            predictions[test_idx] = np.argmax(fold_probabilities, axis=1)
        evaluation_mask = np.isfinite(probabilities[:, 0])
        eval_target = target.iloc[evaluation_mask].to_numpy()
        eval_probabilities = probabilities[evaluation_mask]
        records.append(
            {
                "variant": variant,
                "number_of_features": len(columns),
                "oof_accuracy": accuracy_score(eval_target, predictions[evaluation_mask]),
                "oof_log_loss": log_loss(eval_target, eval_probabilities, labels=[0, 1, 2]),
                "oof_brier_score": compute_multiclass_brier_score(eval_target, eval_probabilities),
            }
        )

    result = pd.DataFrame(records)
    full = result.loc[result["variant"] == "full_model"].iloc[0]
    result["delta_log_loss_vs_full"] = result["oof_log_loss"] - full["oof_log_loss"]
    result["delta_brier_vs_full"] = result["oof_brier_score"] - full["oof_brier_score"]
    result.attrs["missing_poisson_mc_family_features"] = family_missing
    result.attrs["feature_sets"] = variants
    result.attrs["splits"] = splits
    result.attrs["probability_sums"] = np.asarray(probability_sums, dtype=np.float64)
    return result


def get_feature_columns(
    df: pd.DataFrame, exclude_cols: frozenset[str] | set[str] | None = None
) -> list[str]:
    """Identify eligible numeric/boolean feature columns from a feature table.

    Pipeline Stage: Phase 4 — ML Layer

    Parameters
    ----------
    df : pd.DataFrame
        Input feature table DataFrame.
    exclude_cols : frozenset[str] | set[str] | None, optional
        Columns to exclude from training, by default DEFAULT_EXCLUDE_COLUMNS.

    Returns
    -------
    list[str]
        Ordered list of feature column names.
    """
    if exclude_cols is None:
        exclude_cols = DEFAULT_EXCLUDE_COLUMNS

    return [col for col in df.columns if col not in exclude_cols]


def train_xgboost(
    features: pd.DataFrame,
    target: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Train XGBoost multi-class classifier using chronological walk-forward validation.

    Pipeline Stage: Phase 4 — ML Layer

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame of feature rows (must be sorted chronologically).
    target : pd.Series
        Series of encoded outcome targets (0: Home, 1: Draw, 2: Away).
    config : dict[str, Any]
        Central configuration dictionary containing `ml.xgboost` and `evaluation` sections.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'final_model': Trained XGBClassifier on all available training data
        - 'fold_metrics': DataFrame with accuracy, log loss, and Brier score per fold
        - 'oof_probabilities': (N, 3) ndarray of out-of-fold probability predictions
        - 'oof_predictions': (N,) ndarray of out-of-fold predicted class labels
        - 'feature_importance': Series of gain-based feature importances (descending)
        - 'splits': List of (train_idx, test_idx) tuples used during walk-forward
        - 'feature_names': List of feature column names used for training
    """
    xgb_params = get_xgboost_params(config)
    initial_train_size, step_size = get_walk_forward_params(config)

    # Cast boolean columns to float so XGBoost handles them natively without warning
    x_matrix = features.copy()
    for col in x_matrix.select_dtypes(include="bool").columns:
        x_matrix[col] = x_matrix[col].astype(float)

    y_vector = target.to_numpy()
    n_samples = len(x_matrix)

    splits = generate_walk_forward_splits(
        n_samples=n_samples,
        initial_train_size=initial_train_size,
        step_size=step_size,
    )

    oof_probabilities = np.full((n_samples, 3), np.nan, dtype=float)
    oof_predictions = np.full(n_samples, np.nan, dtype=float)

    fold_records: list[dict[str, Any]] = []

    # Common XGBoost multi-class classifier configuration
    model_kwargs = {
        **xgb_params,
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
    }
    # Set random_state if present in config
    random_seed = config.get("monte_carlo", {}).get("random_seed", 42)
    model_kwargs.setdefault("random_state", random_seed)

    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        x_train, y_train = x_matrix.iloc[train_idx], y_vector[train_idx]
        x_test, y_test = x_matrix.iloc[test_idx], y_vector[test_idx]

        fold_clf = xgb.XGBClassifier(**model_kwargs)
        fold_clf.fit(x_train, y_train)
        probs = fold_clf.predict_proba(x_test).astype(np.float64)
        # Ensure exact probability simplex normalization to avoid float32 precision warnings
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)

        oof_probabilities[test_idx] = probs
        oof_predictions[test_idx] = preds

        acc = float(accuracy_score(y_test, preds))
        ll = float(log_loss(y_test, probs, labels=[0, 1, 2]))
        brier = compute_multiclass_brier_score(y_test, probs)

        fold_records.append(
            {
                "fold": fold_num,
                "train_start": int(train_idx[0]),
                "train_end": int(train_idx[-1]),
                "train_size": len(train_idx),
                "test_start": int(test_idx[0]),
                "test_end": int(test_idx[-1]),
                "test_size": len(test_idx),
                "accuracy": acc,
                "log_loss": ll,
                "brier_score": brier,
            }
        )

    fold_metrics = pd.DataFrame(fold_records)

    # Fit final model on the complete dataset
    final_model = xgb.XGBClassifier(**model_kwargs)
    final_model.fit(x_matrix, y_vector)

    # Extract gain-based feature importances
    booster = final_model.get_booster()
    score_dict = booster.get_score(importance_type="gain")
    # Include any zero-importance features as 0.0
    all_scores = {col: float(score_dict.get(col, 0.0)) for col in x_matrix.columns}
    feature_importance = pd.Series(all_scores).sort_values(ascending=False)

    return {
        "final_model": final_model,
        "fold_metrics": fold_metrics,
        "oof_probabilities": oof_probabilities,
        "oof_predictions": oof_predictions,
        "feature_importance": feature_importance,
        "splits": splits,
        "feature_names": list(x_matrix.columns),
    }


def save_model_artifact(model: xgb.XGBClassifier, output_path: str | Path) -> None:
    """Save trained XGBoost model artifact to disk.

    Pipeline Stage: Phase 4 — ML Layer

    Parameters
    ----------
    model : xgb.XGBClassifier
        Trained XGBoost classifier instance.
    output_path : str | Path
        Target file path (.json format recommended by XGBoost).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load_model_artifact(input_path: str | Path) -> xgb.XGBClassifier:
    """Load trained XGBoost model artifact from disk.

    Pipeline Stage: Phase 4 — ML Layer

    Parameters
    ----------
    input_path : str | Path
        Path to saved XGBoost model file (.json).

    Returns
    -------
    xgb.XGBClassifier
        Instantiated and restored XGBClassifier model.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {path}")
    clf = xgb.XGBClassifier()
    clf.load_model(str(path))
    return clf
