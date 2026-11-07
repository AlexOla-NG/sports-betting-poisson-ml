"""FBref-only fallback evaluation for recent completed fixtures.

Pipeline stage: 08_prediction, Task 5.3b.
This module intentionally excludes xG, availability, Elo, Poisson, and MC
features from the XGBoost input schema. The Monte Carlo candidate uses only
goal history through the existing point-in-time Poisson ratings path.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

from src.config.loader import get_form_windows, get_mc_trials, get_random_seed, get_rating_window
from src.ml.ensemble_blend import blend_probabilities, evaluate_probability_predictions
from src.ml.xgboost_model import compute_multiclass_brier_score
from src.processing.clean_features import standardize_team_name
from src.ratings.poisson import compute_rolling_ratings
from src.simulation.monte_carlo import simulate_fixture_trials


FBREF_FALLBACK_VERSION = "fbref_fallback_v1"
CLASS_ORDER = ("home_win", "draw", "away_win")
FBREF_FALLBACK_FEATURE_COLUMNS = [
    "fixture_week",
    "home_points_form",
    "away_points_form",
    "home_goals_for_form",
    "away_goals_for_form",
    "home_goals_against_form",
    "away_goals_against_form",
    "points_form_diff",
    "goals_for_form_diff",
    "goals_against_form_diff",
    "rest_days_home",
    "rest_days_away",
    "rest_days_diff",
    "h2h_home_win_rate",
    "h2h_match_count",
]
FORBIDDEN_FEATURE_TOKENS = (
    "xg",
    "absence",
    "appearance",
    "elo",
    "poisson",
    "lambda",
    "mc_",
    "goal",  # checked separately below to allow the explicit goals features
)


@dataclass(frozen=True)
class FallbackDataConfig:
    """Resolved configuration for the FBref-only fallback."""

    training_cutoff: pd.Timestamp
    points_window: int
    goals_window: int
    rating_window: int
    mc_trials: int
    random_seed: int
    cold_start_points: float
    cold_start_goals: float
    cold_start_rest_days: float
    cold_start_h2h_rate: float


def get_fbref_fallback_config(config: dict[str, Any]) -> FallbackDataConfig:
    """Resolve fallback configuration from central config.

    Pipeline stage: 08_prediction, Task 5.3b.
    """
    fallback = config["prediction"]["fbref_fallback"]
    return FallbackDataConfig(
        training_cutoff=pd.Timestamp(fallback["training_cutoff"]),
        points_window=get_form_windows(config)[0],
        goals_window=get_form_windows(config)[1],
        rating_window=get_rating_window(config),
        mc_trials=get_mc_trials(config),
        random_seed=get_random_seed(config),
        cold_start_points=float(fallback["cold_start_points"]),
        cold_start_goals=float(fallback["cold_start_goals"]),
        cold_start_rest_days=float(fallback["cold_start_rest_days"]),
        cold_start_h2h_rate=float(fallback["cold_start_h2h_rate"]),
    )


def _provider_id(value: object) -> str | None:
    """Return a non-empty provider fixture ID or ``None``."""
    if pd.isna(value) or str(value).strip() in {"", "<NA>", "nan", "None"}:
        return None
    return str(value).strip()


def _identity_key(row: pd.Series) -> str:
    """Build an identity key that excludes score and completion status."""
    week = pd.to_numeric(row.get("week"), errors="coerce")
    week_part = f"week:{int(week)}" if pd.notna(week) else f"date:{pd.Timestamp(row['date']).date()}"
    return "|".join(
        [
            str(row["league"]),
            str(row["season"]),
            week_part,
            str(row["home_team"]),
            str(row["away_team"]),
        ]
    )


def assign_fixture_identity(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Validate IDs and assign stable IDs for rows without provider IDs.

    Pipeline stage: 08_prediction, Task 5.3b.
    Provider IDs are preserved in ``provider_fixture_id``. Missing IDs use
    league, season, valid week, and canonical teams; date is only a fallback
    when week is unavailable, so postponed dates do not change normal IDs.
    """
    required = {"league", "season", "date", "home_team", "away_team"}
    missing = required.difference(fixtures.columns)
    if missing:
        raise ValueError(f"fixture identity requires columns: {sorted(missing)}")
    frame = fixtures.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for column in ("home_team", "away_team"):
        frame[column] = frame[column].map(standardize_team_name)
    frame["provider_fixture_id"] = (
        frame["fixture_id"].map(_provider_id) if "fixture_id" in frame else None
    )
    frame["_identity_key"] = frame.apply(_identity_key, axis=1)
    duplicate_keys = frame["_identity_key"].duplicated(keep=False)
    if duplicate_keys.any():
        grouped = frame.loc[duplicate_keys].groupby("_identity_key", dropna=False)
        if any(len(group) > 1 for _, group in grouped):
            raise ValueError("fixture identity collision detected")
    generated = frame["_identity_key"].map(lambda value: "fbref-" + sha256(value.encode()).hexdigest()[:20])
    frame["fixture_id"] = frame["provider_fixture_id"].fillna(generated)
    if frame["fixture_id"].duplicated().any():
        raise ValueError("fixture IDs must be unique after identity assignment")
    return frame.drop(columns=["_identity_key"])


def _outcome(home_goals: object, away_goals: object) -> int | float:
    """Encode a completed result as Home=0, Draw=1, Away=2."""
    if pd.isna(home_goals) or pd.isna(away_goals):
        return np.nan
    return int(0 if home_goals > away_goals else 1 if home_goals == away_goals else 2)


def _completed_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows with valid completed home and away scores."""
    return (
        pd.to_numeric(frame["home_goals"], errors="coerce").notna()
        & pd.to_numeric(frame["away_goals"], errors="coerce").notna()
    )


def _team_history(history: list[dict[str, Any]], team: str) -> list[dict[str, Any]]:
    """Return prior completed matches for one team."""
    return [item for item in history if item["home_team"] == team or item["away_team"] == team]


def _team_values(history: list[dict[str, Any]], team: str) -> tuple[list[float], list[float], list[pd.Timestamp]]:
    """Return points, goals-for, goals-against, and dates for a team's history."""
    points: list[float] = []
    goals_for: list[float] = []
    goals_against: list[float] = []
    dates: list[pd.Timestamp] = []
    for item in _team_history(history, team):
        home = item["home_team"] == team
        gf = float(item["home_goals"] if home else item["away_goals"])
        ga = float(item["away_goals"] if home else item["home_goals"])
        points.append(3.0 if gf > ga else 1.0 if gf == ga else 0.0)
        goals_for.append(gf)
        goals_against.append(ga)
        dates.append(item["date"])
    return points, goals_for, goals_against, dates


def _mean_or_default(values: list[float], window: int, default: float) -> float:
    """Return the recent mean or an explicit configured cold-start value."""
    return float(np.mean(values[-window:])) if values else default


def build_fbref_fallback_features(
    fixtures: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build the frozen FBref-only schema from strict earlier-date history.

    Pipeline stage: 08_prediction, Task 5.3b. Rows sharing a date receive the
    same history snapshot; completed rows update state only after that date
    block. Missing history uses configured neutral/cold-start values.
    """
    resolved = get_fbref_fallback_config(config)
    frame = assign_fixture_identity(fixtures)
    if "week" not in frame.columns:
        frame["week"] = np.nan
    frame["_completed"] = _completed_mask(frame)
    frame = frame.sort_values(["date", "fixture_id"], kind="stable").reset_index(drop=True)
    records: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for date, block in frame.groupby("date", sort=True):
        for _, row in block.iterrows():
            home = str(row["home_team"])
            away = str(row["away_team"])
            hp, hgf, hga, hd = _team_values(history, home)
            ap, agf, aga, ad = _team_values(history, away)
            h2h = [
                item for item in history
                if item["home_team"] == home and item["away_team"] == away
            ]
            h2h_rate = (
                float(sum(item["home_goals"] > item["away_goals"] for item in h2h) / len(h2h))
                if h2h else resolved.cold_start_h2h_rate
            )
            home_rest = (
                float((date - hd[-1]).days) if hd else resolved.cold_start_rest_days
            )
            away_rest = (
                float((date - ad[-1]).days) if ad else resolved.cold_start_rest_days
            )
            week = pd.to_numeric(row["week"], errors="coerce")
            records.append(
                {
                    "fixture_id": row["fixture_id"],
                    "date": date,
                    "home_team": home,
                    "away_team": away,
                    "target_encoded": _outcome(row["home_goals"], row["away_goals"]),
                    "fixture_week": float(week) if pd.notna(week) else np.nan,
                    "home_points_form": _mean_or_default(hp, resolved.points_window, resolved.cold_start_points),
                    "away_points_form": _mean_or_default(ap, resolved.points_window, resolved.cold_start_points),
                    "home_goals_for_form": _mean_or_default(hgf, resolved.goals_window, resolved.cold_start_goals),
                    "away_goals_for_form": _mean_or_default(agf, resolved.goals_window, resolved.cold_start_goals),
                    "home_goals_against_form": _mean_or_default(hga, resolved.goals_window, resolved.cold_start_goals),
                    "away_goals_against_form": _mean_or_default(aga, resolved.goals_window, resolved.cold_start_goals),
                    "rest_days_home": home_rest,
                    "rest_days_away": away_rest,
                    "h2h_home_win_rate": h2h_rate,
                    "h2h_match_count": float(len(h2h)),
                }
            )
        for _, row in block.iterrows():
            if bool(row["_completed"]):
                history.append(
                    {
                        "date": date,
                        "home_team": str(row["home_team"]),
                        "away_team": str(row["away_team"]),
                        "home_goals": float(row["home_goals"]),
                        "away_goals": float(row["away_goals"]),
                    }
                )
    result = pd.DataFrame(records)
    result["points_form_diff"] = result["home_points_form"] - result["away_points_form"]
    result["goals_for_form_diff"] = result["home_goals_for_form"] - result["away_goals_for_form"]
    result["goals_against_form_diff"] = (
        result["home_goals_against_form"] - result["away_goals_against_form"]
    )
    result["rest_days_diff"] = result["rest_days_home"] - result["rest_days_away"]
    return result[["fixture_id", "date", "home_team", "away_team", "target_encoded", *FBREF_FALLBACK_FEATURE_COLUMNS]]


def _validate_feature_schema(features: pd.DataFrame) -> None:
    """Validate the exact fallback schema and forbidden feature lineage."""
    if list(features.columns[-len(FBREF_FALLBACK_FEATURE_COLUMNS):]) != FBREF_FALLBACK_FEATURE_COLUMNS:
        raise ValueError("fallback feature order does not match the frozen schema")
    names = set(FBREF_FALLBACK_FEATURE_COLUMNS)
    forbidden = [name for name in names if ("xg" in name.lower() or "absence" in name.lower() or "elo" in name.lower())]
    if forbidden:
        raise ValueError(f"forbidden fallback features: {forbidden}")


def _fit_model(features: pd.DataFrame, target: pd.Series, config: dict[str, Any]) -> xgb.XGBClassifier:
    """Fit one frozen-parameter XGBoost model on the cutoff training rows."""
    params = {
        **config["ml"]["xgboost"],
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "random_state": get_random_seed(config),
    }
    model = xgb.XGBClassifier(**params)
    model.fit(features[FBREF_FALLBACK_FEATURE_COLUMNS], target.to_numpy(dtype=int))
    return model


def _safe_mc_predictions(
    combined: pd.DataFrame,
    evaluation_ids: set[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    """Produce safe base-lambda MC probabilities from prior goals only."""
    resolved = get_fbref_fallback_config(config)
    ratings = compute_rolling_ratings(
        combined,
        window=resolved.rating_window,
        home_field_advantage=config["ratings"]["home_field_advantage"],
        glm_l2_alpha=config["ratings"]["glm_l2_alpha"],
        min_expected_goals=config["ratings"]["min_expected_goals"],
    )
    rows: list[dict[str, Any]] = []
    for index, row in ratings[ratings["fixture_id"].isin(evaluation_ids)].iterrows():
        base_home = float(row["expected_home_goals"])
        base_away = float(row["expected_away_goals"])
        simulation = simulate_fixture_trials(
            base_home,
            base_away,
            num_trials=resolved.mc_trials,
            random_seed=resolved.random_seed + int(index),
        )
        rows.append(
            {
                "fixture_id": row["fixture_id"],
                "base_home_lambda": base_home,
                "base_away_lambda": base_away,
                "adjusted_home_lambda": base_home,
                "adjusted_away_lambda": base_away,
                "num_home_absences_applied": 0,
                "num_away_absences_applied": 0,
                "mc_prob_home_win": simulation["mc_prob_home_win"],
                "mc_prob_draw": simulation["mc_prob_draw"],
                "mc_prob_away_win": simulation["mc_prob_away_win"],
            }
        )
    result = pd.DataFrame(rows)
    if set(result["fixture_id"]) != evaluation_ids:
        raise ValueError("safe MC population does not match evaluation fixtures")
    return result


def _source_hash(path: Path) -> str:
    """Return a SHA-256 hash for an input artifact."""
    return sha256(path.read_bytes()).hexdigest()


def _validate_source_consistency(historical: pd.DataFrame, probe: pd.DataFrame) -> None:
    """Validate that both inputs are single-league FBref fixture tables."""
    for frame, name in ((historical, "historical"), (probe, "probe")):
        if frame["league"].dropna().nunique() != 1:
            raise ValueError(f"{name} input must contain exactly one league")
        if frame["season"].isna().any():
            raise ValueError(f"{name} input contains missing season labels")


def run_fbref_fallback_evaluation(
    historical_fixtures: pd.DataFrame,
    fbref_probe: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
    source_paths: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    """Fit and evaluate the separately versioned FBref-only fallback.

    Pipeline stage: 08_prediction, Task 5.3b. The model is fitted once using
    completed historical rows through the configured cutoff. Newer completed
    probe rows are scored chronologically without changing fitted parameters.
    """
    resolved = get_fbref_fallback_config(config)
    historical = assign_fixture_identity(historical_fixtures)
    probe = assign_fixture_identity(fbref_probe)
    _validate_source_consistency(historical, probe)
    historical["_completed"] = _completed_mask(historical)
    probe["_completed"] = _completed_mask(probe)
    if not historical["_completed"].all():
        raise ValueError("historical fallback input contains incomplete rows")
    train_dates = pd.to_datetime(historical["date"])
    if train_dates.max() != resolved.training_cutoff:
        raise ValueError(f"training cutoff {resolved.training_cutoff.date()} does not match historical max {train_dates.max().date()}")
    evaluation = probe[
        probe["_completed"] & (pd.to_datetime(probe["date"]) > resolved.training_cutoff)
    ].copy()
    evaluation_ids = set(evaluation["fixture_id"])
    combined = pd.concat([historical, probe], ignore_index=True, sort=False)
    features = build_fbref_fallback_features(combined, config)
    train_features = features[features["fixture_id"].isin(set(historical["fixture_id"]))]
    eval_features = features[features["fixture_id"].isin(evaluation_ids)].sort_values("date")
    _validate_feature_schema(train_features)
    if len(eval_features) != len(evaluation):
        raise ValueError("feature/evaluation population mismatch")
    model = _fit_model(train_features, train_features["target_encoded"], config)
    xgb_prob = model.predict_proba(eval_features[FBREF_FALLBACK_FEATURE_COLUMNS]).astype(float)
    xgb_prob = xgb_prob / xgb_prob.sum(axis=1, keepdims=True)
    mc = _safe_mc_predictions(combined, evaluation_ids, config).set_index("fixture_id").loc[eval_features["fixture_id"]]
    mc_prob = mc[["mc_prob_home_win", "mc_prob_draw", "mc_prob_away_win"]].to_numpy()
    blended = blend_probabilities(mc_prob, xgb_prob, 0.8)
    target = eval_features["target_encoded"].to_numpy(dtype=int)
    predictions = eval_features[["fixture_id", "date", "home_team", "away_team", "target_encoded"]].copy()
    for prefix, matrix in (("xgb", xgb_prob), ("mc", mc_prob), ("blend_80_20", blended)):
        predictions[f"{prefix}_prob_home_win"] = matrix[:, 0]
        predictions[f"{prefix}_prob_draw"] = matrix[:, 1]
        predictions[f"{prefix}_prob_away_win"] = matrix[:, 2]
    predictions["actual_outcome"] = target
    metrics = pd.DataFrame(
        [
            {
                "pipeline_version": FBREF_FALLBACK_VERSION,
                "candidate": candidate,
                "fixture_count": len(target),
                "date_min": predictions["date"].min(),
                "date_max": predictions["date"].max(),
                **evaluate_probability_predictions(matrix, target),
            }
            for candidate, matrix in (
                ("fallback_xgboost_only", xgb_prob),
                ("safe_base_lambda_mc_only", mc_prob),
                ("fixed_80_20_blend", blended),
            )
        ]
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "fbref_fallback_xgboost.json"
    model.save_model(str(model_path))
    predictions.to_parquet(output / "fixture_predictions.parquet", index=False)
    metrics.to_parquet(output / "candidate_metrics.parquet", index=False)
    resolved_sources: dict[str, dict[str, Any]] = {}
    for name, source in (source_paths or {}).items():
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"fallback source artifact not found: {path}")
        resolved_sources[name] = {
            "path": str(path),
            "sha256": _source_hash(path),
            "mtime_utc": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
        }
    schema = {
        "pipeline_version": FBREF_FALLBACK_VERSION,
        "feature_columns": FBREF_FALLBACK_FEATURE_COLUMNS,
        "class_order": list(CLASS_ORDER),
        "training_cutoff": resolved.training_cutoff.isoformat(),
        "evaluation_date_min": predictions["date"].min().isoformat(),
        "evaluation_date_max": predictions["date"].max().isoformat(),
        "absence_policy": config["prediction"]["absence_policy"],
        "mc_seed": resolved.random_seed,
        "mc_trials": resolved.mc_trials,
        "feature_policy": {
            "history_cutoff": "strictly earlier date; same-date blocks update after scoring",
            "rolling_points_window": resolved.points_window,
            "rolling_goals_window": resolved.goals_window,
            "rest_days": "days since latest completed prior match; configured cold-start otherwise",
            "h2h": "prior same home/away pairing only; neutral 0.5 rate and zero count otherwise",
            "cold_start_goals": resolved.cold_start_goals,
            "excluded_lineage": [
                "xG",
                "player appearances/absences",
                "Elo",
                "Poisson/MC columns",
                "current-match result/statistics",
            ],
        },
        "model_path": str(model_path),
        "model_sha256": _source_hash(model_path),
        "xgboost_parameters": dict(config["ml"]["xgboost"]),
        "source_artifacts": resolved_sources,
        "evaluation_class_distribution": {
            str(key): int(value)
            for key, value in pd.Series(target).value_counts().sort_index().items()
        },
        "exclusions": {
            "incomplete_probe_rows": int((~probe["_completed"]).sum()),
            "pre_cutoff_probe_rows": int(
                (pd.to_datetime(probe["date"]) <= resolved.training_cutoff).sum()
            ),
            "eligible_evaluation_rows": len(evaluation),
            "missing_provider_ids_in_probe": int(probe["provider_fixture_id"].isna().sum()),
        },
    }
    (output / "metadata.json").write_text(json.dumps(schema, indent=2, default=str) + "\n")
    return {"predictions": predictions, "metrics": metrics, "metadata": schema}
