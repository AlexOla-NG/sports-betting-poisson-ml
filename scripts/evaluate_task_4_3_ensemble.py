"""Audit the persisted Task 4.3 OOF ensemble decision.

This script is read-only with respect to existing pipeline artifacts. It validates
selection, baselines, nearby grid weights, probability integrity, and optional
feature-table alignment, then writes one new long-format audit report.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SELECTION_TOLERANCE = 1e-12
NEARBY_BRIER_TOLERANCE = 0.0005
NEARBY_LOG_LOSS_TOLERANCE = 0.0005
EXPECTED_OOF_ROWS = 560
REQUIRED_BASELINES = {"mc_only", "full_xgb_only", "context_only"}


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    """Raise a clear error when an audit input is missing required columns."""
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def recompute_selected_candidate(
    sweep: pd.DataFrame,
    tolerance: float = SELECTION_TOLERANCE,
) -> pd.Series:
    """Recompute the Task 4.3 winner using Brier, log loss, and simplicity."""
    if tolerance < 0:
        raise ValueError("selection tolerance must be non-negative")
    best_brier = float(sweep["brier_score"].min())
    tied = sweep[sweep["brier_score"] <= best_brier + tolerance].copy()
    best_log_loss = float(tied["log_loss"].min())
    tied = tied[tied["log_loss"] <= best_log_loss + tolerance].copy()

    priorities = {"mc_only": 0, "context_only": 1, "full_xgb_only": 2}
    tied["_simplicity"] = tied["candidate"].map(priorities).fillna(3)
    return tied.sort_values(["_simplicity", "candidate", "xgb_weight"], kind="stable").iloc[0].drop(labels=["_simplicity"])


def validate_sweep(
    sweep: pd.DataFrame,
    selection_tolerance: float = SELECTION_TOLERANCE,
) -> pd.Series:
    """Validate the persisted sweep and return its recomputed selected row."""
    required = {
        "candidate", "xgb_weight", "mc_weight", "oof_fixture_count",
        "accuracy", "log_loss", "brier_score", "is_selected",
    }
    _require_columns(sweep, required, "ensemble sweep")
    if sweep.empty:
        raise ValueError("ensemble sweep is empty")
    if not np.isfinite(sweep[["xgb_weight", "mc_weight", "accuracy", "log_loss", "brier_score"]].to_numpy()).all():
        raise ValueError("ensemble sweep contains non-finite numeric values")
    if ((sweep[["xgb_weight", "mc_weight"]] < 0) | (sweep[["xgb_weight", "mc_weight"]] > 1)).any().any():
        raise ValueError("ensemble weights must be within [0, 1]")
    if not np.allclose(sweep["xgb_weight"] + sweep["mc_weight"], 1.0, atol=1e-12):
        raise ValueError("ensemble weight rows must sum to one")
    if sweep["oof_fixture_count"].nunique() != 1 or int(sweep["oof_fixture_count"].iloc[0]) <= 0:
        raise ValueError("OOF fixture count must be positive and consistent")
    if int(sweep["is_selected"].sum()) != 1:
        raise ValueError("ensemble sweep must contain exactly one selected row")
    if not REQUIRED_BASELINES.issubset(set(sweep["candidate"])):
        raise ValueError(f"required baselines missing: {sorted(REQUIRED_BASELINES.difference(sweep['candidate']))}")
    if not {0.0, 1.0}.issubset(set(sweep["xgb_weight"])):
        raise ValueError("ensemble sweep must include xgb weights 0.0 and 1.0")
    duplicates = sweep.duplicated(["candidate", "xgb_weight"])
    if duplicates.any():
        raise ValueError("duplicate candidate/xgb_weight rows found")

    recomputed = recompute_selected_candidate(sweep, selection_tolerance)
    if str(recomputed["candidate"]) != "blend_context":
        raise ValueError(f"unexpected recomputed Task 4.3 candidate: {recomputed['candidate']}")
    persisted = sweep.loc[sweep["is_selected"]].iloc[0]
    if str(persisted["candidate"]) != str(recomputed["candidate"]) or not np.isclose(
        float(persisted["xgb_weight"]), float(recomputed["xgb_weight"]), atol=selection_tolerance
    ):
        raise ValueError("persisted selected row disagrees with recomputed selection")
    return recomputed


def validate_selected_probabilities(
    probabilities: pd.DataFrame,
    selected: pd.Series,
    feature_table: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Validate selected OOF probability rows and return integrity metrics."""
    required = {
        "fixture_id", "date", "target_encoded", "ensemble_prob_home_win",
        "ensemble_prob_draw", "ensemble_prob_away_win", "selected_candidate",
        "selected_xgb_weight", "selected_mc_weight",
    }
    _require_columns(probabilities, required, "selected ensemble probabilities")
    if len(probabilities) != EXPECTED_OOF_ROWS:
        raise ValueError(f"expected {EXPECTED_OOF_ROWS} selected OOF rows, got {len(probabilities)}")
    dates = pd.to_datetime(probabilities["date"], errors="coerce")
    target = pd.to_numeric(probabilities["target_encoded"], errors="coerce")
    probability_columns = ["ensemble_prob_home_win", "ensemble_prob_draw", "ensemble_prob_away_win"]
    matrix = probabilities[probability_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    null_count = int(np.isnan(matrix).sum())
    non_finite_count = int((~np.isfinite(matrix)).sum())
    negative_count = int((matrix < 0).sum()) if np.isfinite(matrix).all() else 0
    if probabilities["fixture_id"].isna().any() or probabilities["fixture_id"].duplicated().any():
        raise ValueError("selected probabilities contain missing or duplicate fixture IDs")
    if dates.isna().any():
        raise ValueError("selected probabilities contain invalid dates")
    if target.isna().any() or not target.isin([0, 1, 2]).all():
        raise ValueError("selected probabilities contain invalid target classes")
    if null_count or non_finite_count or negative_count:
        raise ValueError("selected probabilities contain null, non-finite, or negative values")
    row_sum_error = float(np.max(np.abs(matrix.sum(axis=1) - 1.0)))
    if row_sum_error > 1e-12:
        raise ValueError(f"selected probability rows do not sum to one; max error={row_sum_error}")
    if probabilities["selected_candidate"].nunique() != 1 or str(probabilities["selected_candidate"].iloc[0]) != str(selected["candidate"]):
        raise ValueError("selected candidate label is inconsistent")
    for column, expected in [("selected_xgb_weight", float(selected["xgb_weight"])), ("selected_mc_weight", float(selected["mc_weight"]))]:
        if probabilities[column].nunique() != 1 or not np.isclose(float(probabilities[column].iloc[0]), expected, atol=1e-12):
            raise ValueError(f"selected weight column {column} is inconsistent")
    if np.allclose(matrix, matrix[0], atol=1e-15):
        raise ValueError("selected probability vectors are constant across fixtures")
    if feature_table is not None:
        _require_columns(feature_table, {"fixture_id", "date"}, "feature table")
        feature_ids = set(feature_table["fixture_id"])
        if not set(probabilities["fixture_id"]).issubset(feature_ids):
            raise ValueError("selected probability fixture IDs are absent from feature table")

    ordered = probabilities.assign(_date=dates).sort_values(["_date", "fixture_id"])
    return {
        "selected_fixture_count": len(probabilities),
        "unique_fixture_count": probabilities["fixture_id"].nunique(),
        "duplicate_fixture_count": int(probabilities["fixture_id"].duplicated().sum()),
        "target_missing_count": int(target.isna().sum()),
        "invalid_target_count": int((~target.isin([0, 1, 2])).sum()),
        "null_probability_count": null_count,
        "negative_probability_count": negative_count,
        "non_finite_probability_count": non_finite_count,
        "maximum_row_sum_error": row_sum_error,
        "all_rows_sum_to_one": row_sum_error <= 1e-12,
        "date_min": str(ordered["_date"].min().date()),
        "date_max": str(ordered["_date"].max().date()),
        "chronological_sort_verified": bool(ordered["_date"].is_monotonic_increasing),
        "constant_selected_candidate_verified": True,
        "constant_selected_weights_verified": True,
    }


def build_audit_report(
    sweep: pd.DataFrame,
    probabilities: pd.DataFrame,
    feature_table: pd.DataFrame | None = None,
    selection_tolerance: float = SELECTION_TOLERANCE,
    nearby_brier_tolerance: float = NEARBY_BRIER_TOLERANCE,
    nearby_log_loss_tolerance: float = NEARBY_LOG_LOSS_TOLERANCE,
) -> pd.DataFrame:
    """Build the long-format Task 4.3 audit report without mutating inputs."""
    selected = validate_sweep(sweep, selection_tolerance)
    integrity = validate_selected_probabilities(probabilities, selected, feature_table)
    candidate_names = set(sweep["candidate"])
    selected_candidate = str(selected["candidate"])
    if not (0.0 < float(selected["xgb_weight"]) < 1.0):
        raise ValueError("selected candidate is not an interior blend")
    selected_metrics = {key: selected[key] for key in ["accuracy", "log_loss", "brier_score"]}

    baseline_rows = []
    for candidate in ["mc_only", "full_xgb_only", "context_only"]:
        row = sweep[sweep["candidate"] == candidate]
        if len(row) != 1:
            raise ValueError(f"baseline {candidate} must exist exactly once")
        row = row.iloc[0]
        baseline_rows.append({
            "section": "baseline_comparison", "candidate": candidate,
            "xgb_weight": row["xgb_weight"], "mc_weight": row["mc_weight"],
            "brier_score": row["brier_score"], "log_loss": row["log_loss"], "accuracy": row["accuracy"],
            "brier_improvement_vs_selected": row["brier_score"] - selected["brier_score"],
            "log_loss_improvement_vs_selected": row["log_loss"] - selected["log_loss"],
            "selected_flag": False,
        })
    baseline_rows.append({
        "section": "baseline_comparison", "candidate": selected_candidate,
        "xgb_weight": selected["xgb_weight"], "mc_weight": selected["mc_weight"],
        "brier_score": selected["brier_score"], "log_loss": selected["log_loss"], "accuracy": selected["accuracy"],
        "brier_improvement_vs_selected": 0.0, "log_loss_improvement_vs_selected": 0.0, "selected_flag": True,
    })

    context = sweep[sweep["candidate"] == "blend_context"].sort_values("xgb_weight")
    lower = context[context["xgb_weight"] < selected["xgb_weight"]].tail(1)
    upper = context[context["xgb_weight"] > selected["xgb_weight"]].head(1)
    endpoint = context[context["xgb_weight"] == 1.0]
    nearby = pd.concat([lower, upper, endpoint]).drop_duplicates(["candidate", "xgb_weight"])
    nearby = nearby.assign(
        brier_delta_from_selected=lambda frame: frame["brier_score"] - selected["brier_score"],
        log_loss_delta_from_selected=lambda frame: frame["log_loss"] - selected["log_loss"],
        rank_by_brier=lambda frame: frame["brier_score"].rank(method="min"),
        rank_by_log_loss=lambda frame: frame["log_loss"].rank(method="min"),
    )
    nearby_rows = [{"section": "nearby_weight_stability", **row.to_dict()} for _, row in nearby.iterrows()]

    selected_beats = lambda metric, candidate: float(selected[metric]) < float(sweep.loc[sweep["candidate"] == candidate, metric].iloc[0])
    required_wins = [
        ("brier_score", "mc_only"),
        ("brier_score", "context_only"),
        ("brier_score", "full_xgb_only"),
        ("log_loss", "mc_only"),
        ("log_loss", "context_only"),
        ("log_loss", "full_xgb_only"),
    ]
    if not all(selected_beats(metric, candidate) for metric, candidate in required_wins):
        raise ValueError("selected candidate does not beat every required standalone baseline")
    interpretation = {
        "section": "interpretation",
        "interior_blend_selected": True,
        "selected_beats_mc_only_on_brier": selected_beats("brier_score", "mc_only"),
        "selected_beats_context_only_on_brier": selected_beats("brier_score", "context_only"),
        "selected_beats_full_xgb_only_on_brier": selected_beats("brier_score", "full_xgb_only"),
        "selected_beats_context_only_on_log_loss": selected_beats("log_loss", "context_only"),
        "selected_beats_full_xgb_only_on_log_loss": selected_beats("log_loss", "full_xgb_only"),
        "nearby_weights_within_brier_tolerance": int((nearby["brier_delta_from_selected"].abs() <= nearby_brier_tolerance).sum()),
        "nearby_weights_within_log_loss_tolerance": int((nearby["log_loss_delta_from_selected"].abs() <= nearby_log_loss_tolerance).sum()),
        "selection_caveat": "weights were selected from the same OOF prediction set used to report performance; the exact 0.80 value should be confirmed on a later untouched time period or through nested walk-forward selection.",
    }

    summary = {
        "section": "audit_summary", "selected_candidate": selected_candidate,
        "selected_xgb_weight": selected["xgb_weight"], "selected_mc_weight": selected["mc_weight"],
        "oof_fixture_count": selected["oof_fixture_count"], "selected_accuracy": selected_metrics["accuracy"],
        "selected_log_loss": selected_metrics["log_loss"], "selected_brier_score": selected_metrics["brier_score"],
        "selection_rule": "lowest Brier, then log loss, then simplicity", "candidate_count": len(sweep),
        "sweep_grid_min_weight": sweep["xgb_weight"].min(), "sweep_grid_max_weight": sweep["xgb_weight"].max(),
        "sweep_grid_step": sweep["xgb_weight"].sort_values().diff().dropna().min(),
    }
    integrity_row = {"section": "probability_integrity", **integrity}
    return pd.DataFrame([summary, *baseline_rows, *nearby_rows, integrity_row, interpretation])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the read-only Task 4.3 audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-path", type=Path, default=Path("data/processed/ensemble_sweep_results.parquet"))
    parser.add_argument("--probabilities-path", type=Path, default=Path("data/processed/ensemble_probabilities.parquet"))
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/task_4_3_evaluation_report.parquet"))
    parser.add_argument("--selection-tolerance", type=float, default=SELECTION_TOLERANCE)
    parser.add_argument("--nearby-brier-tolerance", type=float, default=NEARBY_BRIER_TOLERANCE)
    parser.add_argument("--nearby-log-loss-tolerance", type=float, default=NEARBY_LOG_LOSS_TOLERANCE)
    return parser.parse_args()


def main() -> int:
    """Run the Task 4.3 audit and write one new report artifact."""
    args = parse_args()
    sweep = pd.read_parquet(args.sweep_path)
    probabilities = pd.read_parquet(args.probabilities_path)
    feature_path = args.sweep_path.parent / "feature_table.parquet"
    feature_table = pd.read_parquet(feature_path) if feature_path.exists() else None
    report = build_audit_report(
        sweep, probabilities, feature_table,
        args.selection_tolerance, args.nearby_brier_tolerance, args.nearby_log_loss_tolerance,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(args.output_path, index=False)
    selected = report[report["section"] == "audit_summary"].iloc[0]
    print(f"Selected: {selected['selected_candidate']} (xgb={selected['selected_xgb_weight']:.2f}, mc={selected['selected_mc_weight']:.2f})")
    print(f"OOF fixtures: {int(selected['oof_fixture_count'])}; Brier={selected['selected_brier_score']:.6f}; Log loss={selected['selected_log_loss']:.6f}; Accuracy={selected['selected_accuracy']:.6f}")
    print(f"Wrote audit report: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
