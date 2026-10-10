import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_task_4_3_ensemble import (
    build_audit_report,
    recompute_selected_candidate,
    validate_selected_probabilities,
    validate_sweep,
)


def make_sweep(selected=True):
    rows = [
        ["mc_only", 0.0, 1.0, 0.66, 1.10, 0.47, False],
        ["full_xgb_only", 1.0, 0.0, 0.62, 1.03, 0.52, False],
        ["context_only", 1.0, 0.0, 0.616, 1.028, 0.50, False],
        ["blend_context", 0.75, 0.25, 0.614, 1.025, 0.51, False],
        ["blend_context", 0.80, 0.20, 0.612, 1.023, 0.51, selected],
        ["blend_context", 0.85, 0.15, 0.613, 1.024, 0.50, False],
        ["blend_context", 1.0, 0.0, 0.616, 1.028, 0.50, False],
        ["blend_full", 0.0, 1.0, 0.66, 1.10, 0.47, False],
        ["blend_full", 1.0, 0.0, 0.62, 1.03, 0.52, False],
    ]
    return pd.DataFrame(rows, columns=["candidate", "xgb_weight", "mc_weight", "brier_score", "log_loss", "accuracy", "is_selected"]).assign(oof_fixture_count=560)


def make_probabilities(n=560):
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    probabilities = np.tile([0.4, 0.3, 0.3], (n, 1)).astype(float)
    probabilities[:, 0] += np.linspace(0, 0.1, n)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return pd.DataFrame({
        "fixture_id": [f"f{i}" for i in range(n)],
        "date": dates,
        "target_encoded": np.arange(n) % 3,
        "ensemble_prob_home_win": probabilities[:, 0],
        "ensemble_prob_draw": probabilities[:, 1],
        "ensemble_prob_away_win": probabilities[:, 2],
        "selected_candidate": ["blend_context"] * n,
        "selected_xgb_weight": [0.8] * n,
        "selected_mc_weight": [0.2] * n,
    })


def test_valid_audit_report_and_nearby_weights():
    report = build_audit_report(make_sweep(), make_probabilities())
    assert set(report["section"]) == {"audit_summary", "baseline_comparison", "nearby_weight_stability", "probability_integrity", "interpretation"}
    assert len(report[report["section"] == "baseline_comparison"]) == 4


def test_selection_tie_break_and_simplicity():
    sweep = make_sweep()
    sweep.loc[sweep["candidate"] == "blend_context", "brier_score"] = 0.612
    sweep.loc[sweep["candidate"] == "blend_context", "log_loss"] = 1.023
    assert recompute_selected_candidate(sweep)["candidate"] == "blend_context"

    tied = pd.DataFrame([
        {"candidate": "blend_context", "xgb_weight": 0.8, "brier_score": 0.5, "log_loss": 1.0},
        {"candidate": "context_only", "xgb_weight": 1.0, "brier_score": 0.5, "log_loss": 1.0},
    ])
    assert recompute_selected_candidate(tied)["candidate"] == "context_only"


def test_invalid_sweep_and_persisted_selection_raise():
    sweep = make_sweep()
    sweep.loc[0, "xgb_weight"] = 1.2
    with pytest.raises(ValueError):
        validate_sweep(sweep)
    sweep = make_sweep(selected=False)
    with pytest.raises(ValueError, match="exactly one"):
        validate_sweep(sweep)


def test_invalid_probability_rows_raise():
    probabilities = make_probabilities()
    probabilities.loc[0, "ensemble_prob_home_win"] = np.nan
    with pytest.raises(ValueError):
        validate_selected_probabilities(probabilities, make_sweep().loc[4])

    probabilities = make_probabilities()
    probabilities.loc[0, "ensemble_prob_home_win"] = -0.1
    with pytest.raises(ValueError):
        validate_selected_probabilities(probabilities, make_sweep().loc[4])

    probabilities = make_probabilities()
    probabilities.loc[0, "fixture_id"] = probabilities.loc[1, "fixture_id"]
    with pytest.raises(ValueError):
        validate_selected_probabilities(probabilities, make_sweep().loc[4])


def test_report_artifact_can_be_written(tmp_path):
    report = build_audit_report(make_sweep(), make_probabilities())
    output = tmp_path / "task_4_3_evaluation_report.parquet"
    report.to_parquet(output, index=False)
    loaded = pd.read_parquet(output)
    assert len(loaded) > 0