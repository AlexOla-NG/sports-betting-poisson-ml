import pandas as pd

from src.adjustments.lambda_adjustment import (
    adjust_fixture_lambdas,
    adjust_lambda,
    build_adjusted_lambdas,
)


CONFIG = {
    "injury": {
        "positional_weights": {"GK": 1.4, "DEF": 1.2, "MID": 1.2, "FWD": 1.05},
        "injury_decay": 0.7,
        "min_absence_matches": 3,
    }
}


def absence_rows(positions, dates=None, fixture_ids=None):
    dates = dates or pd.date_range("2025-01-01", periods=len(positions), freq="7D")
    fixture_ids = fixture_ids or [f"f{index}" for index in range(len(positions))]
    return pd.DataFrame(
        {
            "fixture_id": fixture_ids,
            "date": dates,
            "team": ["Newcastle United"] * len(positions),
            "player_id": [f"p{index}" for index in range(len(positions))],
            "position_group": positions,
            "absence_type": ["injury"] * len(positions),
            "starter_minute_share": [1.0] * len(positions),
        }
    )


def test_adjust_lambda_without_absences_is_unchanged():
    assert adjust_lambda(1.5, pd.DataFrame(), "Newcastle United", "2025-01-01", CONFIG) == 1.5


def test_adjust_lambda_reduces_scoring_lambda_for_single_forward_absence():
    absences = absence_rows(["FWD"])
    adjusted = adjust_lambda(2.0, absences, "Newcastle United", absences.loc[0, "date"], CONFIG)
    assert adjusted == 1.9


def test_multiple_simultaneous_absences_compound_attacking_reductions():
    absences = absence_rows(
        ["FWD", "MID"],
        dates=[pd.Timestamp("2025-01-01")] * 2,
    )
    adjusted = adjust_lambda(2.0, absences, "Newcastle United", absences.loc[0, "date"], CONFIG)
    assert adjusted == 1.52


def test_three_match_streak_applies_compounding_decay():
    dates = pd.date_range("2025-01-01", periods=3, freq="7D")
    absences = pd.DataFrame(
        {
            "fixture_id": ["f1", "f2", "f3"],
            "date": dates,
            "team": ["Newcastle United"] * 3,
            "player_id": ["p1"] * 3,
            "position_group": ["FWD"] * 3,
            "absence_type": ["injury"] * 3,
            "starter_minute_share": [1.0] * 3,
        }
    )
    adjusted = adjust_lambda(2.0, absences, "Newcastle United", dates[2], CONFIG)
    assert adjusted == 2.0 * (1 - 0.05 * (0.7**2))


def test_defensive_absence_increases_opponent_lambda():
    absences = absence_rows(["DEF"], fixture_ids=["fixture-1"])
    result = adjust_fixture_lambdas(
        "fixture-1",
        "Newcastle United",
        "Arsenal",
        absences.loc[0, "date"],
        1.5,
        1.2,
        absences,
        CONFIG,
    )
    assert result["adjusted_home_lambda"] == 1.5
    assert result["adjusted_away_lambda"] == 1.44
    assert result["num_home_absences_applied"] == 1


def test_build_adjusted_lambdas_reconciles_provider_fixture_ids():
    ratings = pd.DataFrame(
        {
            "fixture_id": ["fbref-hash"],
            "date": [pd.Timestamp("2025-01-01")],
            "home_team": ["Newcastle United"],
            "away_team": ["Arsenal"],
            "expected_home_goals": [1.5],
            "expected_away_goals": [1.2],
        }
    )
    absences = absence_rows(["FWD"], fixture_ids=[26700])
    output = build_adjusted_lambdas(ratings, absences, CONFIG)
    assert len(output) == 1
    assert output.loc[0, "num_home_absences_applied"] == 1
    assert output.attrs["unmatched_absences"].empty
