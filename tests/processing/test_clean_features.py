import numpy as np
import pandas as pd
import pytest

from src.processing.clean_features import (
    TEAM_NAME_MAPPING,
    add_base_differentials,
    clean_fixtures_pipeline,
    standardize_team_columns,
    standardize_team_name,
)


def test_standardize_team_name_maps_known_aliases():
    assert standardize_team_name("Man United") == "Manchester United"
    assert standardize_team_name("Manchester Utd") == "Manchester United"
    assert standardize_team_name("Wolves") == "Wolverhampton Wanderers"
    assert standardize_team_name("Arsenal") == "Arsenal"
    assert standardize_team_name(None) is None


def test_standardize_team_columns_updates_specified_columns():
    df = pd.DataFrame(
        {
            "home_team": ["Man United", "Wolves"],
            "away_team": ["Leicester", "Brighton"],
            "other_col": ["Unchanged", "Unchanged"],
        }
    )
    result = standardize_team_columns(df, ["home_team", "away_team"])

    assert result["home_team"].tolist() == ["Manchester United", "Wolverhampton Wanderers"]
    assert result["away_team"].tolist() == ["Leicester City", "Brighton & Hove Albion"]
    assert result["other_col"].tolist() == ["Unchanged", "Unchanged"]


def test_add_base_differentials_computes_home_minus_away():
    df = pd.DataFrame(
        {
            "home_goals": [2, 1],
            "away_goals": [1, 1],
            "home_xg": [1.8, 0.5],
            "away_xg": [0.9, 1.2],
            "home_shots": [12, 8],
            "away_shots": [6, 10],
            "home_sot": [5, 2],
            "away_sot": [3, 4],
        }
    )
    result = add_base_differentials(df)

    assert result["goals_diff"].tolist() == [1, 0]
    assert np.isclose(result["xg_diff"].tolist(), [0.9, -0.7]).all()
    assert result["shots_diff"].tolist() == [6, -2]
    assert result["sot_diff"].tolist() == [2, -2]


def test_clean_fixtures_pipeline_handles_missing_xg_and_sorts_dates():
    fixtures = pd.DataFrame(
        {
            "fixture_id": ["fix2", "fix1"],
            "date": ["2024-08-20", "2024-08-16"],
            "home_team": ["Man United", "Arsenal"],
            "away_team": ["Fulham", "Wolves"],
            "home_goals": [1, 2],
            "away_goals": [0, 0],
            "home_xg": [1.4, np.nan],
            "away_xg": [0.4, np.nan],
        }
    )

    cleaned = clean_fixtures_pipeline(fixtures)

    # Should be sorted chronologically by date
    assert cleaned["fixture_id"].tolist() == ["fix1", "fix2"]
    assert cleaned.loc[0, "home_team"] == "Arsenal"
    assert cleaned.loc[0, "away_team"] == "Wolverhampton Wanderers"
    assert cleaned.loc[0, "is_xg_missing"] == True
    assert cleaned.loc[1, "is_xg_missing"] == False
    assert cleaned.loc[1, "goals_diff"] == 1
    assert np.isclose(cleaned.loc[1, "xg_diff"], 1.0)
