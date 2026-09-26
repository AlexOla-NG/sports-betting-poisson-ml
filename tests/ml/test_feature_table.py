"""
Unit tests for src/ml/feature_table.py
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.feature_table import (
    build_rolling_team_features,
    compute_head_to_head_features,
    construct_feature_table,
)


def test_build_rolling_team_features_leakage():
    """Verify that match 1 outcome never leaks into match 1's own rolling features (.shift(1) check)."""
    fixtures = pd.DataFrame(
        {
            "fixture_id": ["fix1", "fix2"],
            "date": ["2024-08-16", "2024-08-23"],
            "home_team": ["Arsenal", "Arsenal"],
            "away_team": ["Chelsea", "Everton"],
            "home_goals": [3, 1],
            "away_goals": [0, 1],
            "home_xg": [2.5, 1.2],
            "away_xg": [0.3, 0.9],
        }
    )

    rolling = build_rolling_team_features(
        fixtures, points_window=5, goals_window=5, xg_for_window=8, xg_against_window=8
    )

    # For Arsenal's 1st match (fix1), pre-match rolling form must be NaN
    fix1_row = rolling[rolling["fixture_id"] == "fix1"].iloc[0]
    assert pd.isna(fix1_row["home_points_form"])
    assert pd.isna(fix1_row["home_goals_form"])

    # For Arsenal's 2nd match (fix2), pre-match points form should reflect fix1 ONLY (3.0 pts)
    fix2_row = rolling[rolling["fixture_id"] == "fix2"].iloc[0]
    assert fix2_row["home_points_form"] == 3.0
    assert fix2_row["home_goals_form"] == 3.0
    assert fix2_row["home_goals_against_form"] == 0.0


def test_compute_head_to_head_features():
    """Verify head-to-head calculation considers ONLY matches strictly before current date."""
    fixtures = pd.DataFrame(
        {
            "fixture_id": ["fix1", "fix2"],
            "date": ["2024-08-16", "2024-08-23"],
            "home_team": ["Arsenal", "Chelsea"],
            "away_team": ["Chelsea", "Arsenal"],
            "home_goals": [2, 1],
            "away_goals": [1, 1],
        }
    )

    h2h_df = compute_head_to_head_features(fixtures, max_h2h_matches=5)

    # For fix1, no prior H2H meeting exists
    assert pd.isna(h2h_df.loc[0, "h2h_home_win_rate"])
    assert h2h_df.loc[0, "h2h_match_count"] == 0

    # For fix2 (Chelsea vs Arsenal on 2024-08-23), prior meeting fix1 was won by Arsenal
    # So Chelsea (current home team) win rate in prior meetings is 0.0
    assert h2h_df.loc[1, "h2h_home_win_rate"] == 0.0
    assert h2h_df.loc[1, "h2h_match_count"] == 1


def test_construct_feature_table_differentials_and_cold_start():
    """Test full feature table construction, diffs, and preservation of cold-start rows."""
    df_clean = pd.DataFrame(
        {
            "fixture_id": ["fix1", "fix2"],
            "league": ["ENG-Premier League", "ENG-Premier League"],
            "season": ["2425", "2425"],
            "week": [1, 2],
            "date": ["2024-08-16", "2024-08-23"],
            "home_team": ["Arsenal", "Chelsea"],
            "away_team": ["Chelsea", "Arsenal"],
            "home_goals": [2, 0],
            "away_goals": [0, 1],
            "home_xg": [1.8, 0.9],
            "away_xg": [0.4, 1.2],
            "is_xg_missing": [False, False],
        }
    )

    df_ratings = pd.DataFrame(
        {
            "fixture_id": ["fix1", "fix2"],
            "home_attack_rating": [1.0, 1.1],
            "away_attack_rating": [1.0, 0.9],
            "home_defense_rating": [1.0, 0.95],
            "away_defense_rating": [1.0, 1.05],
            "expected_home_goals": [1.509, 1.6],
            "expected_away_goals": [1.207, 1.1],
            "is_fallback_rating": [False, False],
        }
    )

    df_lambdas = pd.DataFrame(
        {
            "fixture_id": ["fix1", "fix2"],
            "home_team": ["Arsenal", "Chelsea"],
            "away_team": ["Chelsea", "Arsenal"],
            "base_home_lambda": [1.509, 1.6],
            "adjusted_home_lambda": [1.509, 1.6],
            "base_away_lambda": [1.207, 1.1],
            "adjusted_away_lambda": [1.207, 1.1],
            "num_home_absences_applied": [0, 1],
            "num_away_absences_applied": [0, 0],
        }
    )

    df_mc = pd.DataFrame(
        {
            "fixture_id": ["fix1", "fix2"],
            "mc_prob_home_win": [0.44, 0.55],
            "mc_prob_draw": [0.26, 0.24],
            "mc_prob_away_win": [0.30, 0.21],
            "mc_prob_over_2_5": [0.48, 0.50],
            "mc_prob_under_2_5": [0.52, 0.50],
            "mc_prob_btts": [0.50, 0.52],
        }
    )

    ft = construct_feature_table(df_clean, df_ratings, df_lambdas, df_mc)

    # Check length preservation
    assert len(ft) == 2

    # Check targets
    assert ft.loc[0, "target"] == "H"
    assert ft.loc[0, "target_encoded"] == 0
    assert ft.loc[1, "target"] == "A"
    assert ft.loc[1, "target_encoded"] == 2

    # Check differential columns exist
    diff_cols = [
        "points_form_diff",
        "goals_form_diff",
        "poisson_rating_diff",
        "adjusted_lambda_diff",
        "rest_days_diff",
    ]
    for c in diff_cols:
        assert c in ft.columns

    # Check fallback rating boolean indicators exist
    assert "is_fallback_rating_home" in ft.columns
    assert "is_fallback_rating_away" in ft.columns
    assert isinstance(ft.loc[0, "is_fallback_rating_home"], (bool, np.bool_))
