"""
Unit tests for src/simulation/scoreline_matrix.py
"""

import numpy as np
import pandas as pd
import pytest

from src.simulation.scoreline_matrix import (
    build_scoreline_matrix,
    compute_fixture_scoreline_outcomes,
    extract_1x2_probabilities,
    extract_btts_probability,
    extract_most_likely_scoreline,
    extract_over_under_probabilities,
)


def test_build_scoreline_matrix_shape_and_normalization():
    """Test that build_scoreline_matrix returns correct shape and sums to 1.0."""
    matrix = build_scoreline_matrix(lambda_home=1.5, lambda_away=1.2, max_goals=6)
    assert matrix.shape == (7, 7)
    assert pytest.approx(np.sum(matrix), abs=1e-6) == 1.0


def test_build_scoreline_matrix_invalid_inputs():
    """Test that invalid lambdas or max_goals raise ValueError."""
    with pytest.raises(ValueError, match="cannot be NaN"):
        build_scoreline_matrix(np.nan, 1.2)

    with pytest.raises(ValueError, match="must be positive"):
        build_scoreline_matrix(0.0, 1.2)

    with pytest.raises(ValueError, match="must be positive"):
        build_scoreline_matrix(1.5, -0.5)

    with pytest.raises(ValueError, match="must be at least 1"):
        build_scoreline_matrix(1.5, 1.2, max_goals=0)


def test_extract_1x2_probabilities_sum_to_one():
    """Test that 1X2 probabilities sum to 1.0."""
    matrix = build_scoreline_matrix(lambda_home=1.8, lambda_away=1.1, max_goals=6)
    probs = extract_1x2_probabilities(matrix)

    assert "prob_home_win" in probs
    assert "prob_draw" in probs
    assert "prob_away_win" in probs

    total = probs["prob_home_win"] + probs["prob_draw"] + probs["prob_away_win"]
    assert pytest.approx(total, abs=1e-6) == 1.0


def test_known_symmetric_lambdas():
    """Test that equal home/away lambdas yield equal home/away win probabilities."""
    matrix = build_scoreline_matrix(lambda_home=1.5, lambda_away=1.5, max_goals=6)
    probs = extract_1x2_probabilities(matrix)

    assert pytest.approx(probs["prob_home_win"], abs=1e-6) == probs["prob_away_win"]


def test_extract_over_under_probabilities_sum_to_one():
    """Test that Over and Under probabilities sum to 1.0."""
    matrix = build_scoreline_matrix(lambda_home=2.0, lambda_away=1.0, max_goals=6)
    probs = extract_over_under_probabilities(matrix, threshold=2.5)

    assert "prob_over" in probs
    assert "prob_under" in probs
    assert pytest.approx(probs["prob_over"] + probs["prob_under"], abs=1e-6) == 1.0


def test_extract_btts_probability():
    """Test BTTS probability bounds."""
    matrix = build_scoreline_matrix(lambda_home=1.5, lambda_away=1.5, max_goals=6)
    btts_prob = extract_btts_probability(matrix)

    assert 0.0 <= btts_prob <= 1.0


def test_extract_most_likely_scoreline():
    """Test extraction of most likely scoreline for asymmetric lambdas."""
    # Strong home favorite
    matrix = build_scoreline_matrix(lambda_home=3.0, lambda_away=0.5, max_goals=6)
    h_goals, a_goals, prob = extract_most_likely_scoreline(matrix)

    assert h_goals > a_goals
    assert 0.0 < prob <= 1.0


def test_compute_fixture_scoreline_outcomes():
    """Test compute_fixture_scoreline_outcomes on a sample DataFrame."""
    df = pd.DataFrame(
        {
            "fixture_id": ["fix1", "fix2"],
            "home_team": ["Arsenal", "Chelsea"],
            "away_team": ["Everton", "Fulham"],
            "adjusted_home_lambda": [2.1, 1.4],
            "adjusted_away_lambda": [0.8, 1.2],
        }
    )

    out_df = compute_fixture_scoreline_outcomes(df, max_goals=6)

    expected_cols = [
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "prob_over_2_5",
        "prob_under_2_5",
        "prob_btts",
        "most_likely_score",
        "most_likely_score_prob",
    ]
    for col in expected_cols:
        assert col in out_df.columns

    # Verify rows
    assert len(out_df) == 2
    assert not out_df["prob_home_win"].isna().any()
    assert out_df["most_likely_score"].iloc[0] is not None
