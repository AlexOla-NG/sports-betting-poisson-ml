"""
Unit tests for src/simulation/monte_carlo.py
"""

import numpy as np
import pandas as pd
import pytest

from src.simulation.monte_carlo import (
    run_monte_carlo_dataset,
    simulate_fixture_from_matrix,
    simulate_fixture_trials,
)
from src.simulation.scoreline_matrix import build_scoreline_matrix, extract_1x2_probabilities


def test_simulate_fixture_trials_basic():
    """Test basic simulation output structure and expectation alignment."""
    res = simulate_fixture_trials(lambda_home=1.8, lambda_away=1.2, num_trials=10000, random_seed=42)

    assert "mc_prob_home_win" in res
    assert "mc_prob_draw" in res
    assert "mc_prob_away_win" in res

    prob_sum = res["mc_prob_home_win"] + res["mc_prob_draw"] + res["mc_prob_away_win"]
    assert pytest.approx(prob_sum, abs=1e-6) == 1.0

    # Mean goals should be close to input lambdas for N=10,000
    assert pytest.approx(res["mc_mean_home_goals"], abs=0.05) == 1.8
    assert pytest.approx(res["mc_mean_away_goals"], abs=0.05) == 1.2


def test_simulate_fixture_trials_reproducibility():
    """Test that specifying random_seed produces identical results."""
    res1 = simulate_fixture_trials(1.5, 1.0, num_trials=5000, random_seed=123)
    res2 = simulate_fixture_trials(1.5, 1.0, num_trials=5000, random_seed=123)

    assert res1 == res2


def test_simulate_fixture_trials_invalid_inputs():
    """Test ValueError for invalid lambdas or trial count."""
    with pytest.raises(ValueError, match="cannot be NaN"):
        simulate_fixture_trials(np.nan, 1.2)

    with pytest.raises(ValueError, match="must be positive"):
        simulate_fixture_trials(0.0, 1.2)

    with pytest.raises(ValueError, match="must be at least 1"):
        simulate_fixture_trials(1.5, 1.2, num_trials=0)


def test_monte_carlo_convergence_to_analytical():
    """Test that Monte Carlo probabilities converge to exact analytical Poisson matrix probabilities as N increases."""
    lambda_home, lambda_away = 1.6, 1.1

    # Exact analytical probabilities from scoreline matrix
    matrix = build_scoreline_matrix(lambda_home, lambda_away, max_goals=6)
    analytical_1x2 = extract_1x2_probabilities(matrix)
    true_home_win = analytical_1x2["prob_home_win"]

    # Monte Carlo at N = 100, 1000, 10000
    errs = []
    for n_trials in [100, 1000, 10000]:
        sim_res = simulate_fixture_trials(lambda_home, lambda_away, num_trials=n_trials, random_seed=42)
        err = abs(sim_res["mc_prob_home_win"] - true_home_win)
        errs.append(err)

    # 10k trials error should be smaller than 100 trials error
    assert errs[2] < errs[0]
    # 10k trials error should be within 1.5% of analytical truth
    assert errs[2] < 0.015


def test_simulate_fixture_from_matrix():
    """Test sampling from discrete scoreline matrix."""
    matrix = build_scoreline_matrix(lambda_home=2.0, lambda_away=0.8, max_goals=6)
    res = simulate_fixture_from_matrix(matrix, num_trials=5000, random_seed=42)

    assert "mc_prob_home_win" in res
    assert res["mc_prob_home_win"] > res["mc_prob_away_win"]


def test_run_monte_carlo_dataset():
    """Test run_monte_carlo_dataset on a sample DataFrame."""
    df = pd.DataFrame(
        {
            "fixture_id": ["f1", "f2"],
            "home_team": ["Arsenal", "Liverpool"],
            "away_team": ["Spurs", "Everton"],
            "adjusted_home_lambda": [2.0, 2.5],
            "adjusted_away_lambda": [1.0, 0.5],
        }
    )

    out_df = run_monte_carlo_dataset(df, num_trials=1000, random_seed=42)

    mc_cols = [
        "mc_prob_home_win",
        "mc_prob_draw",
        "mc_prob_away_win",
        "mc_prob_over_2_5",
        "mc_prob_under_2_5",
        "mc_prob_btts",
    ]
    for col in mc_cols:
        assert col in out_df.columns

    assert len(out_df) == 2
    assert not out_df["mc_prob_home_win"].isna().any()
