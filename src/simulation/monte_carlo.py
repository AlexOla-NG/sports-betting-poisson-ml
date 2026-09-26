"""
Pipeline Stage: Phase 3 — Simulation (Task 3.2: Monte Carlo Simulation)

This module provides functions to simulate match outcomes via stochastic Monte Carlo sampling from
Poisson distributions or discrete scoreline matrices. It calculates simulated 1X2 win/draw/loss
probabilities, total goals distributions, and BTTS probabilities across N stochastic trials.
"""

from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import poisson


def simulate_fixture_trials(
    lambda_home: float,
    lambda_away: float,
    num_trials: int = 10000,
    random_seed: int | None = None,
) -> dict[str, float]:
    """Simulate a single fixture N times using independent Poisson distributions.

    Pipeline Stage: Phase 3 — Simulation

    Parameters
    ----------
    lambda_home : float
        Expected goals for the home team (must be > 0).
    lambda_away : float
        Expected goals for the away team (must be > 0).
    num_trials : int, optional
        Number of Monte Carlo trials to run, by default 10000.
    random_seed : int | None, optional
        Random seed for sampling reproducibility, by default None.

    Returns
    -------
    dict[str, float]
        Dictionary of simulated probabilities and expectations:
        ['mc_prob_home_win', 'mc_prob_draw', 'mc_prob_away_win',
         'mc_prob_over_2_5', 'mc_prob_under_2_5', 'mc_prob_btts',
         'mc_mean_home_goals', 'mc_mean_away_goals']
    """
    if np.isnan(lambda_home) or np.isnan(lambda_away):
        raise ValueError("Expected goals lambdas cannot be NaN.")
    if lambda_home <= 0 or lambda_away <= 0:
        raise ValueError(
            f"Expected goals lambdas must be positive. Got home={lambda_home}, away={lambda_away}"
        )
    if num_trials < 1:
        raise ValueError(f"num_trials must be at least 1. Got {num_trials}")

    rng = np.random.default_rng(random_seed)

    home_goals = rng.poisson(lambda_home, size=num_trials)
    away_goals = rng.poisson(lambda_away, size=num_trials)

    prob_home_win = float(np.mean(home_goals > away_goals))
    prob_draw = float(np.mean(home_goals == away_goals))
    prob_away_win = float(np.mean(home_goals < away_goals))

    total_goals = home_goals + away_goals
    prob_over_2_5 = float(np.mean(total_goals > 2.5))
    prob_under_2_5 = float(np.mean(total_goals < 2.5))

    prob_btts = float(np.mean((home_goals > 0) & (away_goals > 0)))

    return {
        "mc_prob_home_win": prob_home_win,
        "mc_prob_draw": prob_draw,
        "mc_prob_away_win": prob_away_win,
        "mc_prob_over_2_5": prob_over_2_5,
        "mc_prob_under_2_5": prob_under_2_5,
        "mc_prob_btts": prob_btts,
        "mc_mean_home_goals": float(np.mean(home_goals)),
        "mc_mean_away_goals": float(np.mean(away_goals)),
    }


def simulate_fixture_from_matrix(
    matrix: np.ndarray,
    num_trials: int = 10000,
    random_seed: int | None = None,
) -> dict[str, float]:
    """Simulate a single fixture N times by sampling from a scoreline probability matrix.

    Pipeline Stage: Phase 3 — Simulation

    Parameters
    ----------
    matrix : np.ndarray
        2D scoreline probability matrix where rows are Home goals and cols are Away goals.
    num_trials : int, optional
        Number of Monte Carlo trials to run, by default 10000.
    random_seed : int | None, optional
        Random seed for sampling reproducibility, by default None.

    Returns
    -------
    dict[str, float]
        Dictionary of simulated outcome probabilities.
    """
    if num_trials < 1:
        raise ValueError(f"num_trials must be at least 1. Got {num_trials}")

    flat_probs = matrix.ravel()
    total_prob = np.sum(flat_probs)
    if total_prob <= 0 or np.isnan(total_prob):
        raise ValueError("Scoreline matrix probabilities must sum to > 0 and not be NaN.")

    flat_probs = flat_probs / total_prob  # ensure normalized sum=1.0

    rng = np.random.default_rng(random_seed)
    chosen_indices = rng.choice(len(flat_probs), size=num_trials, p=flat_probs)

    home_goals, away_goals = np.unravel_index(chosen_indices, matrix.shape)

    prob_home_win = float(np.mean(home_goals > away_goals))
    prob_draw = float(np.mean(home_goals == away_goals))
    prob_away_win = float(np.mean(home_goals < away_goals))

    total_goals = home_goals + away_goals
    prob_over_2_5 = float(np.mean(total_goals > 2.5))
    prob_under_2_5 = float(np.mean(total_goals < 2.5))

    prob_btts = float(np.mean((home_goals > 0) & (away_goals > 0)))

    return {
        "mc_prob_home_win": prob_home_win,
        "mc_prob_draw": prob_draw,
        "mc_prob_away_win": prob_away_win,
        "mc_prob_over_2_5": prob_over_2_5,
        "mc_prob_under_2_5": prob_under_2_5,
        "mc_prob_btts": prob_btts,
        "mc_mean_home_goals": float(np.mean(home_goals)),
        "mc_mean_away_goals": float(np.mean(away_goals)),
    }


def run_monte_carlo_dataset(
    df: pd.DataFrame,
    home_lambda_col: str = "adjusted_home_lambda",
    away_lambda_col: str = "adjusted_away_lambda",
    num_trials: int = 10000,
    random_seed: int | None = 42,
) -> pd.DataFrame:
    """Run Monte Carlo simulation across all fixtures in a DataFrame.

    Pipeline Stage: Phase 3 — Simulation

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing fixture rows with home and away expected goals.
    home_lambda_col : str, optional
        Column name for home expected goals, by default 'adjusted_home_lambda'.
    away_lambda_col : str, optional
        Column name for away expected goals, by default 'adjusted_away_lambda'.
    num_trials : int, optional
        Number of trials per fixture, by default 10000.
    random_seed : int | None, optional
        Base random seed for reproducibility, by default 42.

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame in 1-row-per-fixture format with added columns:
        ['mc_prob_home_win', 'mc_prob_draw', 'mc_prob_away_win',
         'mc_prob_over_2_5', 'mc_prob_under_2_5', 'mc_prob_btts']
    """
    if home_lambda_col not in df.columns or away_lambda_col not in df.columns:
        raise KeyError(
            f"Required expected goal columns '{home_lambda_col}' and '{away_lambda_col}' not in DataFrame."
        )

    result_df = df.copy()

    mc_home_win_list = []
    mc_draw_list = []
    mc_away_win_list = []
    mc_over_2_5_list = []
    mc_under_2_5_list = []
    mc_btts_list = []

    for idx, row in result_df.iterrows():
        h_lam = row[home_lambda_col]
        a_lam = row[away_lambda_col]

        if pd.isna(h_lam) or pd.isna(a_lam) or h_lam <= 0 or a_lam <= 0:
            mc_home_win_list.append(np.nan)
            mc_draw_list.append(np.nan)
            mc_away_win_list.append(np.nan)
            mc_over_2_5_list.append(np.nan)
            mc_under_2_5_list.append(np.nan)
            mc_btts_list.append(np.nan)
            continue

        # Deterministic per-fixture seed if base seed is provided
        fixture_seed = (
            (random_seed + int(idx)) if random_seed is not None else None
        )

        sim_res = simulate_fixture_trials(
            lambda_home=float(h_lam),
            lambda_away=float(a_lam),
            num_trials=num_trials,
            random_seed=fixture_seed,
        )

        mc_home_win_list.append(sim_res["mc_prob_home_win"])
        mc_draw_list.append(sim_res["mc_prob_draw"])
        mc_away_win_list.append(sim_res["mc_prob_away_win"])
        mc_over_2_5_list.append(sim_res["mc_prob_over_2_5"])
        mc_under_2_5_list.append(sim_res["mc_prob_under_2_5"])
        mc_btts_list.append(sim_res["mc_prob_btts"])

    result_df["mc_prob_home_win"] = mc_home_win_list
    result_df["mc_prob_draw"] = mc_draw_list
    result_df["mc_prob_away_win"] = mc_away_win_list
    result_df["mc_prob_over_2_5"] = mc_over_2_5_list
    result_df["mc_prob_under_2_5"] = mc_under_2_5_list
    result_df["mc_prob_btts"] = mc_btts_list

    return result_df
