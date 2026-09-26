"""
Pipeline Stage: Phase 3 — Simulation (Task 3.1: Scoreline Matrix)

This module provides functions to construct scoreline probability matrices from
home and away expected goal lambdas under independent Poisson distributions, and to
extract derived betting outcome probabilities (1X2, Over/Under totals, Both Teams
To Score (BTTS), and most likely scorelines).
"""

from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import poisson


def build_scoreline_matrix(
    lambda_home: float, lambda_away: float, max_goals: int = 6
) -> np.ndarray:
    """Construct a scoreline probability matrix using independent Poisson distributions.

    Pipeline Stage: Phase 3 — Simulation

    Parameters
    ----------
    lambda_home : float
        Expected goals for the home team (must be > 0).
    lambda_away : float
        Expected goals for the away team (must be > 0).
    max_goals : int, optional
        Maximum number of goals per team in matrix grid, by default 6 (0 to 6 goals).

    Returns
    -------
    np.ndarray
        2D numpy array of shape (max_goals + 1, max_goals + 1) where entry [i, j]
        is the probability of Home scoring i goals and Away scoring j goals.
    """
    if np.isnan(lambda_home) or np.isnan(lambda_away):
        raise ValueError("Expected goals lambdas cannot be NaN.")
    if lambda_home <= 0 or lambda_away <= 0:
        raise ValueError(
            f"Expected goals lambdas must be positive. Got home={lambda_home}, away={lambda_away}"
        )
    if max_goals < 1:
        raise ValueError(f"max_goals must be at least 1. Got {max_goals}")

    goals_range = np.arange(max_goals + 1)
    home_probs = poisson.pmf(goals_range, lambda_home)
    away_probs = poisson.pmf(goals_range, lambda_away)

    # Outer product for independent Poisson probabilities
    matrix = np.outer(home_probs, away_probs)

    # Re-normalize to account for grid truncation beyond max_goals
    total_prob = np.sum(matrix)
    if total_prob > 0:
        matrix = matrix / total_prob

    return matrix


def extract_1x2_probabilities(matrix: np.ndarray) -> dict[str, float]:
    """Extract 1X2 (Home Win, Draw, Away Win) probabilities from a scoreline matrix.

    Pipeline Stage: Phase 3 — Simulation

    Parameters
    ----------
    matrix : np.ndarray
        Scoreline matrix where rows are Home goals and columns are Away goals.

    Returns
    -------
    dict[str, float]
        Dictionary with keys 'prob_home_win', 'prob_draw', 'prob_away_win'.
    """
    prob_draw = float(np.trace(matrix))
    prob_home_win = float(np.sum(np.tril(matrix, -1)))
    prob_away_win = float(np.sum(np.triu(matrix, 1)))

    return {
        "prob_home_win": prob_home_win,
        "prob_draw": prob_draw,
        "prob_away_win": prob_away_win,
    }


def extract_over_under_probabilities(
    matrix: np.ndarray, threshold: float = 2.5
) -> dict[str, float]:
    """Extract Over/Under total goals probabilities from a scoreline matrix.

    Pipeline Stage: Phase 3 — Simulation

    Parameters
    ----------
    matrix : np.ndarray
        Scoreline matrix where rows are Home goals and columns are Away goals.
    threshold : float, optional
        Goal threshold (e.g. 2.5), by default 2.5.

    Returns
    -------
    dict[str, float]
        Dictionary with keys 'prob_over', 'prob_under'.
    """
    max_goals = matrix.shape[0] - 1
    goals = np.arange(max_goals + 1)
    grid_total = goals[:, None] + goals[None, :]

    prob_over = float(np.sum(matrix[grid_total > threshold]))
    prob_under = float(np.sum(matrix[grid_total < threshold]))

    return {"prob_over": prob_over, "prob_under": prob_under}


def extract_btts_probability(matrix: np.ndarray) -> float:
    """Extract Both Teams To Score (BTTS - Yes) probability from a scoreline matrix.

    Pipeline Stage: Phase 3 — Simulation

    Parameters
    ----------
    matrix : np.ndarray
        Scoreline matrix where rows are Home goals and columns are Away goals.

    Returns
    -------
    float
        Probability of both home and away teams scoring at least 1 goal.
    """
    return float(np.sum(matrix[1:, 1:]))


def extract_most_likely_scoreline(matrix: np.ndarray) -> tuple[int, int, float]:
    """Extract the single most likely exact scoreline and its probability.

    Pipeline Stage: Phase 3 — Simulation

    Parameters
    ----------
    matrix : np.ndarray
        Scoreline matrix where rows are Home goals and columns are Away goals.

    Returns
    -------
    tuple[int, int, float]
        (home_goals, away_goals, probability) for the highest probability matrix cell.
    """
    max_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    home_goals, away_goals = int(max_idx[0]), int(max_idx[1])
    prob = float(matrix[home_goals, away_goals])

    return home_goals, away_goals, prob


def compute_fixture_scoreline_outcomes(
    df: pd.DataFrame,
    home_lambda_col: str = "adjusted_home_lambda",
    away_lambda_col: str = "adjusted_away_lambda",
    max_goals: int = 6,
) -> pd.DataFrame:
    """Compute scoreline outcome probabilities for every fixture in a DataFrame.

    Pipeline Stage: Phase 3 — Simulation

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing fixture rows with home and away expected goals.
    home_lambda_col : str, optional
        Column name for home expected goals, by default 'adjusted_home_lambda'.
    away_lambda_col : str, optional
        Column name for away expected goals, by default 'adjusted_away_lambda'.
    max_goals : int, optional
        Maximum goals per team bound for matrix calculation, by default 6.

    Returns
    -------
    pd.DataFrame
        Augmented DataFrame in 1-row-per-fixture format with added columns:
        ['prob_home_win', 'prob_draw', 'prob_away_win', 'prob_over_2_5',
         'prob_under_2_5', 'prob_btts', 'most_likely_score', 'most_likely_score_prob']
    """
    if home_lambda_col not in df.columns or away_lambda_col not in df.columns:
        raise KeyError(
            f"Required expected goal columns '{home_lambda_col}' and '{away_lambda_col}' not in DataFrame."
        )

    result_df = df.copy()

    prob_home_win_list = []
    prob_draw_list = []
    prob_away_win_list = []
    prob_over_2_5_list = []
    prob_under_2_5_list = []
    prob_btts_list = []
    most_likely_score_list = []
    most_likely_score_prob_list = []

    for _, row in result_df.iterrows():
        h_lam = row[home_lambda_col]
        a_lam = row[away_lambda_col]

        if pd.isna(h_lam) or pd.isna(a_lam) or h_lam <= 0 or a_lam <= 0:
            prob_home_win_list.append(np.nan)
            prob_draw_list.append(np.nan)
            prob_away_win_list.append(np.nan)
            prob_over_2_5_list.append(np.nan)
            prob_under_2_5_list.append(np.nan)
            prob_btts_list.append(np.nan)
            most_likely_score_list.append(None)
            most_likely_score_prob_list.append(np.nan)
            continue

        matrix = build_scoreline_matrix(
            lambda_home=float(h_lam),
            lambda_away=float(a_lam),
            max_goals=max_goals,
        )

        probs_1x2 = extract_1x2_probabilities(matrix)
        probs_ou = extract_over_under_probabilities(matrix, threshold=2.5)
        prob_btts = extract_btts_probability(matrix)
        h_g, a_g, ml_p = extract_most_likely_scoreline(matrix)

        prob_home_win_list.append(probs_1x2["prob_home_win"])
        prob_draw_list.append(probs_1x2["prob_draw"])
        prob_away_win_list.append(probs_1x2["prob_away_win"])
        prob_over_2_5_list.append(probs_ou["prob_over"])
        prob_under_2_5_list.append(probs_ou["prob_under"])
        prob_btts_list.append(prob_btts)
        most_likely_score_list.append(f"{h_g}-{a_g}")
        most_likely_score_prob_list.append(ml_p)

    result_df["prob_home_win"] = prob_home_win_list
    result_df["prob_draw"] = prob_draw_list
    result_df["prob_away_win"] = prob_away_win_list
    result_df["prob_over_2_5"] = prob_over_2_5_list
    result_df["prob_under_2_5"] = prob_under_2_5_list
    result_df["prob_btts"] = prob_btts_list
    result_df["most_likely_score"] = most_likely_score_list
    result_df["most_likely_score_prob"] = most_likely_score_prob_list

    return result_df
