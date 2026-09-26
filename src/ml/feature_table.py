"""
Pipeline Stage: Phase 4 — ML Layer (Task 4.1: Feature Table Construction)

This module constructs the single-row-per-fixture feature table combining all upstream
pipeline outputs (clean fixtures, Poisson ratings, injury-adjusted lambdas, Monte Carlo
probabilities) and computing point-in-time non-leaking rolling features (form, xG, rest days,
head-to-head statistics, and fallback rating indicators).
"""

from typing import Any
import numpy as np
import pandas as pd

from src.config.loader import get_form_windows, get_xg_windows


def build_rolling_team_features(
    df: pd.DataFrame,
    points_window: int = 5,
    goals_window: int = 5,
    xg_for_window: int = 8,
    xg_against_window: int = 8,
) -> pd.DataFrame:
    """Compute point-in-time rolling form, xG, and rest days for home and away teams.

    Pipeline Stage: Phase 4 — ML Layer

    Strict point-in-time non-leakage guarantee: .shift(1) is applied to all team-level
    time-series before rolling windows are calculated. A fixture's own outcome is NEVER
    included in its pre-match feature values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of sorted fixtures containing goals, xG, dates, and team names.
    points_window : int, optional
        Rolling window length for points form, by default 5.
    goals_window : int, optional
        Rolling window length for goals form, by default 5.
    xg_for_window : int, optional
        Rolling window length for xG for form, by default 8.
    xg_against_window : int, optional
        Rolling window length for xG against form, by default 8.

    Returns
    -------
    pd.DataFrame
        DataFrame containing home and away rolling feature columns:
        ['home_points_form', 'away_points_form', 'home_goals_form', 'away_goals_form',
         'home_goals_against_form', 'away_goals_against_form', 'home_xg_for_form',
         'away_xg_for_form', 'home_xg_against_form', 'away_xg_against_form',
         'rest_days_home', 'rest_days_away']
    """
    work_df = df.copy()
    work_df["date"] = pd.to_datetime(work_df["date"])
    work_df = work_df.sort_values("date").reset_index(drop=True)

    team_matches = []
    for idx, row in work_df.iterrows():
        h_g = row.get("home_goals", np.nan)
        a_g = row.get("away_goals", np.nan)

        if pd.notna(h_g) and pd.notna(a_g):
            h_pts = 3 if h_g > a_g else (1 if h_g == a_g else 0)
            a_pts = 3 if a_g > h_g else (1 if h_g == a_g else 0)
        else:
            h_pts, a_pts = np.nan, np.nan

        team_matches.append(
            {
                "fixture_id": row["fixture_id"],
                "date": row["date"],
                "team": row["home_team"],
                "is_home": True,
                "goals_for": h_g,
                "goals_against": a_g,
                "xg_for": row.get("home_xg", np.nan),
                "xg_against": row.get("away_xg", np.nan),
                "points": h_pts,
            }
        )
        team_matches.append(
            {
                "fixture_id": row["fixture_id"],
                "date": row["date"],
                "team": row["away_team"],
                "is_home": False,
                "goals_for": a_g,
                "goals_against": h_g,
                "xg_for": row.get("away_xg", np.nan),
                "xg_against": row.get("home_xg", np.nan),
                "points": a_pts,
            }
        )

    tm_df = pd.DataFrame(team_matches)
    tm_df = tm_df.sort_values(["team", "date"]).reset_index(drop=True)

    # Apply .shift(1) BEFORE rolling windows to guarantee non-leakage
    tm_df["rolling_points_form"] = tm_df.groupby("team")["points"].transform(
        lambda s: s.shift(1).rolling(points_window, min_periods=1).mean()
    )
    tm_df["rolling_goals_form"] = tm_df.groupby("team")["goals_for"].transform(
        lambda s: s.shift(1).rolling(goals_window, min_periods=1).mean()
    )
    tm_df["rolling_goals_against_form"] = tm_df.groupby("team")["goals_against"].transform(
        lambda s: s.shift(1).rolling(goals_window, min_periods=1).mean()
    )
    tm_df["rolling_xg_for_form"] = tm_df.groupby("team")["xg_for"].transform(
        lambda s: s.shift(1).rolling(xg_for_window, min_periods=1).mean()
    )
    tm_df["rolling_xg_against_form"] = tm_df.groupby("team")["xg_against"].transform(
        lambda s: s.shift(1).rolling(xg_against_window, min_periods=1).mean()
    )

    # Compute rest days strictly from prior date using .shift(1)
    tm_df["prev_date"] = tm_df.groupby("team")["date"].shift(1)
    tm_df["rest_days"] = (tm_df["date"] - tm_df["prev_date"]).dt.days

    # Split back to home and away per fixture
    home_tm = tm_df[tm_df["is_home"]].set_index("fixture_id")
    away_tm = tm_df[~tm_df["is_home"]].set_index("fixture_id")

    rolling_df = pd.DataFrame(index=work_df["fixture_id"])
    rolling_df["home_points_form"] = home_tm["rolling_points_form"]
    rolling_df["away_points_form"] = away_tm["rolling_points_form"]
    rolling_df["home_goals_form"] = home_tm["rolling_goals_form"]
    rolling_df["away_goals_form"] = away_tm["rolling_goals_form"]
    rolling_df["home_goals_against_form"] = home_tm["rolling_goals_against_form"]
    rolling_df["away_goals_against_form"] = away_tm["rolling_goals_against_form"]
    rolling_df["home_xg_for_form"] = home_tm["rolling_xg_for_form"]
    rolling_df["away_xg_for_form"] = away_tm["rolling_xg_for_form"]
    rolling_df["home_xg_against_form"] = home_tm["rolling_xg_against_form"]
    rolling_df["away_xg_against_form"] = away_tm["rolling_xg_against_form"]
    rolling_df["rest_days_home"] = home_tm["rest_days"]
    rolling_df["rest_days_away"] = away_tm["rest_days"]

    return rolling_df.reset_index()


def compute_head_to_head_features(
    df: pd.DataFrame, max_h2h_matches: int = 5
) -> pd.DataFrame:
    """Compute head-to-head win rate over prior meetings strictly before current match date.

    Pipeline Stage: Phase 4 — ML Layer

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of sorted fixtures containing fixture_id, date, home_team, away_team, goals.
    max_h2h_matches : int, optional
        Maximum number of prior head-to-head meetings to consider, by default 5.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['fixture_id', 'h2h_home_win_rate', 'h2h_match_count'].
    """
    work_df = df.copy()
    work_df["date"] = pd.to_datetime(work_df["date"])

    h2h_rates = []
    h2h_counts = []

    for idx, row in work_df.iterrows():
        h_team = row["home_team"]
        a_team = row["away_team"]
        m_date = row["date"]

        # Filter matches strictly before m_date between these two teams
        prior = work_df[
            (work_df["date"] < m_date)
            & (
                ((work_df["home_team"] == h_team) & (work_df["away_team"] == a_team))
                | ((work_df["home_team"] == a_team) & (work_df["away_team"] == h_team))
            )
        ]

        if len(prior) == 0:
            h2h_rates.append(np.nan)
            h2h_counts.append(0)
        else:
            recent = prior.tail(max_h2h_matches)
            h_wins = 0
            for _, p_row in recent.iterrows():
                p_h_g = p_row.get("home_goals", np.nan)
                p_a_g = p_row.get("away_goals", np.nan)
                if pd.isna(p_h_g) or pd.isna(p_a_g):
                    continue

                if p_row["home_team"] == h_team and p_h_g > p_a_g:
                    h_wins += 1
                elif p_row["away_team"] == h_team and p_a_g > p_h_g:
                    h_wins += 1

            h2h_rates.append(h_wins / len(recent))
            h2h_counts.append(len(recent))

    return pd.DataFrame(
        {
            "fixture_id": work_df["fixture_id"],
            "h2h_home_win_rate": h2h_rates,
            "h2h_match_count": h2h_counts,
        }
    )


def construct_feature_table(
    df_clean: pd.DataFrame,
    df_ratings: pd.DataFrame,
    df_lambdas: pd.DataFrame,
    df_mc: pd.DataFrame,
    df_elo: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Combine all upstream pipeline outputs into a single fixture-level feature table.

    Pipeline Stage: Phase 4 — ML Layer

    Enforces 1-row-per-fixture layout, point-in-time correctness (.shift(1)),
    and explicit diff features across form, xG, ratings, lambdas, and rest days.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned fixtures table (data/processed/clean_fixtures.parquet).
    df_ratings : pd.DataFrame
        Poisson ratings table (data/processed/ratings.parquet).
    df_lambdas : pd.DataFrame
        Adjusted lambdas table (data/processed/adjusted_lambdas.parquet).
    df_mc : pd.DataFrame
        Monte Carlo probabilities table (data/processed/mc_probabilities.parquet).
    df_elo : pd.DataFrame | None, optional
        Optional ClubElo ratings table, by default None.
    config : dict[str, Any] | None, optional
        Pipeline configuration dictionary, by default None.

    Returns
    -------
    pd.DataFrame
        Feature table with 1 row per fixture and home/away differential features.
    """
    if config is not None:
        pts_win, g_win = get_form_windows(config)
        xg_f_win, xg_a_win = get_xg_windows(config)
    else:
        pts_win, g_win, xg_f_win, xg_a_win = 5, 5, 8, 8

    # 1. Base merge on fixture_id
    base_cols = [
        "fixture_id",
        "league",
        "season",
        "week",
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "is_xg_missing",
    ]
    merged = df_clean[base_cols].copy()

    # Join Poisson Ratings
    ratings_cols = [
        "fixture_id",
        "home_attack_rating",
        "home_defense_rating",
        "away_attack_rating",
        "away_defense_rating",
        "expected_home_goals",
        "expected_away_goals",
        "is_fallback_rating",
    ]
    merged = merged.merge(
        df_ratings[[c for c in ratings_cols if c in df_ratings.columns]],
        on="fixture_id",
        how="left",
    )

    # Join Adjusted Lambdas
    lambda_cols = [
        "fixture_id",
        "base_home_lambda",
        "adjusted_home_lambda",
        "base_away_lambda",
        "adjusted_away_lambda",
        "num_home_absences_applied",
        "num_away_absences_applied",
    ]
    merged = merged.merge(
        df_lambdas[[c for c in lambda_cols if c in df_lambdas.columns]],
        on="fixture_id",
        how="left",
    )

    # Join Monte Carlo Probabilities
    mc_cols = [
        "fixture_id",
        "mc_prob_home_win",
        "mc_prob_draw",
        "mc_prob_away_win",
        "mc_prob_over_2_5",
        "mc_prob_under_2_5",
        "mc_prob_btts",
    ]
    merged = merged.merge(
        df_mc[[c for c in mc_cols if c in df_mc.columns]],
        on="fixture_id",
        how="left",
    )

    # Verify merge preserved exact 1-row-per-fixture total count
    if len(merged) != len(df_clean):
        raise ValueError(
            f"Row count mismatch after merging: expected {len(df_clean)}, got {len(merged)}"
        )

    # 2. Compute Rolling Form, xG, and Rest Days
    rolling_df = build_rolling_team_features(
        df_clean,
        points_window=pts_win,
        goals_window=g_win,
        xg_for_window=xg_f_win,
        xg_against_window=xg_a_win,
    )
    merged = merged.merge(rolling_df, on="fixture_id", how="left")

    # 3. Compute Head-to-Head Win Rate
    h2h_df = compute_head_to_head_features(df_clean, max_h2h_matches=5)
    merged = merged.merge(h2h_df, on="fixture_id", how="left")

    # 4. Derive Differential Features (Home minus Away)
    merged["points_form_diff"] = merged["home_points_form"] - merged["away_points_form"]
    merged["goals_form_diff"] = merged["home_goals_form"] - merged["away_goals_form"]
    merged["goals_against_form_diff"] = (
        merged["home_goals_against_form"] - merged["away_goals_against_form"]
    )
    merged["xg_for_form_diff"] = merged["home_xg_for_form"] - merged["away_xg_for_form"]
    merged["xg_against_form_diff"] = (
        merged["home_xg_against_form"] - merged["away_xg_against_form"]
    )

    merged["xg_diff"] = (
        merged["home_xg_for_form"] - merged["away_xg_against_form"]
    ) - (merged["away_xg_for_form"] - merged["home_xg_against_form"])

    merged["poisson_attack_diff"] = (
        merged["home_attack_rating"] - merged["away_attack_rating"]
    )
    merged["poisson_defense_diff"] = (
        merged["home_defense_rating"] - merged["away_defense_rating"]
    )
    merged["poisson_rating_diff"] = (
        merged["home_attack_rating"] - merged["away_attack_rating"]
    ) + (merged["away_defense_rating"] - merged["home_defense_rating"])

    merged["adjusted_lambda_diff"] = (
        merged["adjusted_home_lambda"] - merged["adjusted_away_lambda"]
    )
    merged["rest_days_diff"] = merged["rest_days_home"] - merged["rest_days_away"]

    # 5. Sourced Fallback Rating Indicator Features
    if "is_fallback_rating" in merged.columns:
        merged["is_fallback_rating_home"] = merged["is_fallback_rating"].astype(bool)
        merged["is_fallback_rating_away"] = merged["is_fallback_rating"].astype(bool)
    else:
        merged["is_fallback_rating"] = False
        merged["is_fallback_rating_home"] = False
        merged["is_fallback_rating_away"] = False

    # 6. Optional ClubElo features
    if df_elo is not None and "elo_home" in df_elo.columns and "elo_away" in df_elo.columns:
        merged = merged.merge(
            df_elo[["fixture_id", "elo_home", "elo_away"]], on="fixture_id", how="left"
        )
        merged["elo_diff"] = merged["elo_home"] - merged["elo_away"]
    else:
        merged["elo_home"] = np.nan
        merged["elo_away"] = np.nan
        merged["elo_diff"] = np.nan

    # 7. Post-match Target Column Generation
    def _compute_target(row):
        h_g, a_g = row["home_goals"], row["away_goals"]
        if pd.isna(h_g) or pd.isna(a_g):
            return np.nan
        if h_g > a_g:
            return "H"
        elif h_g == a_g:
            return "D"
        else:
            return "A"

    def _compute_target_encoded(row):
        t = row["target"]
        if t == "H":
            return 0
        elif t == "D":
            return 1
        elif t == "A":
            return 2
        return np.nan

    merged["target"] = merged.apply(_compute_target, axis=1)
    merged["target_encoded"] = merged.apply(_compute_target_encoded, axis=1)

    return merged
