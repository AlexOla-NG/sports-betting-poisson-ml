"""Poisson attack and defense ratings module.

Pipeline Stage: 03_ratings
Inputs: Cleaned fixtures DataFrame (data/processed/clean_fixtures.parquet)
Outputs: Rolling attack/defense ratings per team (data/processed/ratings.parquet)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def prepare_glm_dataset(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape 1-row-per-fixture DataFrame into 2-row-per-fixture GLM observations.

    Parameters
    ----------
    fixtures_df : pd.DataFrame
        Cleaned fixtures table with home_team, away_team, home_goals, away_goals.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: team, opponent, home, goals.
    """
    valid_fixtures = fixtures_df.dropna(subset=["home_goals", "away_goals"]).copy()
    if valid_fixtures.empty:
        return pd.DataFrame(columns=["team", "opponent", "home", "goals"])

    home_obs = pd.DataFrame(
        {
            "team": valid_fixtures["home_team"],
            "opponent": valid_fixtures["away_team"],
            "home": 1,
            "goals": valid_fixtures["home_goals"].astype(float),
        }
    )

    away_obs = pd.DataFrame(
        {
            "team": valid_fixtures["away_team"],
            "opponent": valid_fixtures["home_team"],
            "home": 0,
            "goals": valid_fixtures["away_goals"].astype(float),
        }
    )

    return pd.concat([home_obs, away_obs], ignore_index=True)


def fit_poisson_ratings(
    fixtures_df: pd.DataFrame, home_field_advantage: bool = True
) -> tuple[dict[str, float], dict[str, float], float]:
    """Fit Poisson GLM on a set of past fixtures to estimate team attack and defense strength.

    Parameters
    ----------
    fixtures_df : pd.DataFrame
        Past match fixtures.
    home_field_advantage : bool
        Whether to include home advantage term in GLM. Default is True.

    Returns
    -------
    tuple[dict[str, float], dict[str, float], float]
        (attack_ratings, defense_ratings, home_advantage_multiplier)
        Attack and defense ratings are normalized so that league average = 1.0.
    """
    teams = sorted(
        set(fixtures_df["home_team"].dropna()).union(set(fixtures_df["away_team"].dropna()))
    )
    if len(teams) < 2 or len(fixtures_df) < 2:
        # Default cold start ratings
        default_attack = {team: 1.0 for team in teams}
        default_defense = {team: 1.0 for team in teams}
        return default_attack, default_defense, 1.25

    glm_data = prepare_glm_dataset(fixtures_df)
    if len(glm_data) < 4:
        default_attack = {team: 1.0 for team in teams}
        default_defense = {team: 1.0 for team in teams}
        return default_attack, default_defense, 1.25

    formula = "goals ~ home + C(team) + C(opponent)" if home_field_advantage else "goals ~ C(team) + C(opponent)"

    try:
        model = smf.glm(formula=formula, data=glm_data, family=sm.families.Poisson()).fit()
        params = model.params

        # Base intercept + home advantage
        intercept = params.get("Intercept", 0.0)
        home_coef = params.get("home", 0.22) if home_field_advantage else 0.0
        home_adv_multiplier = float(np.exp(home_coef))

        # Extract attack coefficients
        team_coefs = {}
        ref_team = sorted(glm_data["team"].unique())[0]

        for team in teams:
            team_key = f"C(team)[T.{team}]"
            coef = params.get(team_key, 0.0) if team != ref_team else 0.0
            team_coefs[team] = coef

        # Extract defense coefficients
        opp_coefs = {}
        for team in teams:
            opp_key = f"C(opponent)[T.{team}]"
            coef = params.get(opp_key, 0.0) if team != ref_team else 0.0
            opp_coefs[team] = coef

        # Normalize so mean attack = 1.0 and mean defense = 1.0
        raw_att = np.exp(list(team_coefs.values()))
        raw_def = np.exp(list(opp_coefs.values()))

        mean_att = np.mean(raw_att) if len(raw_att) > 0 else 1.0
        mean_def = np.mean(raw_def) if len(raw_def) > 0 else 1.0

        attack_ratings = {
            team: float(np.exp(coef) / mean_att) for team, coef in team_coefs.items()
        }
        defense_ratings = {
            team: float(np.exp(coef) / mean_def) for team, coef in opp_coefs.items()
        }

        return attack_ratings, defense_ratings, home_adv_multiplier

    except Exception:
        # Fallback to empirical average goals if GLM fails to converge
        avg_home_goals = glm_data[glm_data["home"] == 1]["goals"].mean() if not glm_data.empty else 1.5
        avg_away_goals = glm_data[glm_data["home"] == 0]["goals"].mean() if not glm_data.empty else 1.1
        home_adv = (avg_home_goals / avg_away_goals) if avg_away_goals > 0 else 1.25

        attack_ratings = {}
        defense_ratings = {}
        league_avg_goals = glm_data["goals"].mean() if not glm_data.empty else 1.3

        for team in teams:
            team_scored = glm_data[glm_data["team"] == team]["goals"].mean()
            team_conceded = glm_data[glm_data["opponent"] == team]["goals"].mean()

            att = (team_scored / league_avg_goals) if (pd.notna(team_scored) and league_avg_goals > 0) else 1.0
            defn = (team_conceded / league_avg_goals) if (pd.notna(team_conceded) and league_avg_goals > 0) else 1.0

            attack_ratings[team] = float(att)
            defense_ratings[team] = float(defn)

        return attack_ratings, defense_ratings, float(home_adv)


def compute_rolling_ratings(
    fixtures_df: pd.DataFrame, window: int = 8, home_field_advantage: bool = True
) -> pd.DataFrame:
    """Compute point-in-time Poisson attack/defense ratings using rolling N-match window.

    Strict point-in-time correctness (.shift(1)) is enforced: ratings for match K
    are computed ONLY using matches played prior to match K.

    Parameters
    ----------
    fixtures_df : pd.DataFrame
        Cleaned fixtures table sorted chronologically by date.
    window : int
        Number of past matches per team to include in rolling GLM fit. Default is 8.
    home_field_advantage : bool
        Whether to include home advantage term in GLM. Default is True.

    Returns
    -------
    pd.DataFrame
        Fixtures table enriched with home_attack_rating, home_defense_rating,
        away_attack_rating, away_defense_rating, expected_home_goals, expected_away_goals.
    """
    df = fixtures_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    home_att_list = []
    home_def_list = []
    away_att_list = []
    away_def_list = []
    expected_home_list = []
    expected_away_list = []

    # Maintain team match history
    team_history: dict[str, list[int]] = {}

    for idx, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        # Past fixtures strictly prior to current match (idx)
        past_fixtures = df.iloc[:idx].copy()

        if idx == 0 or past_fixtures.empty:
            # Match 1: Cold start default ratings
            h_att, h_def = 1.0, 1.0
            a_att, a_def = 1.0, 1.0
            home_adv = 1.25
        else:
            # Filter past matches involving home_team or away_team in the last N matches
            home_past_indices = past_fixtures[
                (past_fixtures["home_team"] == home_team) | (past_fixtures["away_team"] == home_team)
            ].tail(window).index

            away_past_indices = past_fixtures[
                (past_fixtures["home_team"] == away_team) | (past_fixtures["away_team"] == away_team)
            ].tail(window).index

            window_indices = sorted(set(home_past_indices).union(set(away_past_indices)))
            subset_fixtures = past_fixtures.loc[window_indices]

            if len(subset_fixtures) < 2:
                h_att, h_def = 1.0, 1.0
                a_att, a_def = 1.0, 1.0
                home_adv = 1.25
            else:
                att_map, def_map, home_adv = fit_poisson_ratings(
                    subset_fixtures, home_field_advantage=home_field_advantage
                )
                h_att = att_map.get(home_team, 1.0)
                h_def = def_map.get(home_team, 1.0)
                a_att = att_map.get(away_team, 1.0)
                a_def = def_map.get(away_team, 1.0)

        # Baseline expected goal calculation
        league_avg_lambda = 1.35
        exp_home = league_avg_lambda * h_att * a_def * (home_adv ** 0.5)
        exp_away = league_avg_lambda * a_att * h_def / (home_adv ** 0.5)

        home_att_list.append(h_att)
        home_def_list.append(h_def)
        away_att_list.append(a_att)
        away_def_list.append(a_def)
        expected_home_list.append(round(float(exp_home), 3))
        expected_away_list.append(round(float(exp_away), 3))

    df["home_attack_rating"] = home_att_list
    df["home_defense_rating"] = home_def_list
    df["away_attack_rating"] = away_att_list
    df["away_defense_rating"] = away_def_list
    df["expected_home_goals"] = expected_home_list
    df["expected_away_goals"] = expected_away_list

    return df
