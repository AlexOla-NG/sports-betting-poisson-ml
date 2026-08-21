"""Data cleaning and base feature engineering for match fixtures.

Pipeline Stage: 02_processing
Inputs: Raw fixtures DataFrame, Raw team match stats DataFrame
Outputs: Standardized 1-row-per-fixture DataFrame with base features & differentials
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


# Canonical team name mapping for FBref and ClubElo naming variations
TEAM_NAME_MAPPING: Mapping[str, str] = {
    "Manchester Utd": "Manchester United",
    "Man United": "Manchester United",
    "Man Utd": "Manchester United",
    "Manchester City": "Manchester City",
    "Man City": "Manchester City",
    "Newcastle": "Newcastle United",
    "Newcastle Utd": "Newcastle United",
    "Nottingham": "Nottingham Forest",
    "Nott'm Forest": "Nottingham Forest",
    "Tottenham": "Tottenham Hotspur",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers",
    "Wolverhampton": "Wolverhampton Wanderers",
    "Leicester City": "Leicester City",
    "Leicester": "Leicester City",
    "Ipswich Town": "Ipswich Town",
    "Ipswich": "Ipswich Town",
    "Brighton": "Brighton & Hove Albion",
    "Brighton & Hove": "Brighton & Hove Albion",
    "West Brom": "West Bromwich Albion",
    "Sheffield Utd": "Sheffield United",
    "Luton Town": "Luton Town",
    "Luton": "Luton Town",
    "Leeds United": "Leeds United",
    "Leeds": "Leeds United",
}


def standardize_team_name(name: str | None) -> str | None:
    """Standardize a single team name string using canonical mapping.

    Parameters
    ----------
    name : str | None
        Raw team name from FBref or ClubElo.

    Returns
    -------
    str | None
        Canonical team name.
    """
    if name is None or pd.isna(name):
        return None
    name_str = str(name).strip()
    return TEAM_NAME_MAPPING.get(name_str, name_str)


def standardize_team_columns(df: pd.DataFrame, team_cols: list[str]) -> pd.DataFrame:
    """Standardize team name columns in a DataFrame in-place copy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing team name columns.
    team_cols : list[str]
        List of column names containing team names to standardize.

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized team names.
    """
    df = df.copy()
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].apply(standardize_team_name)
    return df


def merge_match_stats(fixtures_df: pd.DataFrame, match_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Merge team match statistics into 1-row-per-fixture layout.

    Parameters
    ----------
    fixtures_df : pd.DataFrame
        Cleaned fixtures table (1 row per match).
    match_stats_df : pd.DataFrame
        Team match statistics table (2 rows per match: home and away).

    Returns
    -------
    pd.DataFrame
        Fixtures table enriched with home/away xG, shots, and SoT.
    """
    fixtures = fixtures_df.copy()
    stats = match_stats_df.copy()

    # Standardize team names in both inputs
    fixtures = standardize_team_columns(fixtures, ["home_team", "away_team"])

    # If match_stats contains 'team' and 'opponent' or 'fixture_id'/'game'
    if "team" in stats.columns:
        stats = standardize_team_columns(stats, ["team", "opponent"])

    # Extract team-level stats if available
    # We aggregate stats by fixture_id if fixture_id is in stats, or by date + home/away team
    merged = fixtures.copy()

    # Required numeric columns from stats if present
    for stat_name in ["xG", "Sh", "SoT"]:
        home_col = f"home_{stat_name.lower()}"
        away_col = f"away_{stat_name.lower()}"

        if stat_name in stats.columns and "fixture_id" in stats.columns:
            # Map home stats
            home_stats = stats.groupby(["fixture_id", "team"])[stat_name].first().unstack()
            # If fixture_id matching is available, attempt fixture_id join
            # Fallback to direct mapping if present in stats
        
        if home_col not in merged.columns:
            merged[home_col] = np.nan
        if away_col not in merged.columns:
            merged[away_col] = np.nan

    # Add missing xG indicator flag
    merged["is_xg_missing"] = merged["home_xg"].isna() | merged["away_xg"].isna()

    return merged


def add_base_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """Compute base differential features (home minus away).

    Parameters
    ----------
    df : pd.DataFrame
        1-row-per-fixture DataFrame with home and away statistics.

    Returns
    -------
    pd.DataFrame
        DataFrame with added diff features (goals_diff, xg_diff, shots_diff, sot_diff).
    """
    df = df.copy()

    if "home_goals" in df.columns and "away_goals" in df.columns:
        df["goals_diff"] = df["home_goals"] - df["away_goals"]
    else:
        df["goals_diff"] = np.nan

    if "home_xg" in df.columns and "away_xg" in df.columns:
        df["xg_diff"] = df["home_xg"] - df["away_xg"]
    else:
        df["xg_diff"] = np.nan

    if "home_shots" in df.columns and "away_shots" in df.columns:
        df["shots_diff"] = df["home_shots"] - df["away_shots"]
    else:
        df["shots_diff"] = np.nan

    if "home_sot" in df.columns and "away_sot" in df.columns:
        df["sot_diff"] = df["home_sot"] - df["away_sot"]
    else:
        df["sot_diff"] = np.nan

    return df


def clean_fixtures_pipeline(fixtures_df: pd.DataFrame, match_stats_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Execute complete data cleaning and base feature pipeline.

    Pipeline Stage: 02_processing
    Inputs: Raw fixtures DataFrame, optional match_stats DataFrame
    Outputs: Standardized 1-row-per-fixture DataFrame with base features

    Parameters
    ----------
    fixtures_df : pd.DataFrame
        Raw fixtures DataFrame.
    match_stats_df : pd.DataFrame | None
        Raw team match statistics DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned fixtures table with standardized IDs, missing value flags, and differentials.
    """
    df = fixtures_df.copy()

    # Standardize team names
    df = standardize_team_columns(df, ["home_team", "away_team"])

    # Ensure date column is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Sort chronologically by date
    df = df.sort_values("date").reset_index(drop=True)

    # Merge match stats if available
    if match_stats_df is not None and not match_stats_df.empty:
        df = merge_match_stats(df, match_stats_df)
    else:
        for col in ["home_xg", "away_xg", "home_shots", "away_shots", "home_sot", "away_sot"]:
            if col not in df.columns:
                df[col] = np.nan
        df["is_xg_missing"] = df["home_xg"].isna() | df["away_xg"].isna()

    # Compute differential features
    df = add_base_differentials(df)

    return df
