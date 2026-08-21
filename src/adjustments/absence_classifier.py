"""Player absence classification module.

Pipeline Stage: 04_adjustments
Inputs: Player match logs DataFrame
Outputs: Classified player absences DataFrame (data/processed/absences.parquet)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


POSITION_MAP: dict[str, str] = {
    "GK": "GK",
    "DF": "DEF",
    "CB": "DEF",
    "LB": "DEF",
    "RB": "DEF",
    "WB": "DEF",
    "MF": "MID",
    "DM": "MID",
    "CM": "MID",
    "AM": "MID",
    "FW": "FWD",
    "ST": "FWD",
    "LW": "FWD",
    "RW": "FWD",
}


def map_fbref_position(pos_str: str | None) -> str:
    """Map raw FBref position string to standard 4-group category (GK, DEF, MID, FWD).

    Parameters
    ----------
    pos_str : str | None
        Raw FBref position (e.g. "GK", "DF", "DF,MF", "FW,MF").

    Returns
    -------
    str
        Standardized position group ("GK", "DEF", "MID", "FWD").
    """
    if pos_str is None or pd.isna(pos_str):
        return "MID"

    primary_pos = str(pos_str).split(",")[0].strip().upper()
    return POSITION_MAP.get(primary_pos, "MID")


def classify_consecutive_absence(consecutive_missed: int) -> str:
    """Classify absence type based on consecutive missed matches count.

    Parameters
    ----------
    consecutive_missed : int
        Number of consecutive matches missed by the player.

    Returns
    -------
    str
        "injury" if consecutive_missed >= 2 else "rotation_suspension".
    """
    if consecutive_missed >= 2:
        return "injury"
    return "rotation_suspension"


def identify_player_absences(
    player_logs_df: pd.DataFrame,
    key_starter_threshold: float = 0.45,
    rolling_window: int = 5,
) -> pd.DataFrame:
    """Identify and classify absences for key starters across match fixtures.

    Parameters
    ----------
    player_logs_df : pd.DataFrame
        Player match logs containing player_id, player_name, team, date, minutes_played, position.
    key_starter_threshold : float
        Minimum rolling minute share (0.0 to 1.0) in past matches to qualify as key starter. Default 0.45.
    rolling_window : int
        Number of past matches to include in rolling minute share calculation. Default 5.

    Returns
    -------
    pd.DataFrame
        Table of classified absences with columns: fixture_id, date, team, player_id,
        player_name, position_group, absence_type, starter_minute_share.
    """
    df = player_logs_df.copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "date",
                "team",
                "player_id",
                "player_name",
                "position_group",
                "absence_type",
                "starter_minute_share",
            ]
        )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["player_id", "date"]).reset_index(drop=True)

    # Standardize position
    if "position" in df.columns:
        df["position_group"] = df["position"].apply(map_fbref_position)
    else:
        df["position_group"] = "MID"

    # Compute rolling past minutes played per player using .shift(1) to avoid data leakage
    df["rolling_minutes"] = (
        df.groupby("player_id")["minutes_played"]
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=1).mean().shift(1))
        .fillna(0.0)
    )
    df["starter_minute_share"] = df["rolling_minutes"] / 90.0

    # Track consecutive missed matches per player
    df["is_zero_minutes"] = (df["minutes_played"] == 0) | df["minutes_played"].isna()

    consecutive_counts = []
    current_count = 0
    prev_player = None

    for idx, row in df.iterrows():
        player = row["player_id"]
        is_zero = row["is_zero_minutes"]

        if player != prev_player:
            current_count = 1 if is_zero else 0
            prev_player = player
        else:
            if is_zero:
                current_count += 1
            else:
                current_count = 0

        consecutive_counts.append(current_count)

    df["consecutive_missed"] = consecutive_counts

    # Filter key starter absences (starter_minute_share >= threshold AND minutes_played == 0)
    absence_mask = (df["starter_minute_share"] >= key_starter_threshold) & df["is_zero_minutes"]
    absences = df[absence_mask].copy()

    if absences.empty:
        return pd.DataFrame(
            columns=[
                "fixture_id",
                "date",
                "team",
                "player_id",
                "player_name",
                "position_group",
                "absence_type",
                "starter_minute_share",
            ]
        )

    absences["absence_type"] = absences["consecutive_missed"].apply(classify_consecutive_absence)

    output_cols = [
        col
        for col in [
            "fixture_id",
            "date",
            "team",
            "player_id",
            "player_name",
            "position_group",
            "absence_type",
            "starter_minute_share",
        ]
        if col in absences.columns
    ]

    return absences[output_cols].reset_index(drop=True)
