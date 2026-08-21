import numpy as np
import pandas as pd
import pytest

from src.adjustments.absence_classifier import (
    classify_consecutive_absence,
    identify_player_absences,
    map_fbref_position,
)


def test_map_fbref_position_standardizes_position_tags():
    assert map_fbref_position("GK") == "GK"
    assert map_fbref_position("DF") == "DEF"
    assert map_fbref_position("DF,MF") == "DEF"
    assert map_fbref_position("MF,FW") == "MID"
    assert map_fbref_position("FW") == "FWD"
    assert map_fbref_position(None) == "MID"


def test_classify_consecutive_absence_distinguishes_injury_and_rotation():
    assert classify_consecutive_absence(1) == "rotation_suspension"
    assert classify_consecutive_absence(2) == "injury"
    assert classify_consecutive_absence(5) == "injury"


def test_identify_player_absences_flags_key_starters_without_leakage():
    # Player A is a key starter (plays 90m in matches 1-5, then misses match 6 and match 7)
    dates = pd.date_range("2024-08-01", periods=7, freq="7D")
    logs = pd.DataFrame(
        {
            "fixture_id": [f"f{i+1}" for i in range(7)],
            "date": dates,
            "team": ["Arsenal"] * 7,
            "player_id": ["p1"] * 7,
            "player_name": ["Bukayo Saka"] * 7,
            "position": ["FW"] * 7,
            "minutes_played": [90, 90, 90, 90, 90, 0, 0],
        }
    )

    absences = identify_player_absences(logs, key_starter_threshold=0.45, rolling_window=5)

    assert len(absences) == 2
    # Match 6 (first missed game): rotation_suspension
    assert absences.loc[0, "fixture_id"] == "f6"
    assert absences.loc[0, "absence_type"] == "rotation_suspension"
    assert absences.loc[0, "position_group"] == "FWD"

    # Match 7 (second missed game): injury
    assert absences.loc[1, "fixture_id"] == "f7"
    assert absences.loc[1, "absence_type"] == "injury"
