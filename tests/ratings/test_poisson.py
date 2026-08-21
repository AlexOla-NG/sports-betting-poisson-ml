import numpy as np
import pandas as pd
import pytest

from src.ratings.poisson import (
    compute_rolling_ratings,
    fit_poisson_ratings,
    prepare_glm_dataset,
)


def test_prepare_glm_dataset_duplicates_rows_for_home_and_away():
    fixtures = pd.DataFrame(
        {
            "home_team": ["Arsenal"],
            "away_team": ["Chelsea"],
            "home_goals": [2],
            "away_goals": [1],
        }
    )
    glm_df = prepare_glm_dataset(fixtures)

    assert len(glm_df) == 2
    assert glm_df.loc[0, "team"] == "Arsenal"
    assert glm_df.loc[0, "opponent"] == "Chelsea"
    assert glm_df.loc[0, "home"] == 1
    assert glm_df.loc[0, "goals"] == 2

    assert glm_df.loc[1, "team"] == "Chelsea"
    assert glm_df.loc[1, "opponent"] == "Arsenal"
    assert glm_df.loc[1, "home"] == 0
    assert glm_df.loc[1, "goals"] == 1


def test_fit_poisson_ratings_returns_expected_structure():
    fixtures = pd.DataFrame(
        {
            "home_team": ["Arsenal", "Chelsea", "Liverpool", "Arsenal"],
            "away_team": ["Chelsea", "Liverpool", "Arsenal", "Liverpool"],
            "home_goals": [2, 1, 0, 3],
            "away_goals": [0, 1, 2, 1],
        }
    )

    att_map, def_map, home_adv = fit_poisson_ratings(fixtures)

    assert isinstance(att_map, dict)
    assert isinstance(def_map, dict)
    assert "Arsenal" in att_map
    assert "Chelsea" in att_map
    assert home_adv > 0.0


def test_compute_rolling_ratings_enforces_shift_one_no_leakage():
    # Fixtures where Arsenal scores 5 goals in match 2
    fixtures = pd.DataFrame(
        {
            "fixture_id": ["f1", "f2", "f3"],
            "date": ["2024-08-16", "2024-08-20", "2024-08-25"],
            "home_team": ["Arsenal", "Arsenal", "Chelsea"],
            "away_team": ["Wolves", "Chelsea", "Arsenal"],
            "home_goals": [1, 5, 0],
            "away_goals": [0, 0, 1],
        }
    )

    ratings_df = compute_rolling_ratings(fixtures, window=2)

    # Match 1 (f1): cold start, no past matches
    assert ratings_df.loc[0, "home_attack_rating"] == 1.0

    # Match 2 (f2): uses only f1 (Arsenal 1 - 0 Wolves)
    att_before_f2 = ratings_df.loc[1, "home_attack_rating"]

    # Match 3 (f3): uses f1 and f2 (after Arsenal scored 5 goals)
    att_before_f3 = ratings_df.loc[2, "away_attack_rating"]

    # The 5-0 win in f2 should affect f3's pre-match rating, but NOT f2's pre-match rating
    assert ratings_df.loc[1, "fixture_id"] == "f2"
    assert ratings_df.loc[2, "fixture_id"] == "f3"
    assert att_before_f3 > att_before_f2
