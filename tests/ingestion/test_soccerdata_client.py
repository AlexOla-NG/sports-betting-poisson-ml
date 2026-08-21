from datetime import datetime

import pandas as pd
import pytest

from src.ingestion.soccerdata_client import SoccerDataClient, SoccerDataConfig


class DummyFBref:
    def __init__(self, schedule, team_stats=None, player_stats=None):
        self.schedule = schedule
        self.team_stats = team_stats if team_stats is not None else pd.DataFrame()
        self.player_stats = player_stats if player_stats is not None else pd.DataFrame()
        self.schedule_calls = 0
        self.team_stats_calls = 0
        self.player_stats_calls = 0

    def read_schedule(self, force_cache=False):
        self.schedule_calls += 1
        return self.schedule

    def read_team_match_stats(self, stat_type="schedule"):
        self.team_stats_calls += 1
        return self.team_stats

    def read_player_match_stats(self, stat_type="summary", match_id=None):
        self.player_stats_calls += 1
        return self.player_stats


class DummyClubElo:
    def __init__(self, ratings):
        self.ratings = ratings
        self.calls = []

    def read_by_date(self, date=None):
        self.calls.append(date)
        return self.ratings


def test_season_name_accepts_slash_and_hyphen():
    assert SoccerDataClient._season_name("2024/2025") == "2024-2025"
    assert SoccerDataClient._season_name("2024-2025") == "2024-2025"


def test_season_name_rejects_ambiguous_format():
    with pytest.raises(ValueError):
        SoccerDataClient._season_name("2024")


def test_fetch_fixtures_normalizes_schedule_and_scores():
    schedule = pd.DataFrame(
        {
            "game_id": ["abc123"],
            "date": ["2024-08-16"],
            "home_team": ["Arsenal"],
            "score": ["2–1"],
            "away_team": ["Wolves"],
        }
    )
    fbref = DummyFBref(schedule)
    client = SoccerDataClient(fbref=fbref)

    fixtures = client.fetch_fixtures("2024/2025", league_id=47)

    assert fixtures.loc[0, "fixture_id"] == "abc123"
    assert fixtures.loc[0, "home_goals"] == 2
    assert fixtures.loc[0, "away_goals"] == 1
    assert fbref.schedule_calls == 1


def test_fetch_match_stats_filters_fixture():
    stats = pd.DataFrame(
        {"team": ["Arsenal", "Wolves", "Other"], "shots": [12, 5, 3]},
        index=pd.MultiIndex.from_tuples(
            [
                ("ENG-Premier League", "2425", "Arsenal", "abc123"),
                ("ENG-Premier League", "2425", "Wolves", "abc123"),
                ("ENG-Premier League", "2425", "Other", "other"),
            ],
            names=["league", "season", "team", "game"],
        ),
    )
    fbref = DummyFBref(pd.DataFrame(), team_stats=stats)
    client = SoccerDataClient(fbref=fbref)

    result = client.fetch_match_stats("abc123")

    assert result["fixture_id"].tolist() == ["abc123", "abc123"]
    assert fbref.team_stats_calls == 1


def test_fetch_player_minutes_reads_match_summary():
    players = pd.DataFrame({"player": ["Player A"], "minutes": [90]})
    fbref = DummyFBref(pd.DataFrame(), player_stats=players)
    client = SoccerDataClient(fbref=fbref)

    result = client.fetch_player_minutes("abc123")

    assert result.loc[0, "minutes"] == 90
    assert fbref.player_stats_calls == 1


def test_fetch_elo_uses_clubelo_reader():
    ratings = pd.DataFrame({"team": ["Arsenal"], "elo": [1800]})
    clubelo = DummyClubElo(ratings)
    client = SoccerDataClient(clubelo=clubelo)

    result = client.fetch_elo(date="2024-08-16")

    assert result.loc[0, "elo"] == 1800
    assert clubelo.calls == ["2024-08-16"]


def test_save_parquet_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "fixtures.parquet"
    SoccerDataClient.save_parquet(pd.DataFrame({"fixture_id": [1]}), str(path))

    assert path.exists()
    assert pd.read_parquet(path).loc[0, "fixture_id"] == 1


def test_sanitize_match_stats_coerces_numeric_object_columns():
    df = pd.DataFrame(
        {
            "team": ["Arsenal", "Chelsea", "Liverpool"],
            "GF": [2, "1", ""],
            "GA": ["0", 3, None],
            "xG": [1.5, "0.8", "2.1"],
            "notes": ["Postponed", "", "Fine"],
        }
    )

    sanitized = SoccerDataClient._sanitize_match_stats(df)

    assert pd.api.types.is_integer_dtype(sanitized["GF"])
    assert pd.api.types.is_integer_dtype(sanitized["GA"])
    assert pd.api.types.is_float_dtype(sanitized["xG"])
    assert sanitized["team"].dtype == object or pd.api.types.is_string_dtype(sanitized["team"])
    assert sanitized["notes"].dtype == object or pd.api.types.is_string_dtype(sanitized["notes"])
    assert sanitized.loc[0, "GF"] == 2
    assert sanitized.loc[1, "GF"] == 1
    assert pd.isna(sanitized.loc[2, "GF"])


def test_fetch_all_match_stats_sanitizes_full_season_table():
    raw_stats = pd.DataFrame(
        {
            "team": ["Arsenal"],
            "GF": ["2"],
            "GA": [1],
        }
    )
    fbref = DummyFBref(pd.DataFrame(), team_stats=raw_stats)
    client = SoccerDataClient(fbref=fbref)

    result = client.fetch_all_match_stats("2024/2025")

    assert pd.api.types.is_integer_dtype(result["GF"])
    assert result.loc[0, "GF"] == 2
    assert fbref.team_stats_calls == 1

