import json
import pandas as pd
import requests

from src.ingestion.fotmob_client import FotMobClient, RetryConfig


class DummyResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def json(self):
        return self._json

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise requests.HTTPError(f"status {self.status_code}")


class DummySession:
    def __init__(self, responses):
        self._responses = responses
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        key = url.split('/')[-1]
        return self._responses.get(key, DummyResponse([], 404))


def test_fetch_fixtures_returns_dataframe():
    sample = [{"id": 1, "home_team": "A", "away_team": "B"}]
    session = DummySession({"fixtures": DummyResponse(sample)})
    client = FotMobClient(session=session, retry=RetryConfig(attempts=1))
    df = client.fetch_fixtures(season="2020", league_id=100)
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 1


def test_fetch_match_stats_returns_dataframe():
    sample = [{"stat": "shots", "home": 10, "away": 8}]
    session = DummySession({"match_stats": DummyResponse(sample)})
    client = FotMobClient(session=session, retry=RetryConfig(attempts=1))
    df = client.fetch_match_stats(fixture_id=555)
    assert isinstance(df, pd.DataFrame)
    assert "stat" in df.columns


def test_fetch_player_minutes_returns_dataframe():
    sample = [{"player_id": 11, "minutes": 90}]
    session = DummySession({"player_minutes": DummyResponse(sample)})
    client = FotMobClient(session=session, retry=RetryConfig(attempts=1))
    df = client.fetch_player_minutes(fixture_id=555)
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["minutes"] == 90
