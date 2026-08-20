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


    def test_fetch_fixtures_with_retry():
        """Test retry logic on transient failures (retries after initial 500)."""
        attempts_made = [0]
    
        class CountingSession:
            def __init__(self):
                self.calls = []
        
            def get(self, url, params=None, timeout=None):
                self.calls.append((url, params))
                attempts_made[0] += 1
                # Fail first attempt, succeed second
                if attempts_made[0] == 1:
                    return DummyResponse({}, 500)
                return DummyResponse([{"id": 1, "home": "A", "away": "B"}], 200)
    
        session = CountingSession()
        client = FotMobClient(session=session, retry=RetryConfig(attempts=2, backoff_factor=0))
    
        # With retry, should succeed on second attempt
        df = client.fetch_fixtures(season="2020", league_id=100)
        assert attempts_made[0] == 2
        assert isinstance(df, pd.DataFrame)


    def test_fetch_fixtures_backoff_timing():
        """Test exponential backoff with timing."""
        import time
    
        attempts = [0]
    
        class SlowSession:
            def get(self, url, params=None, timeout=None):
                attempts[0] += 1
                if attempts[0] <= 2:
                    return DummyResponse({}, 500)
                return DummyResponse([{"id": 1}], 200)
    
        session = SlowSession()
        client = FotMobClient(session=session, retry=RetryConfig(attempts=3, backoff_factor=0.01))
    
        start = time.time()
        try:
            df = client.fetch_fixtures(season="2020", league_id=100)
        except:
            pass
        elapsed = time.time() - start
    
        # Should have waited ~0.01 + 0.02 = 0.03s (exponential backoff)
        assert elapsed >= 0.01, "Expected backoff sleep time"
        assert attempts[0] >= 2, "Expected at least 2 retry attempts"


    def test_fetch_fixtures_invalid_season_returns_empty():
        """Test graceful handling of invalid season."""
        session = DummySession({"fixtures": DummyResponse([])})
        client = FotMobClient(session=session, retry=RetryConfig(attempts=1))
    
        df = client.fetch_fixtures(season="9999/9999", league_id=100)
        assert isinstance(df, pd.DataFrame)
        # Should return empty DataFrame, not raise exception
        assert len(df) == 0


    def test_fetch_match_stats_multiple_fixtures():
        """Test fetching stats from multiple fixtures."""
        sample = [
            {"fixture_id": 1, "stat": "shots", "home": 10, "away": 8},
            {"fixture_id": 2, "stat": "shots", "home": 5, "away": 7},
        ]
        session = DummySession({"match_stats": DummyResponse(sample)})
        client = FotMobClient(session=session, retry=RetryConfig(attempts=1))
    
        df = client.fetch_match_stats(fixture_id=1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    assert isinstance(df, pd.DataFrame)
    assert df.iloc[0]["minutes"] == 90
