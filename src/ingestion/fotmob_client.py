"""FotMob ingestion client.

Provides FotMobClient with methods to fetch fixtures, match stats, and player minutes.

This module uses requests with a simple retry/backoff strategy and saves raw Parquet
files under `data/raw/{season}/`.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict

import pandas as pd
import requests


@dataclass
class RetryConfig:
    attempts: int = 3
    backoff_factor: float = 1.0


class FotMobClient:
    """Minimal FotMob client wrapper.

    Notes:
    - Methods return pandas.DataFrame objects built from JSON responses.
    - Network failures are retried with exponential backoff and logged.
    - No assumptions are made about season string or league id formatting.
    """

    def __init__(self, base_url: str = "https://api.fotmob.com", session: requests.Session | None = None,
                 retry: RetryConfig | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.retry = retry or RetryConfig()

    def _get_json(self, path: str, params: Dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/') }"
        last_exc = None
        for attempt in range(1, self.retry.attempts + 1):
            try:
                resp = self.session.get(url, params=params, timeout=10)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # keep broad to ensure robustness in CI tests
                last_exc = exc
                sleep = self.retry.backoff_factor * (2 ** (attempt - 1))
                time.sleep(sleep)
        raise last_exc

    def fetch_fixtures(self, season: str, league_id: int) -> pd.DataFrame:
        """Fetch fixtures for a season/league.

        Parameters
        - season: arbitrary season identifier (e.g., '2020/2021')
        - league_id: provider league id (no assumptions made)

        Returns a DataFrame with the raw fixtures data.
        """
        data = self._get_json(f"leagues/{league_id}/seasons/{season}/fixtures")
        df = pd.json_normalize(data)
        return df

    def fetch_match_stats(self, fixture_id: int) -> pd.DataFrame:
        """Fetch match-level statistics for a fixture id.

        Returns a DataFrame (rows = stat records).
        """
        data = self._get_json(f"fixtures/{fixture_id}/match_stats")
        df = pd.json_normalize(data)
        return df

    def fetch_player_minutes(self, fixture_id: int) -> pd.DataFrame:
        """Fetch player minutes / participation for a fixture.

        Returns a DataFrame with one row per player / participation record.
        """
        data = self._get_json(f"fixtures/{fixture_id}/player_minutes")
        df = pd.json_normalize(data)
        return df

    def save_parquet(self, df: pd.DataFrame, path: str) -> None:
        """Save DataFrame to Parquet, creating parent dirs as needed."""
        df.to_parquet(path, index=False)
