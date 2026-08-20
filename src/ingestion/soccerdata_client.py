"""soccerdata ingestion adapter for the EPL pipeline.

This module delegates retrieval and caching to soccerdata's FBref and ClubElo
readers. It keeps the ingestion-stage method names stable for the notebook and
downstream Parquet outputs while avoiding a custom website scraper.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Protocol

import pandas as pd
import soccerdata as sd


class ScheduleReader(Protocol):
    """Protocol for the subset of soccerdata FBref used by this adapter."""

    def read_schedule(self, force_cache: bool = False) -> pd.DataFrame:
        """Return fixture schedule records."""


class EloReader(Protocol):
    """Protocol for the subset of soccerdata ClubElo used by this adapter."""

    def read_by_date(self, date: str | datetime | None = None) -> pd.DataFrame:
        """Return team Elo ratings for a date."""


@dataclass(frozen=True)
class SoccerDataConfig:
    """Configuration for the soccerdata ingestion stage."""

    league: str = "ENG-Premier League"
    no_cache: bool = False
    no_store: bool = False
    headless: bool = True


class SoccerDataClient:
    """Thin FBref/ClubElo adapter for pipeline ingestion.

    The ingestion stage accepts an optional numeric ``league_id`` for notebook
    compatibility, but source selection uses the configured FBref league name.
    """

    def __init__(
        self,
        config: SoccerDataConfig | None = None,
        fbref: ScheduleReader | None = None,
        clubelo: EloReader | None = None,
    ) -> None:
        self.config = config or SoccerDataConfig()
        self._fbref = fbref
        self._clubelo = clubelo

    @staticmethod
    def _season_name(season: str) -> str:
        """Convert a season label to soccerdata's ``YYYY-YYYY`` format."""
        match = re.fullmatch(r"(\d{4})[/-](\d{4})", season)
        if not match:
            raise ValueError("season must use YYYY/YYYY or YYYY-YYYY format")
        return f"{match.group(1)}-{match.group(2)}"

    def _reader(self, season: str) -> ScheduleReader:
        if self._fbref is None:
            self._fbref = sd.FBref(
                leagues=self.config.league,
                seasons=self._season_name(season),
                no_cache=self.config.no_cache,
                no_store=self.config.no_store,
                headless=self.config.headless,
            )
        return self._fbref

    @staticmethod
    def _normalize_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
        """Normalize FBref schedule columns for the raw fixtures table."""
        if isinstance(schedule.index, pd.MultiIndex):
            frame = schedule.reset_index()
        else:
            frame = schedule.reset_index(drop=True)
        frame = frame.rename(columns={"game_id": "fixture_id"})
        if "fixture_id" not in frame.columns and "match_id" in frame.columns:
            frame = frame.rename(columns={"match_id": "fixture_id"})
        if "score" in frame.columns:
            scores = frame["score"].astype("string").str.extract(
                r"(?P<home_goals>\d+)\D+(?P<away_goals>\d+)"
            )
            frame = pd.concat([frame, scores.astype("Float64")], axis=1)
        return frame

    def fetch_fixtures(self, season: str, league_id: int | None = None) -> pd.DataFrame:
        """Fetch and normalize EPL fixtures from FBref.

        Parameters
        ----------
        season : str
            Season label in ``YYYY/YYYY`` or ``YYYY-YYYY`` format.
        league_id : int | None
            Retained for notebook compatibility; FBref uses the configured
            league name instead of a provider-specific numeric ID.

        Returns
        -------
        pandas.DataFrame
            One row per fixture with FBref identifiers and result fields.
        """
        reader = self._reader(season)
        return self._normalize_schedule(reader.read_schedule())

    def fetch_match_stats(self, fixture_id: str) -> pd.DataFrame:
        """Fetch team match statistics for one FBref game ID."""
        if self._fbref is None:
            raise RuntimeError("Call fetch_fixtures() before fetch_match_stats().")
        stats = self._fbref.read_team_match_stats(stat_type="schedule")
        if "match_report" in stats.columns:
            mask = stats["match_report"].astype("string").str.contains(
                str(fixture_id), regex=False, na=False
            )
            stats = stats[mask].copy()
        elif isinstance(stats.index, pd.MultiIndex) and "game" in stats.index.names:
            mask = stats.index.get_level_values("game").astype(str) == str(fixture_id)
            stats = stats[mask].copy()
        elif "game_id" in stats.columns:
            stats = stats[stats["game_id"].astype(str) == str(fixture_id)]
        stats.insert(0, "fixture_id", str(fixture_id))
        return stats.reset_index(drop=True)

    def fetch_player_minutes(self, fixture_id: str) -> pd.DataFrame:
        """Fetch player match statistics for one FBref game ID."""
        if self._fbref is None:
            raise RuntimeError("Call fetch_fixtures() before fetch_player_minutes().")
        players = self._fbref.read_player_match_stats(
            stat_type="summary",
            match_id=str(fixture_id),
        )
        return players.reset_index(drop=True)

    def fetch_elo(self, date: str | datetime | None = None) -> pd.DataFrame:
        """Fetch ClubElo ratings for an optional date.

        Elo is an additional feature for later ML stages; it does not replace
        the pipeline's Poisson attack and defense ratings.
        """
        if self._clubelo is None:
            self._clubelo = sd.ClubElo(
                no_cache=self.config.no_cache,
                no_store=self.config.no_store,
            )
        return self._clubelo.read_by_date(date=date).reset_index(drop=True)

    @staticmethod
    def save_parquet(df: pd.DataFrame, path: str) -> None:
        """Save a DataFrame to Parquet, creating parent directories as needed."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

