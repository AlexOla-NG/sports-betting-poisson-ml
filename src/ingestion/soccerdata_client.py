"""soccerdata ingestion adapter for the EPL pipeline.

This module delegates retrieval and caching to soccerdata's FBref, ClubElo, and
Understat readers. It keeps the ingestion-stage method names stable for the
notebook and downstream Parquet outputs while avoiding custom website scrapers.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Protocol

import pandas as pd
import soccerdata as sd

from src.config.loader import load_config


class ScheduleReader(Protocol):
    """Protocol for the subset of soccerdata FBref used by this adapter."""

    def read_schedule(self, force_cache: bool = False) -> pd.DataFrame:
        """Return fixture schedule records."""


class EloReader(Protocol):
    """Protocol for the subset of soccerdata ClubElo used by this adapter."""

    def read_by_date(self, date: str | datetime | None = None) -> pd.DataFrame:
        """Return team Elo ratings for a date."""


class UnderstatReader(Protocol):
    """Protocol for the Understat reader used by this adapter."""

    def read_player_match_stats(self, match_id: int | list[int] | None = None) -> pd.DataFrame:
        """Return player match statistics for selected leagues and seasons."""

    def read_schedule(self, include_matches_without_data: bool = True) -> pd.DataFrame:
        """Return the Understat schedule for selected leagues and seasons."""


@dataclass(frozen=True)
class SoccerDataConfig:
    """Configuration for the soccerdata ingestion stage."""

    league: str = "ENG-Premier League"
    seasons: tuple[str, ...] | None = None
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
        understat: UnderstatReader | None = None,
    ) -> None:
        self.config = config or SoccerDataConfig()
        self._fbref = fbref
        self._clubelo = clubelo
        self._understat = understat

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

    def _understat_seasons(self) -> tuple[str, ...]:
        """Resolve Understat seasons from the central repository config."""
        if self.config.seasons is not None:
            return tuple(self.config.seasons)

        config = load_config()
        return tuple(config["data"]["seasons"])

    def _understat_reader(self) -> UnderstatReader:
        """Create or return the configured Understat reader."""
        if self._understat is None:
            self._understat = sd.Understat(
                leagues=self.config.league,
                seasons=self._understat_seasons(),
                no_cache=self.config.no_cache,
                no_store=self.config.no_store,
            )
        return self._understat

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

    @staticmethod
    def _sanitize_match_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Coerce numeric-looking object columns to Int64 or Float64.

        FBref match stats tables often contain mixed types or empty strings/placeholders
        in numeric columns (like GF, GA, Sh, SoT), causing PyArrow schema errors
        when saving to Parquet.
        """
        df = df.copy()
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                non_empty_originals = (
                    df[col].astype(str).str.strip().replace(["", "nan", "None", "<NA>"], pd.NA).dropna()
                )
                if len(non_empty_originals) > 0:
                    numeric_series = pd.to_numeric(df[col], errors="coerce")
                    converted_non_nulls = numeric_series.loc[non_empty_originals.index].dropna()
                    if len(converted_non_nulls) / len(non_empty_originals) >= 0.8:
                        is_int = (numeric_series.dropna() % 1 == 0).all()
                        if is_int:
                            df[col] = numeric_series.astype("Int64")
                        else:
                            df[col] = numeric_series.astype("Float64")
        return df

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

    def fetch_all_match_stats(self, season: str) -> pd.DataFrame:
        """Fetch and sanitize team match statistics for an entire season.

        Parameters
        ----------
        season : str
            Season label in ``YYYY/YYYY`` or ``YYYY-YYYY`` format.

        Returns
        -------
        pandas.DataFrame
            Team match statistics with normalized numeric data types.
        """
        reader = self._reader(season)
        stats = reader.read_team_match_stats(stat_type="schedule").reset_index()
        return self._sanitize_match_stats(stats)

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
        return self._sanitize_match_stats(stats.reset_index(drop=True))

    def fetch_player_minutes(self, fixture_id: str) -> pd.DataFrame:
        """Fetch player match statistics for one FBref game ID."""
        if self._fbref is None:
            raise RuntimeError("Call fetch_fixtures() before fetch_player_minutes().")
        players = self._fbref.read_player_match_stats(
            stat_type="summary",
            match_id=str(fixture_id),
        )
        return players.reset_index(drop=True)

    @staticmethod
    def _normalize_understat_position(position: object) -> str:
        """Translate Understat position labels to FBref-compatible labels.

        Understat uses positional labels such as ``DMC`` and ``FWR`` while
        the absence classifier accepts FBref-style groups such as ``DM`` and
        ``FW``. ``Sub`` is retained because it describes an appearance role,
        not a playing position; missing expected starters inherit their last
        known non-``Sub`` position in the adjustments stage.
        """
        if pd.isna(position):
            return "MID"
        understat_position = str(position).strip().upper()
        position_map = {
            "GK": "GK",
            "DL": "DF",
            "DC": "DF",
            "DR": "DF",
            "DMC": "DM",
            "DML": "DM",
            "DMR": "DM",
            "MC": "MF",
            "ML": "MF",
            "MR": "MF",
            "AMC": "AM",
            "AML": "AM",
            "AMR": "AM",
            "FW": "FW",
            "FWL": "FW",
            "FWR": "FW",
            "SUB": "SUB",
        }
        return position_map.get(understat_position, "MID")

    @staticmethod
    def _normalize_understat_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
        """Extract Understat fixture IDs and dates from columns or index."""
        frame = schedule.reset_index() if isinstance(schedule.index, pd.MultiIndex) else schedule.reset_index(drop=True)
        if "fixture_id" not in frame.columns and "game_id" in frame.columns:
            frame = frame.rename(columns={"game_id": "fixture_id"})
        if "fixture_id" not in frame.columns:
            raise ValueError("Understat schedule must contain game_id")

        date_column = next(
            (column for column in ("date", "match_date", "game") if column in frame.columns),
            None,
        )
        if date_column is None:
            raise ValueError("Understat schedule must contain a match date")
        return frame[["fixture_id", date_column]].rename(columns={date_column: "date"})

    def fetch_player_match_stats(self) -> pd.DataFrame:
        """Fetch and normalize all Understat player appearances.

        Returns
        -------
        pandas.DataFrame
            Player appearance rows with exactly ``fixture_id``, ``date``,
            ``team``, ``player_id``, ``player_name``, ``position``, and
            ``minutes_played``. This is an ingestion-stage table; expected
            missing starter rows are derived later from pre-fixture history.
        """
        output_columns = [
            "fixture_id",
            "date",
            "team",
            "player_id",
            "player_name",
            "position",
            "minutes_played",
        ]
        reader = self._understat_reader()
        stats = reader.read_player_match_stats()
        if stats.empty:
            return pd.DataFrame(columns=output_columns)

        frame = stats.reset_index()
        rename_map = {
            "game_id": "fixture_id",
            "player": "player_name",
            "minutes": "minutes_played",
        }
        frame = frame.rename(columns=rename_map)
        required_columns = {"fixture_id", "team", "player_id", "player_name", "position", "minutes_played"}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"Understat player stats missing columns: {sorted(missing_columns)}")

        schedule = self._normalize_understat_schedule(reader.read_schedule(include_matches_without_data=False))
        frame = frame.merge(schedule, on="fixture_id", how="left", validate="many_to_one")
        frame["position"] = frame["position"].map(self._normalize_understat_position)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["minutes_played"] = pd.to_numeric(frame["minutes_played"], errors="coerce")
        frame = frame[output_columns].dropna(subset=["fixture_id", "date", "team", "player_id"])
        return frame.sort_values(["date", "fixture_id", "team", "player_id"]).reset_index(drop=True)

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

