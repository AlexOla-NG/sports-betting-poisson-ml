"""Expected-goal adjustments for player absences.

Pipeline Stage: 04_adjustments
Inputs: Poisson fixture ratings and classified player absences.
Outputs: One row per fixture with home/away adjusted expected goals.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.processing.clean_features import standardize_team_name


ATTACKING_GROUPS = frozenset({"MID", "FWD"})
DEFENSIVE_GROUPS = frozenset({"GK", "DEF"})


def _absence_streak_positions(absences: pd.DataFrame) -> pd.Series:
    """Return one-based consecutive absence positions per player.

    Parameters
    ----------
    absences : pd.DataFrame
        Absence rows containing player_id and date, optionally team.

    Returns
    -------
    pandas.Series
        Streak position aligned to ``absences.index``. Rows are ordered by
        player, team, and date; each absence row advances that player's streak.
        The persisted absence table contains only inferred missing appearances,
        so each row represents one successive missed fixture for the player.

    Pipeline stage
    --------------
    04_adjustments.
    """
    if absences.empty:
        return pd.Series(dtype="int64", index=absences.index)

    ordered = absences.copy()
    ordered["_original_index"] = ordered.index
    group_columns = ["player_id"]
    if "team" in ordered.columns:
        group_columns.append("team")
    ordered["_date"] = pd.to_datetime(ordered["date"], errors="raise")
    sort_columns = group_columns + ["_date", "fixture_id"]
    if "_fixture_sequence" in ordered.columns:
        sort_columns = group_columns + ["_fixture_sequence"]
    ordered = ordered.sort_values(sort_columns)
    if "_fixture_sequence" in ordered.columns:
        previous_sequence = ordered.groupby(group_columns, sort=False)["_fixture_sequence"].shift()
        new_streak = previous_sequence.isna() | (ordered["_fixture_sequence"] != previous_sequence + 1)
        ordered["_streak_group"] = new_streak.groupby(
            [ordered[column] for column in group_columns], sort=False
        ).cumsum()
        ordered["_streak_position"] = ordered.groupby(
            group_columns + ["_streak_group"], sort=False
        ).cumcount() + 1
    else:
        ordered["_streak_position"] = ordered.groupby(group_columns, sort=False).cumcount() + 1
    return ordered.set_index("_original_index")["_streak_position"].reindex(absences.index)


def _adjustment_factor(
    position_group: str,
    streak_position: int,
    config: dict[str, Any],
) -> float:
    """Return the config-driven multiplier for one absence row.

    Pipeline stage: 04_adjustments.
    """
    weights = config["injury"]["positional_weights"]
    decay = float(config["injury"]["injury_decay"])
    weight = float(weights[position_group])
    return max(0.0, weight - 1.0) * (decay ** (streak_position - 1))


def adjust_lambda(
    base_lambda: float,
    absences: pd.DataFrame,
    team_id: str,
    fixture_date: object,
    config: dict[str, Any],
) -> float:
    """Adjust a team's own scoring lambda for attacking-position absences.

    Parameters
    ----------
    base_lambda : float
        Baseline expected goals for the team.
    absences : pandas.DataFrame
        Classified absence rows with ``team``, ``date``, ``position_group``,
        and ``player_id`` columns.
    team_id : str
        Team whose attacking lambda is being calculated.
    fixture_date : object
        Date identifying the fixture being adjusted.
    config : dict[str, Any]
        Central configuration containing positional weights and injury decay.

    Returns
    -------
    float
        Adjusted scoring lambda. MID/FWD absences reduce this value. GK/DEF
        absences do not reduce the team's own scoring lambda; the fixture-level
        helper applies their defensive effect to the opponent's lambda.

    Pipeline stage
    --------------
    04_adjustments. ``min_absence_matches`` is intentionally deferred.
    """
    if absences.empty:
        return float(base_lambda)

    target_date = pd.to_datetime(fixture_date)
    absence_dates = pd.to_datetime(absences["date"]).dt.normalize()
    relevant = absences[
        (absences["team"] == team_id)
        & (absence_dates == target_date.normalize())
        & absences["position_group"].isin(ATTACKING_GROUPS)
    ].copy()
    if relevant.empty:
        return float(base_lambda)

    streak_positions = _absence_streak_positions(absences)
    adjusted_lambda = float(base_lambda)
    for index, row in relevant.iterrows():
        factor = _adjustment_factor(
            str(row["position_group"]),
            int(streak_positions.loc[index]),
            config,
        )
        adjusted_lambda *= 1.0 - factor
    return float(adjusted_lambda)


def adjust_fixture_lambdas(
    fixture_id: object,
    home_team: str,
    away_team: str,
    fixture_date: object,
    base_home_lambda: float,
    base_away_lambda: float,
    absences: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Adjust both fixture lambdas using attacking and defensive absences.

    Pipeline stage: 04_adjustments. FWD/MID absences reduce the affected
    team's lambda; GK/DEF absences increase the opponent's lambda.
    """
    target_date = pd.to_datetime(fixture_date)
    absence_dates = pd.to_datetime(absences["date"]).dt.normalize()
    fixture_absences = absences[
        (absences["fixture_id"].astype(str) == str(fixture_id))
        & (absence_dates == target_date.normalize())
    ].copy()
    streak_positions = _absence_streak_positions(absences)
    adjusted_home = adjust_lambda(
        base_home_lambda, absences, home_team, target_date, config
    )
    adjusted_away = adjust_lambda(
        base_away_lambda, absences, away_team, target_date, config
    )

    home_defensive_count = 0
    away_defensive_count = 0
    for index, row in fixture_absences.iterrows():
        position_group = str(row["position_group"])
        if position_group not in DEFENSIVE_GROUPS:
            continue
        factor = _adjustment_factor(
            position_group,
            int(streak_positions.loc[index]),
            config,
        )
        if row["team"] == home_team:
            adjusted_away *= 1.0 + factor
            home_defensive_count += 1
        elif row["team"] == away_team:
            adjusted_home *= 1.0 + factor
            away_defensive_count += 1

    return {
        "fixture_id": fixture_id,
        "home_team": home_team,
        "away_team": away_team,
        "base_home_lambda": float(base_home_lambda),
        "adjusted_home_lambda": float(adjusted_home),
        "base_away_lambda": float(base_away_lambda),
        "adjusted_away_lambda": float(adjusted_away),
        "num_home_absences_applied": int(
            len(fixture_absences[fixture_absences["team"] == home_team])
        ),
        "num_away_absences_applied": int(
            len(fixture_absences[fixture_absences["team"] == away_team])
        ),
    }


def _align_absences_to_ratings(
    ratings: pd.DataFrame,
    absences: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map absence fixture IDs to rating IDs and return unresolved rows.

    Pipeline stage: 04_adjustments. Understat and FBref use different fixture
    identifiers, so exact ID matches are attempted first and unresolved rows
    are reconciled by normalized date plus normalized home/away team.
    """
    if absences.empty:
        return absences.copy(), absences.copy()

    rating_lookup = ratings[["fixture_id", "date", "home_team", "away_team"]].copy()
    rating_lookup["_date"] = pd.to_datetime(rating_lookup["date"]).dt.normalize()
    rating_lookup["_home_team"] = rating_lookup["home_team"].map(standardize_team_name)
    rating_lookup["_away_team"] = rating_lookup["away_team"].map(standardize_team_name)
    rating_ids = set(rating_lookup["fixture_id"].astype(str))

    aligned = absences.copy()
    aligned["_source_fixture_id"] = aligned["fixture_id"]
    aligned["_absence_index"] = aligned.index
    aligned["_date"] = pd.to_datetime(aligned["date"], errors="raise").dt.normalize()
    aligned["_team"] = aligned["team"].map(standardize_team_name)
    aligned["fixture_id"] = aligned["fixture_id"].astype(str)
    exact_mask = aligned["fixture_id"].isin(rating_ids)

    unresolved = aligned.loc[~exact_mask].copy()
    if not unresolved.empty:
        home_matches = unresolved.merge(
            rating_lookup,
            left_on=["_date", "_team"],
            right_on=["_date", "_home_team"],
            how="left",
            suffixes=("", "_rating"),
        )
        away_matches = unresolved.merge(
            rating_lookup,
            left_on=["_date", "_team"],
            right_on=["_date", "_away_team"],
            how="left",
            suffixes=("", "_rating"),
        )
        reconciled = pd.concat([home_matches, away_matches], ignore_index=True)
        reconciled = reconciled.dropna(subset=["fixture_id_rating"])
        reconciled = reconciled.drop_duplicates(subset=["_absence_index"])
        if not reconciled.empty:
            aligned.loc[reconciled["_absence_index"], "fixture_id"] = (
                reconciled.set_index("_absence_index")["fixture_id_rating"].astype(str)
            )

    matched_mask = aligned["fixture_id"].isin(rating_ids)
    unmatched = aligned.loc[~matched_mask].copy()
    aligned.loc[matched_mask, "team"] = aligned.loc[matched_mask, "_team"]
    return aligned.loc[matched_mask].drop(columns=["_source_fixture_id", "_date", "_team", "_absence_index"]), unmatched


def build_adjusted_lambdas(
    ratings: pd.DataFrame,
    absences: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build one adjusted lambda row per rated fixture.

    Pipeline stage: 04_adjustments. Fixtures in the ratings table are retained;
    absence rows with no matching fixture are reported in the returned attrs.
    """
    required_ratings = {
        "fixture_id", "date", "home_team", "away_team",
        "expected_home_goals", "expected_away_goals",
    }
    missing = required_ratings.difference(ratings.columns)
    if missing:
        raise ValueError(f"Ratings data missing columns: {sorted(missing)}")

    aligned_absences, unmatched_absences = _align_absences_to_ratings(ratings, absences)
    if not aligned_absences.empty:
        fixture_order = []
        for team_column in ("home_team", "away_team"):
            team_ratings = ratings[["fixture_id", "date", team_column]].rename(
                columns={team_column: "team"}
            )
            team_ratings["team"] = team_ratings["team"].map(standardize_team_name)
            team_ratings = team_ratings.sort_values(["team", "date", "fixture_id"])
            team_ratings["_fixture_sequence"] = team_ratings.groupby("team", sort=False).cumcount()
            fixture_order.append(team_ratings[["fixture_id", "team", "_fixture_sequence"]])
        fixture_order_df = pd.concat(fixture_order, ignore_index=True).drop_duplicates(
            subset=["fixture_id", "team"]
        )
        aligned_absences = aligned_absences.merge(
            fixture_order_df,
            on=["fixture_id", "team"],
            how="left",
            validate="many_to_one",
        )
    rows = [
        adjust_fixture_lambdas(
            row.fixture_id,
            row.home_team,
            row.away_team,
            row.date,
            row.expected_home_goals,
            row.expected_away_goals,
            aligned_absences,
            config,
        )
        for row in ratings.itertuples(index=False)
    ]
    output = pd.DataFrame(rows)
    output.attrs["unmatched_absences"] = unmatched_absences
    return output
