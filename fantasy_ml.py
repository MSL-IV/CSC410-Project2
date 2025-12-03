"""
- Positions: QB / RB / WR / TE (choose with --position)
- Season: configurable via --season (default: 2025)
- Scoring: PPR/half/standard via --scoring
- Features emphasize recent form (last 3–5 games) with a softer season-long average
- Includes defense-vs-position matchup features based on which defense
  the player is facing in the target week.
- Output: Top projected players for the upcoming week (latest completed week + 1)
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import nflreadpy as nfl
import polars as pl
import pandas as pd
from requests import exceptions as req_exc
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

try:
    import pyarrow  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    pyarrow = None


# ------------------------- helpers to resolve columns -------------------------


def _resolve_column(available: set[str], options: Tuple[str, ...], required: bool) -> str | None:
    """
    Return the first column from `options` that is present in `available`.
    If none are found and `required` is True, raise a ValueError.
    """
    for cand in options:
        if cand in available:
            return cand
    if required:
        raise ValueError(
            f"Required column not found. Tried {options}. "
            f"Available columns: {sorted(available)}"
        )
    return None


def _resolve_position_column(df: pl.DataFrame) -> str:
    """Detect a usable position column (position vs position_group vs pos)."""
    available = set(df.columns)
    for cand in ("position", "position_group", "pos"):
        if cand in available:
            return cand
    raise ValueError(f"No position-like column found. Available columns: {sorted(available)}")


def _resolve_fantasy_column(available: set[str], scoring: str) -> str:
    """Pick a fantasy points column based on scoring type."""
    scoring = scoring.lower()
    if scoring == "ppr":
        options = ("fantasy_points_ppr", "fantasy_points")
    elif scoring in ("half", "half_ppr", "0.5ppr"):
        options = ("fantasy_points_half_ppr", "fantasy_points_hppr", "fantasy_points")
    elif scoring in ("standard", "std", "non_ppr"):
        options = ("fantasy_points", "fantasy_points_std", "fantasy_points_ppr")
    else:
        options = ("fantasy_points", "fantasy_points_ppr")

    col = _resolve_column(available, options, required=True)
    assert col is not None
    return col


# ----------------------- load position stats for a season --------------------

POSITION_CHOICES = ("QB", "RB", "WR", "TE")

def load_position_stats(season: int, position: str, scoring: str) -> Tuple[pl.DataFrame, str]:
    """Load stats for a position and resolve the fantasy points column."""
    stats = nfl.load_player_stats(seasons=[season])
    if isinstance(stats, pl.LazyFrame):
        stats = stats.collect()

    pos_col = _resolve_position_column(stats)
    stats = stats.filter(pl.col(pos_col) == position.upper())

    if stats.is_empty():
        raise ValueError(f"No {position.upper()} stats found for season {season}.")

    stats = stats.sort(["player_id", "season", "week"])
    fantasy_col = _resolve_fantasy_column(set(stats.columns), scoring)
    return stats, fantasy_col


# ---------------------- defense-vs-position helper features ------------------


def build_defense_vs_position_features(stats: pl.DataFrame, fantasy_col: str) -> pl.DataFrame:
    """
    For each defense team and week, compute how many fantasy points they allowed
    to this position, then derive rolling features so we can say:
      - def_fp_prev1: last game allowed
      - def_fp_roll3: last 3 games allowed (avg)
      - def_fp_roll5: last 5 games allowed (avg)

    The resulting table has one row per (def_team, week) and the
    rolling stats are constructed so that row with week=W reflects
    performance up through week W-1 (i.e., usable BEFORE week W).
    """
    opp_col = _resolve_column(set(stats.columns), ("opponent_team", "opponent"), required=True)
    assert opp_col is not None

    # Sum fantasy points allowed to this position by defense (opponent) and week
    raw = (
        stats.group_by([opp_col, "week"])
        .agg(
            pl.col(fantasy_col).sum().alias("def_fp_allowed"),
        )
        .sort([opp_col, "week"])
    )

    # Build rolling stats with a shift so that week=W row reflects
    # knowledge from weeks <= W-1.
    def_feats = raw.with_columns(
        [
            pl.col("def_fp_allowed").shift(1).alias("def_fp_prev1"),
            pl.col("def_fp_allowed").shift(1).rolling_mean(3).alias("def_fp_roll3"),
            pl.col("def_fp_allowed").shift(1).rolling_mean(5).alias("def_fp_roll5"),
        ]
    )

    # Standardize defense team column name
    def_feats = def_feats.rename({opp_col: "def_team"})
    return def_feats


# ----------------------- build training data (per season) --------------------


def build_position_training_dataset(season: int, position: str, scoring: str) -> Tuple[pd.DataFrame, int]:
    """
    Build training data for a position in the given season.

    For each week t (starting at 2), we:
      - look at *history up to week t-1* for each player
      - compute features:
          fp_prev1      = fantasy points in last game
          fp_roll3      = mean of last 3 games
          fp_roll5      = mean of last 5 games
          fp_season_avg = average of all games up to t-1
      - defense features (for the defense they face in week t):
          def_fp_prev1  = last game's fantasy points allowed to this position
          def_fp_roll3  = last 3 games average fantasy points allowed to this position
          def_fp_roll5  = last 5 games average fantasy points allowed to this position
      - target_fp      = fantasy points actually scored in week t

    This ensures the model always sees the same kind of information
    that you'll use when predicting the *next* week.
    """
    stats, fantasy_col = load_position_stats(season, position, scoring)

    max_week = stats.select(pl.col("week").max()).row(0)[0]

    # Resolve team column (recent_team in some schemas, team in others)
    team_col = _resolve_column(set(stats.columns), ("recent_team", "team"), required=False) or "team"

    # Build defense-vs-position features once for the season
    def_feats = build_defense_vs_position_features(stats, fantasy_col)

    rows: list[pl.DataFrame] = []

    # Pre-resolve opponent column for targets
    opp_col = _resolve_column(set(stats.columns), ("opponent_team", "opponent"), required=True)
    assert opp_col is not None

    # For each target week t, build features based on weeks <= t-1
    for week in range(2, max_week + 1):
        # History up through t-1
        hist = stats.filter(pl.col("week") <= week - 1)

        # Offensive features per player based on recent + longer-term performance
        feats = (
            hist.group_by("player_id")
            .agg(
                pl.col("player_name").last().alias("player_name"),
                pl.col(team_col).last().alias("team"),
                pl.col(fantasy_col).tail(1).mean().alias("fp_prev1"),
                pl.col(fantasy_col).tail(3).mean().alias("fp_roll3"),
                pl.col(fantasy_col).tail(5).mean().alias("fp_roll5"),
                pl.col(fantasy_col).mean().alias("fp_season_avg"),
            )
            .with_columns(
                pl.lit(season).alias("season"),
                pl.lit(week).alias("week"),
            )
        )

        # Actual fantasy points in week t (target) AND the defense they faced
        targets = (
            stats.filter(pl.col("week") == week)
            .select(
                "player_id",
                pl.col(fantasy_col).alias("target_fp"),
                pl.col(opp_col).alias("def_team"),
            )
        )

        # Join offensive features with target + defense team
        joined = feats.join(targets, on="player_id", how="inner")

        # Join in defense-vs-position features for this week (week=W row uses info up to W-1)
        joined = joined.join(
            def_feats,
            on=["def_team", "week"],
            how="left",
        )

        rows.append(joined)

    train_df_pl = pl.concat(rows)

    # Convert to pandas for scikit-learn
    train_df_pd = train_df_pl.to_pandas()

    # Drop rows without enough offensive or defensive history
    required_cols = [
        "fp_prev1",
        "fp_roll3",
        "fp_roll5",
        "fp_season_avg",
        "def_fp_prev1",
        "def_fp_roll3",
        "def_fp_roll5",
        "target_fp",
    ]
    train_df_pd = train_df_pd.dropna(subset=required_cols)

    return train_df_pd, max_week


# -------------------------- training & prediction ----------------------------


def train_position_model(season: int, position: str, scoring: str) -> Tuple[RandomForestRegressor, List[str], int]:
    """
    Train a RandomForest model on position data for a season.
    Returns (model, feature_columns, max_week_observed).
    """
    df, max_week = build_position_training_dataset(season, position, scoring)

    feature_cols = [
        "fp_prev1",
        "fp_roll3",
        "fp_roll5",
        "fp_season_avg",
        "week",
        "def_fp_prev1",
        "def_fp_roll3",
        "def_fp_roll5",
    ]
    target_col = "target_fp"

    X = df[feature_cols]
    y = df[target_col]

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    # Just to get a sense of fit (in-sample MAE)
    preds_in_sample = model.predict(X)
    mae = mean_absolute_error(y, preds_in_sample)
    print(f"In-sample MAE ({season} {position.upper()}, weeks 2..{max_week}): {mae:.2f} fantasy points")

    return model, feature_cols, max_week


def build_upcoming_week_features(max_week: int, season: int, position: str, scoring: str) -> pd.DataFrame:
    """
    Build feature rows for the *upcoming* week, which we define as max_week + 1.

    For each player, we look at all games up through max_week and compute:
      fp_prev1, fp_roll3, fp_roll5, fp_season_avg,
    then figure out which defense they'll face in the upcoming week using the
    schedule, attach the defense-vs-position rolling features, and set 'week' to
    upcoming_week for all rows.
    """
    stats, fantasy_col = load_position_stats(season, position, scoring)

    # Only consider offensive history up through the last completed week
    hist = stats.filter(pl.col("week") <= max_week)

    # Resolve team column (recent_team in some schemas, team in others)
    team_col = _resolve_column(set(stats.columns), ("recent_team", "team"), required=False) or "team"

    feats_pl = (
        hist.group_by("player_id")
        .agg(
            pl.col("player_name").last().alias("player_name"),
            pl.col(team_col).last().alias("team"),
            pl.col(fantasy_col).tail(1).mean().alias("fp_prev1"),
            pl.col(fantasy_col).tail(3).mean().alias("fp_roll3"),
            pl.col(fantasy_col).tail(5).mean().alias("fp_roll5"),
            pl.col(fantasy_col).mean().alias("fp_season_avg"),
        )
    )

    upcoming_week = max_week + 1

    # Convert to pandas for easier schedule joining
    feats_df = feats_pl.to_pandas()
    feats_df["season"] = season
    feats_df["week"] = upcoming_week

    # ---------------------- attach upcoming opponent (def_team) via schedule ----------------------

    # Load schedule for this season
    sched = nfl.load_schedules(seasons=[season])
    if isinstance(sched, pl.LazyFrame):
        sched = sched.collect()
    sched_df = sched.to_pandas()

    # Resolve schedule columns
    sched_available = set(sched_df.columns)
    season_col = _resolve_column(sched_available, ("season",), required=True)
    week_col = _resolve_column(sched_available, ("week",), required=True)
    home_col = _resolve_column(sched_available, ("home_team", "home"), required=True)
    away_col = _resolve_column(sched_available, ("away_team", "away"), required=True)

    # Filter to games in the upcoming week
    sched_upcoming = sched_df[
        (sched_df[season_col] == season) & (sched_df[week_col] == upcoming_week)
    ]
    if sched_upcoming.empty:
        raise ValueError(f"No schedule data for season {season}, week {upcoming_week}.")

    # Build a mapping from (team, week) -> def_team
    opp_rows = []
    for _, game in sched_upcoming.iterrows():
        home_team = game[home_col]
        away_team = game[away_col]
        opp_rows.append({"team": home_team, "week": upcoming_week, "def_team": away_team})
        opp_rows.append({"team": away_team, "week": upcoming_week, "def_team": home_team})
    opp_map_df = pd.DataFrame(opp_rows)

    # Merge to attach def_team to each player
    feats_df = feats_df.merge(opp_map_df, on=["team", "week"], how="inner")

    # ---------------------- attach defense-vs-position rolling features ----------------------

    def_feats_pl = build_defense_vs_position_features(stats, fantasy_col)

    # For upcoming week (max_week + 1), use the latest available defensive
    # rolling stats (up through max_week) for each defense.
    def_feats_latest = (
        def_feats_pl
        .filter(pl.col("week") <= max_week)
        .sort(["def_team", "week"])
        .group_by("def_team")
        .tail(1)
        .with_columns(pl.lit(upcoming_week).alias("week"))
        .to_pandas()
    )

    # Merge on def_team + week to bring in def_fp_prev1/roll3/roll5
    feats_df = feats_df.merge(
        def_feats_latest,
        on=["def_team", "week"],
        how="left",
    )

    # Drop rows where we don't have defensive history (very early in season)
    feats_df = feats_df.dropna(subset=["def_fp_prev1", "def_fp_roll3", "def_fp_roll5"])

    return feats_df


def predict_upcoming_week_topn(season: int, position: str, scoring: str, top_n: int = 10) -> pd.DataFrame:
    """
    Train the model and then predict the top players
    for the upcoming week (max_week + 1).
    """
    model, feature_cols, max_week = train_position_model(season, position, scoring)
    print(f"Latest completed week in {season} {position.upper()} data: {max_week}")
    upcoming_week = max_week + 1
    print(f"Predicting for upcoming week: {upcoming_week}")

    feats_df = build_upcoming_week_features(max_week, season, position, scoring)

    # Ensure feature columns are present
    missing = [c for c in feature_cols if c not in feats_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns for prediction: {missing}")

    X_pred = feats_df[feature_cols]
    preds = model.predict(X_pred)

    feats_df["predicted_fantasy_points"] = preds
    feats_df["predicted_week"] = upcoming_week

    # Sort and return top N
    topn = (
        feats_df.sort_values("predicted_fantasy_points", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return topn


# ------------------------------ table printing -------------------------------


def print_top_table(df: pd.DataFrame, top_n: int, position: str) -> None:
    """
    Print a simple text table of the top projections.
    Columns: rank, player, team, week, pred_fp
    """
    if df.empty:
        print("No projection data available.")
        return

    # Create a simple, aligned table similar to nfl.py output
    columns = ["rank", "player", "pos", "team", "week", "pred_fp"]
    table_rows = [
        (
            idx + 1,
            row.player_name,
            position.upper(),
            row.team,
            int(row.predicted_week),
            f"{row.predicted_fantasy_points:.2f}",
        )
        for idx, row in df.head(top_n).iterrows()
    ]

    widths = {
        col: max(len(col), *(len(str(row[idx])) for row in table_rows))
        for idx, col in enumerate(columns)
    }

    def render_row(values: tuple) -> str:
        return " | ".join(str(value).ljust(widths[col]) for value, col in zip(values, columns))

    separator = "-+-".join("-" * widths[col] for col in columns)
    lines = [
        render_row(columns),
        separator,
        *(render_row(row) for row in table_rows),
    ]

    print(f"\nTop projected {position.upper()}s (upcoming week)")
    print("\n".join(lines))


# ---------------------------------- main -------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a fantasy-point model (with defense matchup) and show top projected players for the upcoming week."
    )
    parser.add_argument("--season", type=int, default=2025, help="Season to train on (default: 2025)")
    parser.add_argument(
        "--scoring",
        type=str,
        default="ppr",
        choices=["ppr", "half", "standard"],
        help="Scoring type to choose fantasy column (default: ppr)",
    )
    parser.add_argument(
        "--position",
        type=str,
        default="WR",
        choices=list(POSITION_CHOICES),
        help="Position to model (QB/RB/WR/TE). Default: WR",
    )
    parser.add_argument("--top", type=int, default=10, help="How many players to show (default: 10)")
    args = parser.parse_args()

    if pyarrow is None:
        print(
            "pyarrow is required to read the nflverse parquet files but is not installed.\n"
            'Install it with: pip install "pyarrow>=14,<16"'
        )
        return

    try:
        top_df = predict_upcoming_week_topn(args.season, args.position, args.scoring, args.top)
    except (req_exc.RequestException, ConnectionError) as err:
        # nflreadpy can surface requests.ConnectionError or builtin ConnectionError
        print(
            "Unable to download stats from nflverse (network/DNS blocked). "
            "Re-run once you have internet access.\n"
            f"Details: {err}"
        )
        return
    except ValueError as err:
        print(err)
        return

    print_top_table(top_df, args.top, args.position)


if __name__ == "__main__":
    main()
