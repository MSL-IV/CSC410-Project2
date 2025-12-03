import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import nflreadpy as nfl
import polars as pl
from requests import exceptions as req_exc

try:
    import pyarrow  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    pyarrow = None


ColumnOptions = Sequence[str]


BASE_COLUMNS: Sequence[Tuple[str, ColumnOptions, bool]] = (
    ("player", ("player_name", "player_display_name", "name"), True),
    ("team", ("team", "recent_team"), True),
    ("opponent", ("opponent_team", "opponent"), True),
    ("week", ("week",), True),
    ("fantasy_points", ("fantasy_points_ppr", "fantasy_points"), True),
)

POSITION_COLUMNS: dict[str, Sequence[Tuple[str, ColumnOptions, bool]]] = {
    "WR": (
        ("targets", ("targets",), False),
        ("receptions", ("receptions", "rec"), False),
        ("receiving_yards", ("receiving_yards", "rec_yds"), False),
        ("receiving_td", ("receiving_tds", "rec_td"), False),
    ),
    "TE": (
        ("targets", ("targets",), False),
        ("receptions", ("receptions", "rec"), False),
        ("receiving_yards", ("receiving_yards", "rec_yds"), False),
        ("receiving_td", ("receiving_tds", "rec_td"), False),
    ),
    "RB": (
        ("rushing_attempts", ("rushing_attempts", "rush_attempts", "rush_att"), False),
        ("rushing_yards", ("rushing_yards", "rush_yards", "rush_yds"), False),
        ("rushing_td", ("rushing_tds", "rush_td", "rushing_touchdowns"), False),
        ("receptions", ("receptions", "rec"), False),
        ("receiving_yards", ("receiving_yards", "rec_yds"), False),
    ),
    "QB": (
        ("pass_attempts", ("passing_attempts", "pass_attempts"), False),
        ("pass_completions", ("completions", "passing_completions", "pass_completions"), False),
        ("passing_yards", ("passing_yards", "pass_yards"), False),
        ("passing_td", ("passing_tds", "pass_td"), False),
        ("interceptions", ("interceptions", "int"), False),
        ("rushing_yards", ("rushing_yards", "rush_yards", "rush_yds"), False),
    ),
}


def _resolve_column(
    available: set[str], options: ColumnOptions, required: bool
) -> Optional[str]:
    for candidate in options:
        if candidate in available:
            return candidate
    if required:
        raise ValueError(
            f"Required column not found. Tried {options}. Available columns: {sorted(available)}"
        )
    return None


def _resolve_position_column(df: pl.DataFrame) -> str:
    """Detect a usable position column (position vs position_group)."""
    available_cols = set(df.columns)
    for candidate in ("position", "position_group", "pos"):
        if candidate in available_cols:
            return candidate
    raise ValueError(
        f"No position column found. Available columns: {sorted(available_cols)}"
    )


def _get_display_columns(position: str) -> Sequence[Tuple[str, ColumnOptions, bool]]:
    """Combine base columns with position-specific metrics."""
    position = position.upper()
    extra = POSITION_COLUMNS.get(position, POSITION_COLUMNS["WR"])
    return BASE_COLUMNS + extra


def _resolve_fantasy_column(
    available_columns: set[str], scoring: str
) -> str:
    """
    Choose an appropriate fantasy points column based on scoring type.
    Falls back gracefully if a specific variant isn't present.
    """
    scoring = scoring.lower()
    if scoring == "ppr":
        options = ("fantasy_points_ppr", "fantasy_points")
    elif scoring in ("half", "half_ppr", "0.5ppr"):
        # column names can vary a bit; try a few
        options = (
            "fantasy_points_half_ppr",
            "fantasy_points_hppr",
            "fantasy_points",
        )
    elif scoring in ("standard", "std", "non_ppr"):
        options = ("fantasy_points", "fantasy_points_std", "fantasy_points_ppr")
    else:
        # unknown scoring label: just try generic ones
        options = ("fantasy_points", "fantasy_points_ppr")

    fantasy_col = _resolve_column(available_columns, options, required=True)
    return fantasy_col


def get_top_players_week(
    season: int,
    week: Optional[int],
    position: str = "WR",
    top_n: int = 10,
    scoring: str = "ppr",
) -> Tuple[int, pl.DataFrame]:
    """
    Return (resolved_week, top-N players at a position) by weekly fantasy points.
    """
    player_stats = nfl.load_player_stats(seasons=[season])

    # Ensure concrete DataFrame (nflreadpy can return LazyFrame)
    if isinstance(player_stats, pl.LazyFrame):
        player_stats = player_stats.collect()

    pos_col = _resolve_position_column(player_stats)

    # Normalize requested position (WR, RB, TE, QB)
    position = position.upper()

    # Filter to the desired position
    pos_stats = player_stats.filter(pl.col(pos_col) == position)

    if pos_stats.is_empty():
        unique_positions = (
            player_stats.select(pl.col(pos_col).unique())
            .to_series()
            .to_list()
        )
        raise ValueError(
            f"No stats for position '{position}' in season {season}. "
            f"Available {pos_col} values include: {unique_positions}"
        )

    # Determine week if not provided (use latest available week)
    if week is None:
        week = pos_stats.select(pl.col("week").max()).row(0)[0]

    week_stats = pos_stats.filter(pl.col("week") == week)
    if week_stats.is_empty():
        available_weeks = sorted(pos_stats.get_column("week").unique().to_list())
        raise ValueError(
            f"No {position} stats found for season {season}, week {week}. "
            f"Available weeks: {available_weeks}"
        )

    available_columns = set(week_stats.columns)

    # Resolve the appropriate fantasy column based on scoring
    fantasy_col = _resolve_fantasy_column(available_columns, scoring)

    # Sort by fantasy points
    sorted_week = week_stats.sort(fantasy_col, descending=True)

    # Select display columns using the mapping
    sorted_columns = set(sorted_week.columns)
    select_exprs = []
    for alias, options, required in _get_display_columns(position):
        resolved = _resolve_column(sorted_columns, options, required)
        if resolved is None:
            continue
        select_exprs.append(pl.col(resolved).alias(alias))

    top_players = sorted_week.select(select_exprs).head(top_n)
    return week, top_players


def _format_table(df: pl.DataFrame) -> str:
    """Render a simple ASCII table from the provided DataFrame."""
    if df.is_empty():
        return "No data available."

    columns = df.columns
    rows = df.rows()

    widths = {
        col: max(len(col), *(len(str(row[idx])) for row in rows))
        for idx, col in enumerate(columns)
    }

    def render_row(values: Tuple) -> str:
        return " | ".join(str(value).ljust(widths[col]) for value, col in zip(values, columns))

    separator = "-+-".join("-" * widths[col] for col in columns)
    lines = [
        render_row(columns),
        separator,
        *(render_row(row) for row in rows),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show the top weekly fantasy performances for a given position."
    )
    parser.add_argument("--season", type=int, default=2024, help="Season year (default: 2024)")
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Week number to inspect (default: latest available)",
    )
    parser.add_argument("--top", type=int, default=10, help="Number of rows to show (default: 10)")
    parser.add_argument(
        "--position",
        type=str,
        default="WR",
        choices=["WR", "RB", "TE", "QB"],
        help="Position to filter (default: WR)",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="ppr",
        choices=["ppr", "half", "standard"],
        help="Scoring type to choose fantasy column (default: ppr)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="top_players_week.txt",
        help="Where to write the text table (default: top_players_week.txt)",
    )
    args = parser.parse_args()

    if pyarrow is None:
        print(
            "pyarrow is required to read the nflverse parquet files but is not installed.\n"
            'Install it with: pip install "pyarrow>=14,<16"'
        )
        return

    try:
        resolved_week, top_players = get_top_players_week(
            season=args.season,
            week=args.week,
            position=args.position,
            top_n=args.top,
            scoring=args.scoring,
        )
    except (req_exc.RequestException, ConnectionError) as err:
        # nflreadpy can raise either requests' ConnectionError or the builtin ConnectionError;
        # catch both so a blocked/failed download is reported cleanly.
        print(
            "Unable to download stats from nflverse (network/DNS blocked). "
            "Re-run once you have internet access.\n"
            f"Details: {err}"
        )
        return
    except ValueError as err:
        print(err)
        return

    if args.week is None:
        print(f"Showing latest available week ({resolved_week}).")

    table_str = _format_table(top_players)
    print(table_str)
    output_path = Path(args.output).expanduser().resolve()
    with output_path.open("w", encoding="utf-8") as f:
        f.write(table_str + "\n")
    print(f"Saved table to {output_path}")


if __name__ == "__main__":
    main()
