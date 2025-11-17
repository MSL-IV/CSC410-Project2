import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

import nflreadpy as nfl
import polars as pl
from requests import exceptions as req_exc


ColumnOptions = Sequence[str]


DISPLAY_COLUMNS: Sequence[Tuple[str, ColumnOptions, bool]] = (
    ("player", ("player_name", "player_display_name", "name"), True),
    ("team", ("team", "recent_team"), True),
    ("opponent", ("opponent_team", "opponent"), True),
    ("week", ("week",), True),
    ("targets", ("targets",), False),
    ("receptions", ("receptions", "rec"), False),
    ("receiving_yards", ("receiving_yards", "rec_yds"), False),
    ("receiving_td", ("receiving_tds", "rec_td"), False),
    ("fantasy_points", ("fantasy_points_ppr", "fantasy_points"), True),
)


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


def get_top_wr_week(
    season: int, week: Optional[int], top_n: int = 10
) -> Tuple[int, pl.DataFrame]:
    """Return (resolved_week, top-N WRs) by weekly fantasy points."""
    player_stats = nfl.load_player_stats(seasons=[season])
    wr_stats = player_stats.filter(pl.col("position") == "WR")

    if wr_stats.is_empty():
        raise ValueError(f"No WR stats available for season {season}.")

    if week is None:
        week = wr_stats.select(pl.col("week").max()).row(0)[0]

    week_stats = wr_stats.filter(pl.col("week") == week)
    if week_stats.is_empty():
        available_weeks = sorted(wr_stats["week"].unique().to_list())
        raise ValueError(
            f"No WR stats found for season {season}, week {week}. "
            f"Available weeks: {available_weeks}"
        )

    available_columns = set(week_stats.columns)
    fantasy_col = _resolve_column(
        available_columns, ("fantasy_points_ppr", "fantasy_points"), required=True
    )

    sorted_week = week_stats.sort(fantasy_col, descending=True)

    sorted_columns = set(sorted_week.columns)
    select_exprs = []
    for alias, options, required in DISPLAY_COLUMNS:
        resolved = _resolve_column(sorted_columns, options, required)
        if resolved is None:
            continue
        select_exprs.append(pl.col(resolved).alias(alias))

    top_wr = sorted_week.select(select_exprs).head(top_n)
    return week, top_wr


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
        description="Show the top WR weekly fantasy performances."
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
        "--output",
        type=str,
        default="top_wr_week.txt",
        help="Where to write the text table (default: top_wr_week.txt)",
    )
    args = parser.parse_args()

    try:
        resolved_week, top_wr = get_top_wr_week(args.season, args.week, args.top)
    except req_exc.ConnectionError as err:
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

    table_str = _format_table(top_wr)
    print(table_str)
    output_path = Path(args.output).expanduser().resolve()
    with output_path.open("w", encoding="utf-8") as f:
        f.write(table_str + "\n")
    print(f"Saved table to {output_path}")


if __name__ == "__main__":
    main()
