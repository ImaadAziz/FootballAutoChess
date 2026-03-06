from __future__ import annotations

import argparse
from pathlib import Path


def _parse_seasons(values: list[str] | None, load_all: bool) -> int | list[int] | bool | None:
    if load_all:
        return True
    if not values:
        return None
    return [int(value) for value in values]


def _default_output(seasons: int | list[int] | bool | None, file_format: str) -> Path:
    suffix = ".parquet" if file_format == "parquet" else ".csv"
    if seasons is True:
        stem = "pbp_all"
    elif seasons is None:
        stem = "pbp_current"
    else:
        year_list = seasons if isinstance(seasons, list) else [seasons]
        stem = f"pbp_{'_'.join(str(year) for year in year_list)}"
    return Path("data") / f"{stem}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load nflverse play-by-play data with nflreadpy and save it locally."
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        help="One or more seasons to load, for example: --seasons 2023 2024 2025",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Load every available PBP season since 1999.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Defaults to data/pbp_<seasons>.parquet or .csv.",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Output format for the saved PBP file.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Optional subset of columns to keep before writing the file.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="How many preview rows to print after loading.",
    )
    args = parser.parse_args()

    try:
        import nflreadpy as nfl
    except ImportError as exc:
        raise SystemExit(
            "nflreadpy is not installed. Install it with `pip install nflreadpy` or `uv add nflreadpy`."
        ) from exc

    seasons = _parse_seasons(args.seasons, args.all)
    pbp = nfl.load_pbp(seasons=seasons)

    if args.columns:
        missing = [column for column in args.columns if column not in pbp.columns]
        if missing:
            raise SystemExit(f"Requested columns are missing from the PBP data: {', '.join(missing)}")
        pbp = pbp.select(args.columns)

    output_path = Path(args.output) if args.output else _default_output(seasons, args.format)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "parquet":
        pbp.write_parquet(output_path)
    else:
        pbp.write_csv(output_path)

    print(f"Rows: {pbp.height}")
    print(f"Columns: {pbp.width}")
    print(f"Saved: {output_path.resolve()}")

    if args.head > 0:
        print("")
        print(pbp.head(args.head))


if __name__ == "__main__":
    main()
