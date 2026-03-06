from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_autochess import compute_target_metrics_from_pbp, save_target_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target metrics JSON from nflverse PBP CSV/CSV.GZ")
    parser.add_argument("pbp", help="Path to play-by-play file (.csv or .csv.gz)")
    parser.add_argument("--output", default="data/target_metrics.json", help="Output JSON file")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick runs")
    args = parser.parse_args()

    targets = compute_target_metrics_from_pbp(args.pbp, max_rows=args.max_rows)
    save_target_metrics(targets, args.output)

    print("Saved target metrics:")
    print(json.dumps(targets, indent=2))
    print(f"\nOutput: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
