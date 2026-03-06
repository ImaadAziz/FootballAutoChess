from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_autochess import (  # noqa: E402
    GameState,
    ModelBundle,
    compare_metrics_to_targets,
    load_target_metrics,
    simulate_many_drives,
)
from sample_teams import build_defense, build_offense  # noqa: E402


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate simulator metrics against target metrics")
    parser.add_argument("--targets", default="data/target_metrics.json", help="Target metrics JSON")
    parser.add_argument("--drives", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--tolerance", type=float, default=0.12, help="Relative error tolerance per metric")
    parser.add_argument("--model-bundle", default=None, help="Optional trained model bundle JSON")
    args = parser.parse_args()

    offense = build_offense()
    defense = build_defense()
    state = GameState(
        possession_team_id=offense.id,
        down=1,
        distance=10,
        yard_line=25,
        quarter=1,
        clock_seconds=900,
    )

    model_bundle = ModelBundle.load_json(args.model_bundle) if args.model_bundle else None

    batch = simulate_many_drives(
        state,
        offense,
        defense,
        num_drives=args.drives,
        rng_seed=args.seed,
        model_bundle=model_bundle,
    )

    targets = load_target_metrics(args.targets)
    report = compare_metrics_to_targets(batch.metrics, targets, tolerance=args.tolerance)

    print("=== Validation Report ===")
    print(f"Drives: {args.drives}")
    print(f"Tolerance: {_pct(args.tolerance)}")
    print(f"Pass rate: {_pct(report.pass_rate)}")
    print(f"Mean rel error: {_pct(report.mean_rel_error)}")
    print("")

    for row in sorted(report.comparisons, key=lambda item: item.metric):
        status = "PASS" if row.within_tolerance else "FAIL"
        print(
            f"- {row.metric}: sim={row.simulated:.4f}, target={row.target:.4f}, "
            f"rel_err={_pct(row.rel_error)} [{status}]"
        )


if __name__ == "__main__":
    main()
