from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_autochess import (  # noqa: E402
    GameState,
    ModelBundle,
    SimulationTuning,
    compare_metrics_to_targets,
    load_target_metrics,
    simulate_many_drives,
)
from sample_teams import build_defense, build_offense  # noqa: E402


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _load_calibration(path: str | None) -> tuple[dict[str, float], dict[str, float], SimulationTuning | None]:
    if not path:
        return {}, {}, None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    offense_tendencies = dict(payload.get("offense_tendencies", {}))
    defense_tendencies = dict(payload.get("defense_tendencies", {}))
    tuning = SimulationTuning.from_dict(payload.get("tuning"))
    return offense_tendencies, defense_tendencies, tuning


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate simulator metrics against target metrics")
    parser.add_argument("--targets", default="data/target_metrics.json", help="Target metrics JSON")
    parser.add_argument("--drives", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--tolerance", type=float, default=0.12, help="Relative error tolerance per metric")
    parser.add_argument("--model-bundle", default=None, help="Optional trained model bundle JSON")
    parser.add_argument("--calibration", default=None, help="Optional calibration_result.json")
    args = parser.parse_args()

    offense = build_offense()
    defense = build_defense()
    offense_patch, defense_patch, tuning = _load_calibration(args.calibration)
    if offense_patch:
        offense = replace(offense, tendencies={**offense.tendencies, **offense_patch})
    if defense_patch:
        defense = replace(defense, tendencies={**defense.tendencies, **defense_patch})

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
        tuning=tuning,
    )

    targets = load_target_metrics(args.targets)
    report = compare_metrics_to_targets(batch.metrics, targets, tolerance=args.tolerance)

    print("=== Validation Report ===")
    print(f"Drives: {args.drives}")
    print(f"Tolerance: {_pct(args.tolerance)}")
    if args.calibration:
        print(f"Calibration: {Path(args.calibration).resolve()}")
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
