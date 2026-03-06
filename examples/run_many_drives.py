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

from football_autochess import GameState, ModelBundle, SimulationTuning, simulate_many_drives  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Run bulk drive benchmark")
    parser.add_argument("--drives", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--model-bundle", default=None)
    parser.add_argument("--calibration", default=None, help="Optional calibration_result.json")
    args = parser.parse_args()

    offense = build_offense()
    defense = build_defense()
    offense_patch, defense_patch, tuning = _load_calibration(args.calibration)
    if offense_patch:
        offense = replace(offense, tendencies={**offense.tendencies, **offense_patch})
    if defense_patch:
        defense = replace(defense, tendencies={**defense.tendencies, **defense_patch})

    base_state = GameState(
        possession_team_id=offense.id,
        down=1,
        distance=10,
        yard_line=25,
        quarter=1,
        clock_seconds=900,
    )

    model_bundle = ModelBundle.load_json(args.model_bundle) if args.model_bundle else None

    batch = simulate_many_drives(
        base_state,
        offense,
        defense,
        num_drives=args.drives,
        rng_seed=args.seed,
        max_plays=20,
        model_bundle=model_bundle,
        tuning=tuning,
    )

    print(f"=== {args.drives}-Drive Benchmark ===")
    print(f"Seed: {batch.seed}")
    print(f"Drives: {batch.num_drives}")
    print(f"Total plays: {batch.total_plays}")
    if args.calibration:
        print(f"Calibration: {Path(args.calibration).resolve()}")
    print("")

    print("Outcomes:")
    for key in sorted(batch.outcome_counts.keys()):
        count = batch.outcome_counts[key]
        print(f"- {key}: {count} ({_pct(count / batch.num_drives)})")

    print("")
    print("Core Metrics:")
    print(f"- Plays/drive: {batch.metrics['plays_per_drive']}")
    print(f"- Yards/play: {batch.metrics['yards_per_play']}")
    print(f"- Yards/drive: {batch.metrics['yards_per_drive']}")
    print(f"- First downs/drive: {batch.metrics['first_downs_per_drive']}")
    print(f"- TD rate: {_pct(batch.metrics['touchdown_rate'])}")
    print(f"- Turnover drive rate: {_pct(batch.metrics['turnover_drive_rate'])}")
    print(f"- Run rate: {_pct(batch.metrics['run_rate'])}")
    print(f"- Completion rate: {_pct(batch.metrics['completion_rate'])}")
    print(f"- Sack rate/pass play: {_pct(batch.metrics['sack_rate_per_pass_play'])}")
    print(f"- INT rate/pass play: {_pct(batch.metrics['interception_rate_per_pass_play'])}")
    print(f"- Fumble rate/run play: {_pct(batch.metrics['fumble_rate_per_run_play'])}")
    print(f"- Explosive play rate: {_pct(batch.metrics['explosive_play_rate'])}")


if __name__ == "__main__":
    main()
