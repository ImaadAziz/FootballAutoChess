from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_autochess import (  # noqa: E402
    CalibrationConfig,
    GameState,
    calibrate_simulation_tendencies,
    load_target_metrics,
)
from sample_teams import build_defense, build_offense  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate team tendency knobs to target metrics")
    parser.add_argument("--targets", default="data/target_metrics.json", help="Target metrics JSON")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--drives", type=int, default=300, help="Drives per calibration iteration")
    parser.add_argument("--step", type=float, default=0.08, help="Mutation step size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-bundle", default=None, help="Optional trained model bundle JSON")
    parser.add_argument("--output", default="artifacts/calibration_result.json", help="Calibration output JSON")
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

    targets = load_target_metrics(args.targets)
    config = CalibrationConfig(
        iterations=args.iterations,
        drives_per_iteration=args.drives,
        step_size=args.step,
        seed=args.seed,
    )

    result = calibrate_simulation_tendencies(
        state,
        offense,
        defense,
        targets,
        config=config,
        model_bundle_path=args.model_bundle,
    )

    output_payload = {
        "best_loss": result.best_loss,
        "best_metrics": result.best_metrics,
        "offense_tendencies": result.offense_team.tendencies,
        "defense_tendencies": result.defense_team.tendencies,
        "history": list(result.history),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    print(f"Calibration complete. Loss={result.best_loss:.6f}")
    print(json.dumps(result.best_metrics, indent=2))
    print(f"\nSaved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
