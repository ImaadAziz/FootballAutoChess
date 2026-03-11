from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_autochess import build_round_replay, save_round_replay  # noqa: E402
from roguelite_sample_teams import DEFENSE_TEAM_BUILDERS, STARTER_OFFENSE_BUILDERS, build_defense_matchup, build_starter_offense  # noqa: E402
from run_text_round import build_demo_round_config, pick_demo_play  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a deterministic roguelite round replay JSON for the Godot client.")
    parser.add_argument(
        "--archetype",
        choices=sorted(STARTER_OFFENSE_BUILDERS),
        default="precision-pass",
        help="Starter offense archetype to simulate.",
    )
    parser.add_argument(
        "--defense-identity",
        choices=sorted(DEFENSE_TEAM_BUILDERS),
        default="run-wall-zone",
        help="Prototype defense identity for the replay.",
    )
    parser.add_argument("--seed", type=int, default=23, help="Deterministic replay seed.")
    parser.add_argument("--target-yards", type=int, default=40, help="How many yards the offense must gain to score.")
    parser.add_argument("--play-budget", type=int, default=6, help="How many total snaps the round allows.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "apps" / "godot_client" / "replays" / "prototype_round.json",
        help="Output path for the replay JSON.",
    )
    args = parser.parse_args()

    offense = build_starter_offense(args.archetype)
    defense, defense_identity = build_defense_matchup(args.defense_identity)
    config = build_demo_round_config(args.target_yards, args.play_budget, defense_identity)

    replay = build_round_replay(
        config,
        offense,
        defense,
        play_picker=lambda state, history: pick_demo_play(offense, args.archetype, defense_identity, state, history),
        seed=args.seed,
    )
    output_path = save_round_replay(replay, args.output)
    print(f"Saved replay: {output_path}")
    print(f"Outcome: {replay.meta['outcome']} | Plays used: {replay.meta['plays_used']}")


if __name__ == "__main__":
    main()
