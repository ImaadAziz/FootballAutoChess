from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_autochess import GameState, simulate_down  # noqa: E402
from sample_teams import build_defense, build_offense  # noqa: E402


def main() -> None:
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

    result = simulate_down(state, offense, defense, rng_seed=7)

    print("=== Play Result ===")
    print(f"Summary: {result.summary}")
    print(f"Play: {result.offensive_play.name} ({result.offensive_play.play_type.value}) vs {result.defensive_play.name}")
    print(f"Yards: {result.yards_gained}")
    print(f"First Down: {result.first_down}")
    print(f"Turnover: {result.turnover}")
    print("Events:")
    for event in result.events:
        print(f"- {event}")

    print("\nNext State:")
    print(result.next_state)

    print("\nDebug:")
    print(json.dumps(result.debug, indent=2))


if __name__ == "__main__":
    main()
