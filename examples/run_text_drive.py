from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_autochess import GameState, simulate_drive  # noqa: E402
from sample_teams import build_defense, build_offense  # noqa: E402


def _format_clock(seconds: int) -> str:
    mins = max(0, seconds) // 60
    secs = max(0, seconds) % 60
    return f"{mins:02d}:{secs:02d}"


def _yards_text(yards: int) -> str:
    sign = "+" if yards > 0 else ""
    return f"{sign}{yards}"


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

    drive = simulate_drive(state, offense, defense, rng_seed=17)

    print("=== Text Drive Simulation ===")
    print(f"Offense: {offense.name}")
    print(f"Defense: {defense.name}")
    print(f"Start: Q{state.quarter} {_format_clock(state.clock_seconds)} | {state.down}&{state.distance} @ {state.yard_line}")
    print("")

    current_state = state
    for index, play in enumerate(drive.plays, start=1):
        print(
            f"Play {index:02d} | Q{current_state.quarter} {_format_clock(current_state.clock_seconds)}"
            f" | {current_state.down}&{current_state.distance} @ {current_state.yard_line}"
            f" | {play.offensive_play.name} ({play.offensive_play.play_type.value}) vs {play.defensive_play.name}"
            f" | {play.summary} {_yards_text(play.yards_gained)}"
        )

        terminal_event = play.events[-1] if play.events else "No event"
        print(f"  Event: {terminal_event}")
        current_state = play.next_state

    print("")
    print("=== Drive Summary ===")
    print(f"Outcome: {drive.outcome}")
    print(f"Plays: {len(drive.plays)}")
    print(f"Total Yards: {drive.total_yards}")
    print(f"First Downs: {drive.first_downs}")
    print(f"Turnovers: {drive.turnovers}")
    print(
        f"End: Q{drive.end_state.quarter} {_format_clock(drive.end_state.clock_seconds)}"
        f" | {drive.end_state.down}&{drive.end_state.distance} @ {drive.end_state.yard_line}"
    )
    print(
        f"Score (offense view): {drive.end_state.offense_score}-{drive.end_state.defense_score}"
    )


if __name__ == "__main__":
    main()
