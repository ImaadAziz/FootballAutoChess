from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_autochess import build_round_start_state, simulate_round_play  # noqa: E402
from roguelite_sample_teams import DEFENSE_IDENTITIES, DEFENSE_TEAM_BUILDERS, STARTER_OFFENSE_BUILDERS, build_defense_matchup, build_starter_offense  # noqa: E402
from run_text_round import build_demo_round_config, pick_demo_play  # noqa: E402


def _pct(value: float) -> str:
    return f"{value * 100:5.1f}%"


def _num(value: float) -> str:
    return f"{value:5.2f}"


def _simulate_matchup(
    archetype: str,
    defense_slug: str,
    rounds: int,
    target_yards: int,
    play_budget: int,
    seed_base: int,
) -> dict[str, float]:
    offense = build_starter_offense(archetype)
    defense, defense_identity = build_defense_matchup(defense_slug)
    config = build_demo_round_config(target_yards, play_budget, defense_identity)

    wins = 0
    turnovers = 0
    total_plays = 0
    total_yards = 0
    total_remaining = 0

    for index in range(rounds):
        state = build_round_start_state(offense.id, config.target_yards)
        rng = random.Random(seed_base + index)
        history = []

        for _ in range(config.play_budget):
            offensive_play = pick_demo_play(offense, archetype, defense_identity, state, history)
            snap = simulate_round_play(
                state,
                offense,
                defense,
                offensive_play,
                defense_identity,
                rng_seed=rng.randint(0, 2_147_483_647),
                recent_results=tuple(history),
            )
            result = snap.play_result
            history.append(result)
            total_yards += result.yards_gained
            state = result.next_state

            if result.touchdown:
                wins += 1
                break
            if result.turnover:
                turnovers += 1
                break

        total_plays += len(history)
        total_remaining += max(0, state.distance)

    return {
        "win_rate": wins / rounds,
        "turnover_rate": turnovers / rounds,
        "avg_plays_used": total_plays / rounds,
        "avg_yards_per_play": total_yards / max(1, total_plays),
        "avg_remaining_yards": total_remaining / rounds,
    }


def _print_table(
    title: str,
    defense_slugs: list[str],
    offense_slugs: list[str],
    metrics: dict[tuple[str, str], dict[str, float]],
    metric_key: str,
    formatter,
) -> None:
    defense_labels = [DEFENSE_IDENTITIES[slug].name for slug in defense_slugs]
    offense_labels = [STARTER_OFFENSE_BUILDERS[slug]().name for slug in offense_slugs]
    row_width = max(len(label) for label in offense_labels) + 2
    col_width = max(max(len(label) for label in defense_labels), 12) + 2

    print(title)
    header = "Offense".ljust(row_width) + "".join(label.ljust(col_width) for label in defense_labels)
    print(header)
    for offense_slug, offense_label in zip(offense_slugs, offense_labels):
        row = offense_label.ljust(row_width)
        for defense_slug in defense_slugs:
            row += formatter(metrics[(offense_slug, defense_slug)][metric_key]).ljust(col_width)
        print(row)
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare starter offenses against all prototype defensive identities.")
    parser.add_argument("--rounds", type=int, default=200, help="Number of seeded rounds to simulate per matchup.")
    parser.add_argument("--seed-base", type=int, default=1000, help="Base RNG seed used to build matchup samples.")
    parser.add_argument("--target-yards", type=int, default=40, help="How many yards the offense must gain to score.")
    parser.add_argument("--play-budget", type=int, default=6, help="How many total snaps the round allows.")
    args = parser.parse_args()

    offense_slugs = list(sorted(STARTER_OFFENSE_BUILDERS))
    defense_slugs = list(sorted(DEFENSE_TEAM_BUILDERS))
    metrics: dict[tuple[str, str], dict[str, float]] = {}

    for offense_slug in offense_slugs:
        for defense_slug in defense_slugs:
            metrics[(offense_slug, defense_slug)] = _simulate_matchup(
                offense_slug,
                defense_slug,
                args.rounds,
                args.target_yards,
                args.play_budget,
                args.seed_base,
            )

    print("=== Prototype Matchup Matrix ===")
    print(f"Rounds per matchup: {args.rounds}")
    print(f"Round setup: {args.target_yards} yards, {args.play_budget} plays")
    print("")

    _print_table("Win Rate", defense_slugs, offense_slugs, metrics, "win_rate", _pct)
    _print_table("Turnover Rate", defense_slugs, offense_slugs, metrics, "turnover_rate", _pct)
    _print_table("Average Plays Used", defense_slugs, offense_slugs, metrics, "avg_plays_used", _num)
    _print_table("Average Yards Per Play", defense_slugs, offense_slugs, metrics, "avg_yards_per_play", _num)
    _print_table("Average Yards Remaining", defense_slugs, offense_slugs, metrics, "avg_remaining_yards", _num)


if __name__ == "__main__":
    main()
