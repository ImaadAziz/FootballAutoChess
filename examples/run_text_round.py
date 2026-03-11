from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_autochess import CoverageType, PlayType, RoundConfig, RoundDefenseIdentity, build_round_start_state, simulate_round_play  # noqa: E402
from roguelite_sample_teams import DEFENSE_TEAM_BUILDERS, STARTER_OFFENSE_BUILDERS, build_defense_matchup, build_starter_offense  # noqa: E402


def _format_clock(seconds: int) -> str:
    mins = max(0, seconds) // 60
    secs = max(0, seconds) % 60
    return f"{mins:02d}:{secs:02d}"


def _yards_text(yards: int) -> str:
    sign = "+" if yards > 0 else ""
    return f"{sign}{yards}"


def _pct(value: float) -> str:
    if value < 0:
        return "n/a"
    return f"{value * 100:.1f}%"


def _num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _roll(value: float) -> str:
    if value < 0:
        return "n/a"
    return f"{value:.4f}"


_MAN_BEATER_CONCEPTS = frozenset({"drag", "go", "post", "slant"})
_ZONE_BEATER_CONCEPTS = frozenset({"curl", "dig", "flat", "out", "sail", "stick"})


def build_demo_round_config(target_yards: int, play_budget: int, defense_identity: RoundDefenseIdentity) -> RoundConfig:
    return RoundConfig(
        name="Prototype Round 1",
        target_yards=target_yards,
        play_budget=play_budget,
        defense_identity=defense_identity,
    )


def _pick_first_fresh(candidates, recent_ids):
    for play in candidates:
        if recent_ids.count(play.id) == 0:
            return play
    return candidates[0] if candidates else None


def _play_groups(offense) -> dict[str, list[object]]:
    plays = list(offense.offensive_playbook)
    return {
        "runs_inside": [play for play in plays if play.play_type == PlayType.RUN and play.run_location == "middle"],
        "runs_edge": [play for play in plays if play.play_type == PlayType.RUN and play.run_location in {"left", "right"}],
        "quick_pass": [play for play in plays if play.play_type == PlayType.PASS and play.target_depth <= 5],
        "short_pass": [play for play in plays if play.play_type == PlayType.PASS and play.target_depth <= 7],
        "intermediate_pass": [play for play in plays if play.play_type == PlayType.PASS and 8 <= play.target_depth <= 14],
        "deep_pass": [play for play in plays if play.play_type == PlayType.PASS and play.target_depth >= 15],
        "play_action": [play for play in plays if play.play_type == PlayType.PASS and play.play_action],
        "man_beaters": [
            play
            for play in plays
            if play.play_type == PlayType.PASS and set(play.route_concepts) & _MAN_BEATER_CONCEPTS
        ],
        "zone_beaters": [
            play
            for play in plays
            if play.play_type == PlayType.PASS and set(play.route_concepts) & _ZONE_BEATER_CONCEPTS
        ],
    }


def _base_group_preferences(archetype: str, distance: int, recent_types: list[PlayType]) -> list[str]:
    if archetype == "ground-control":
        if distance >= 18:
            return ["play_action", "deep_pass", "intermediate_pass"]
        if distance >= 10:
            if len(recent_types) == 2 and all(play_type == PlayType.RUN for play_type in recent_types):
                return ["play_action", "intermediate_pass", "runs_edge"]
            return ["runs_inside", "runs_edge", "intermediate_pass"]
        if distance >= 5:
            return ["runs_inside", "short_pass", "runs_edge"]
        return ["runs_inside", "runs_edge", "short_pass"]

    if archetype == "vertical-attack":
        if distance >= 15:
            return ["deep_pass", "intermediate_pass", "play_action"]
        if distance >= 8:
            if len(recent_types) == 2 and all(play_type == PlayType.PASS for play_type in recent_types):
                return ["runs_edge", "short_pass", "intermediate_pass"]
            return ["intermediate_pass", "deep_pass", "short_pass"]
        return ["short_pass", "runs_edge", "deep_pass"]

    if distance >= 18:
        return ["intermediate_pass", "deep_pass", "short_pass"]
    if distance >= 11:
        return ["intermediate_pass", "short_pass", "runs_edge"]
    if distance >= 6:
        if len(recent_types) == 2 and all(play_type == PlayType.PASS for play_type in recent_types):
            return ["runs_edge", "short_pass", "intermediate_pass"]
        return ["short_pass", "intermediate_pass", "runs_edge"]
    return ["short_pass", "runs_edge", "runs_inside"]


def _counter_groups_for_defense(defense_identity: RoundDefenseIdentity, archetype: str, distance: int) -> list[str]:
    front_groups: list[str] = []
    if defense_identity.front_identity == "run_wall":
        front_groups.extend(["runs_edge", "short_pass", "intermediate_pass"])
        if distance >= 12:
            front_groups.append("play_action")
    elif defense_identity.front_identity == "light_box":
        front_groups.extend(["runs_inside", "runs_edge", "play_action"])
    elif defense_identity.front_identity == "pass_rush":
        front_groups.extend(["quick_pass", "runs_edge", "runs_inside"])
    else:
        front_groups.extend(["short_pass", "runs_edge"])

    coverage_groups: list[str] = []
    if defense_identity.coverage_identity == CoverageType.MAN:
        coverage_groups.append("man_beaters")
    elif defense_identity.coverage_identity == CoverageType.ZONE:
        coverage_groups.append("zone_beaters")
    else:
        coverage_groups.append("intermediate_pass")

    if defense_identity.front_identity in {"light_box", "pass_rush"}:
        groups = front_groups + coverage_groups
    else:
        groups = coverage_groups + front_groups

    if archetype == "ground-control" and defense_identity.front_identity == "light_box":
        groups = ["runs_inside", "runs_edge", "play_action", *coverage_groups, "short_pass"]
    elif archetype == "vertical-attack" and defense_identity.coverage_identity == CoverageType.ZONE:
        groups = ["intermediate_pass", "zone_beaters", *front_groups, "deep_pass"]

    if distance >= 15 and defense_identity.front_identity != "pass_rush":
        groups.append("deep_pass")
    return groups


def pick_demo_play(offense, archetype: str, defense_identity: RoundDefenseIdentity, state, recent_results):
    recent_ids = [result.offensive_play.id for result in recent_results[-2:]]
    recent_types = [result.offensive_play.play_type for result in recent_results[-2:]]
    groups = _play_groups(offense)

    def choose(group_names: list[str]):
        candidates = []
        for group_name in group_names:
            candidates.extend(groups.get(group_name, []))
        candidates = list(dict.fromkeys(candidates))
        selected = _pick_first_fresh(candidates, recent_ids)
        if selected is not None:
            return selected
        return next(iter(offense.offensive_playbook))

    base_groups = _base_group_preferences(archetype, state.distance, recent_types)
    counter_groups = _counter_groups_for_defense(defense_identity, archetype, state.distance)
    merged_groups = list(dict.fromkeys(counter_groups + base_groups))
    return choose(merged_groups)


def _prompt_for_play(offense, suggested_play):
    playbook = list(offense.offensive_playbook)
    print("Available plays:")
    for index, play in enumerate(playbook, start=1):
        concept_text = ", ".join(play.route_concepts) if play.route_concepts else play.run_location or play.run_gap or "base"
        print(f"  {index}. {play.name} [{play.play_type.value}] ({concept_text})")
    print(f"Suggested: {suggested_play.name}")

    while True:
        raw_value = input("Choose a play number (or press Enter for suggested): ").strip()
        if not raw_value:
            return suggested_play
        if raw_value.isdigit():
            index = int(raw_value)
            if 1 <= index <= len(playbook):
                return playbook[index - 1]
        print("Please enter a valid number from the list.")


def _print_math_summary(play, debug: dict[str, float]) -> None:
    if play.play_type == PlayType.PASS:
        print(
            "  Math: "
            f"pressure {_pct(debug['pressure_prob'])} | separation {_pct(debug['separation_prob'])}"
        )
        print(
            "  Math: "
            f"sack {_pct(debug['snap_sack_prob'])} | interception {_pct(debug['snap_interception_prob'])}"
            f" | completion {_pct(debug['snap_completion_prob'])} | explosive {_pct(debug['snap_explosive_prob'])}"
        )
        return

    print(
        "  Math: "
        f"line win {_pct(debug['line_win_prob'])} | rush success {_pct(debug['snap_rush_success_prob'])}"
        f" | evade {_pct(debug['snap_evade_prob'])}"
    )
    print(
        "  Math: "
        f"explosive {_pct(debug['snap_explosive_prob'])} | fumble {_pct(debug['snap_fumble_prob'])}"
    )


def _print_full_pass_math(debug: dict[str, float]) -> None:
    print(
        "  Matchup: "
        f"protection {_num(debug['protection_offense_score'])} vs pressure {_num(debug['pressure_defense_score'])}"
        f" | separation {_num(debug['separation_offense_score'])} vs coverage {_num(debug['coverage_defense_score'])}"
    )
    print(
        "  QB math: "
        f"accuracy {_num(debug['qb_accuracy_score'])} | decision {_num(debug['qb_decision_score'])}"
        f" | concept adj {_num(debug['concept_adjustment'])} | anti-spam {_num(debug['anti_spam_penalty'])}"
    )
    print(
        "  Chain: "
        f"sack {_pct(debug['sack_rating_prob'])} -> {_pct(debug['sack_model_prob'])} -> {_pct(debug['sack_baseline_prob'])} -> {_pct(debug['sack_prob'])}"
        f" (shift {_num(debug['sack_rating_shift'], 3)})"
    )
    print(
        "  Chain: "
        f"int {_pct(debug['interception_rating_prob'])} -> {_pct(debug['interception_model_prob'])} -> {_pct(debug['interception_baseline_prob'])} -> {_pct(debug['interception_prob'])}"
        f" (shift {_num(debug['interception_rating_shift'], 3)})"
    )
    print(
        "  Chain: "
        f"comp {_pct(debug['completion_rating_prob'])} -> {_pct(debug['completion_model_prob'])} -> {_pct(debug['completion_baseline_prob'])} -> {_pct(debug['completion_prob'])}"
        f" (shift {_num(debug['completion_rating_shift'], 3)})"
    )
    print(
        "  Chain: "
        f"explosive {_pct(debug['explosive_rating_prob'])} -> {_pct(debug['explosive_model_prob'])} -> {_pct(debug['explosive_baseline_prob'])} -> {_pct(debug['explosive_prob'])}"
        f" (shift {_num(debug['explosive_rating_shift'], 3)})"
    )
    print(
        "  Rolls: "
        f"pressure {_roll(debug['under_pressure_roll'])} | sack {_roll(debug['sack_roll'])}"
        f" | int {_roll(debug['interception_roll'])} | comp {_roll(debug['completion_roll'])}"
        f" | explosive {_roll(debug['explosive_roll'])}"
    )
    print(
        "  Yards: "
        f"air {int(debug['air_yards'])} | yac {int(debug['yac'])}"
        f" | pressure penalty {int(debug['pressure_penalty'])} | explosive bonus {int(debug['explosive_bonus'])}"
    )
    print(
        "  Repeats: "
        f"same play {int(debug['same_play_repeats'])} | same depth {int(debug['same_depth_repeats'])}"
        f" | same location {int(debug['same_location_repeats'])} | same type {int(debug['same_type_repeats'])}"
    )


def _print_full_run_math(debug: dict[str, float]) -> None:
    print(
        "  Matchup: "
        f"run block {_num(debug['run_block_score'])} vs run defense {_num(debug['run_defense_score'])}"
        f" | front adj {_num(debug['front_adjustment'])}"
    )
    print(
        "  Ball carrier: "
        f"rb skill {_num(debug['rb_skill_score'])} | pursuit+tackle {_num(debug['pursuit_tackling_score'])}"
        f" | anti-spam {_num(debug['anti_spam_penalty'])}"
    )
    print(
        "  Chain: "
        f"rush {_pct(debug['rush_success_rating_prob'])} -> {_pct(debug['rush_success_model_prob'])} -> {_pct(debug['rush_success_baseline_prob'])} -> {_pct(debug['rush_success_prob'])}"
        f" (shift {_num(debug['rush_success_rating_shift'], 3)})"
    )
    print(
        "  Chain: "
        f"explosive {_pct(debug['explosive_rating_prob'])} -> {_pct(debug['explosive_model_prob'])} -> {_pct(debug['explosive_baseline_prob'])} -> {_pct(debug['explosive_prob'])}"
        f" (shift {_num(debug['explosive_rating_shift'], 3)})"
    )
    print(
        "  Chain: "
        f"fumble {_pct(debug['fumble_rating_prob'])} -> {_pct(debug['fumble_model_prob'])} -> {_pct(debug['fumble_baseline_prob'])} -> {_pct(debug['fumble_prob'])}"
        f" (shift {_num(debug['fumble_rating_shift'], 3)})"
    )
    print(
        "  Rolls: "
        f"rush {_roll(debug['rush_success_roll'])} | evade {_roll(debug['evade_roll'])}"
        f" | explosive {_roll(debug['explosive_roll'])} | fumble {_roll(debug['fumble_roll'])}"
    )
    print(
        "  Yards: "
        f"base {int(debug['base_yards'])} | evade bonus {int(debug['evade_bonus'])}"
        f" | explosive bonus {int(debug['explosive_bonus'])}"
    )
    print(
        "  Repeats: "
        f"same play {int(debug['same_play_repeats'])} | same lane {int(debug['same_lane_repeats'])}"
        f" | same gap {int(debug['same_gap_repeats'])} | same type {int(debug['same_type_repeats'])}"
    )


def _print_defense_debug(defense_debug: dict[str, object]) -> None:
    print(
        "  Defense math: "
        f"same play {int(defense_debug['same_play_repeats'])} | same type {int(defense_debug['same_type_repeats'])}"
        f" | adaptation x{float(defense_debug['adaptation_multiplier']):.2f}"
        f" | selection roll {_roll(float(defense_debug['selection_roll']))}/{_num(float(defense_debug['selection_total_weight']))}"
    )
    print(
        "  Defense read: "
        f"run {_pct(float(defense_debug['observed_run_rate']))}"
        f" | short {_pct(float(defense_debug['observed_short_rate']))}"
        f" | deep {_pct(float(defense_debug['observed_deep_rate']))}"
        f" | comp {_pct(float(defense_debug['observed_completion_rate']))}"
    )
    print("  Defense weights:")
    for weighted_call in defense_debug["weighted_calls"]:
        print(
            "    "
            f"{weighted_call['play_name']}: {float(weighted_call['weight']):.3f}"
            f" | {weighted_call['coverage_type']} | {weighted_call['rushers']} rushers | front={weighted_call['front']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a roguelite-style single-round football simulation.")
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
        help="Prototype defense identity to line up across from the offense.",
    )
    parser.add_argument("--seed", type=int, default=23, help="Deterministic RNG seed for the round.")
    parser.add_argument("--target-yards", type=int, default=40, help="How many yards the offense must gain to score.")
    parser.add_argument("--play-budget", type=int, default=6, help="How many total snaps the round allows.")
    parser.add_argument("--interactive", action="store_true", help="Prompt for the offensive play each snap.")
    parser.add_argument("--show-full-math", action="store_true", help="Show full matchup ingredients and probability chains.")
    args = parser.parse_args()

    offense = build_starter_offense(args.archetype)
    defense, defense_identity = build_defense_matchup(args.defense_identity)
    config = build_demo_round_config(args.target_yards, args.play_budget, defense_identity)

    state = build_round_start_state(offense.id, config.target_yards)
    rng = random.Random(args.seed)
    history = []
    outcome = "PLAY_BUDGET_EXHAUSTED"

    print("=== Roguelite Round Prototype ===")
    print(f"Round: {config.name}")
    print(f"Offense: {offense.name}")
    print(f"Defense: {defense.name} ({config.defense_identity.name})")
    print(f"Target: Score from {config.target_yards} yards out in {config.play_budget} plays")
    print(f"Defense strength: {config.defense_identity.strength_note}")
    print(f"Defense weakness: {config.defense_identity.weakness_note}")
    print(f"Defense identity tell: {config.defense_identity.tell}")

    for _ in range(config.play_budget):
        plays_left = config.play_budget - len(history)
        suggested_play = pick_demo_play(offense, args.archetype, config.defense_identity, state, history)
        offensive_play = _prompt_for_play(offense, suggested_play) if args.interactive else suggested_play
        snap_seed = rng.randint(0, 2_147_483_647)
        snap = simulate_round_play(
            state,
            offense,
            defense,
            offensive_play,
            config.defense_identity,
            rng_seed=snap_seed,
            recent_results=tuple(history),
        )
        result = snap.play_result
        debug = result.debug

        print("")
        print(
            f"Play {len(history) + 1:02d} | Q{state.quarter} {_format_clock(state.clock_seconds)}"
            f" | {state.down}&{state.distance} @ {state.yard_line}"
            f" | Plays left: {plays_left}"
        )
        print(f"  Call: {offensive_play.name}")
        print(f"  Pre-snap tell: {snap.defensive_tell}")
        print(f"  Defense call: {result.defensive_play.name}")
        for event in result.events[2:]:
            print(f"  Note: {event}")
        _print_math_summary(offensive_play, debug)
        if args.show_full_math:
            if offensive_play.play_type == PlayType.PASS:
                _print_full_pass_math(debug)
            else:
                _print_full_run_math(debug)
            _print_defense_debug(snap.defense_debug)
        print(f"  Result: {result.summary} {_yards_text(result.yards_gained)}")

        history.append(result)
        state = result.next_state

        if result.touchdown:
            outcome = "TOUCHDOWN"
            break
        if result.turnover:
            outcome = "TURNOVER"
            break

        print(f"  Now: {state.down}&{state.distance} @ {state.yard_line} | Plays remaining: {config.play_budget - len(history)}")

    print("")
    print("=== Round Summary ===")
    print(f"Outcome: {outcome}")
    print(f"Plays used: {len(history)} / {config.play_budget}")
    print(f"Yards remaining: {max(0, state.distance)}")
    print(f"Score (offense view): {state.offense_score}-{state.defense_score}")
    print(f"End clock: Q{state.quarter} {_format_clock(state.clock_seconds)}")


if __name__ == "__main__":
    main()
