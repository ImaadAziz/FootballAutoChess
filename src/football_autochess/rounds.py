from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Sequence

from .ml_models import ModelBundle
from .models import CoverageType, DefensivePlay, GameState, OffensivePlay, PlayResult, PlayType, SimulationTuning, Team
from .simulation import simulate_called_play

_MAN_BEATER_CONCEPTS = frozenset({"drag", "go", "post", "slant"})
_ZONE_BEATER_CONCEPTS = frozenset({"curl", "dig", "flat", "out", "sail", "stick"})


@dataclass(frozen=True)
class RoundDefenseIdentity:
    name: str
    front_identity: str
    coverage_identity: CoverageType
    tell: str
    strength_note: str
    weakness_note: str
    adaptation_rate: float = 0.35


@dataclass(frozen=True)
class RoundConfig:
    name: str
    target_yards: int
    play_budget: int
    defense_identity: RoundDefenseIdentity


@dataclass(frozen=True)
class RoundSnapResult:
    defensive_tell: str
    play_result: PlayResult
    defense_debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoundResult:
    config: RoundConfig
    start_state: GameState
    end_state: GameState
    snaps: tuple[RoundSnapResult, ...]
    outcome: str
    plays_used: int
    plays_remaining: int
    yards_remaining: int


@dataclass(frozen=True)
class _ObservedOffenseSnapshot:
    run_rate: float
    short_rate: float
    deep_rate: float
    completion_rate: float
    same_play_repeats: int
    same_type_repeats: int


def build_round_start_state(
    offense_team_id: str,
    target_yards: int,
    *,
    quarter: int = 1,
    clock_seconds: int = 900,
) -> GameState:
    if target_yards <= 0 or target_yards >= 100:
        raise ValueError("target_yards must be between 1 and 99")

    return GameState(
        possession_team_id=offense_team_id,
        down=1,
        distance=target_yards,
        yard_line=100 - target_yards,
        quarter=quarter,
        clock_seconds=clock_seconds,
    )


def find_offensive_play(team: Team, play_key: str) -> OffensivePlay:
    normalized = play_key.strip().lower()
    for play in team.offensive_playbook:
        if play.id.lower() == normalized or play.name.lower() == normalized:
            return play
    raise ValueError(f"Unknown offensive play: {play_key}")


def _advance_round_state(
    state: GameState,
    yards_gained: int,
    turnover: bool,
    touchdown: bool,
    offense_team_id: str,
    defense_team_id: str,
    play_seconds: int,
) -> tuple[GameState, bool]:
    new_clock = max(0, state.clock_seconds - play_seconds)
    new_yard_line = max(1, min(100, state.yard_line + yards_gained))

    if turnover:
        next_state = GameState(
            possession_team_id=defense_team_id,
            down=1,
            distance=max(1, 100 - new_yard_line),
            yard_line=max(1, 100 - new_yard_line),
            quarter=state.quarter,
            clock_seconds=new_clock,
            offense_score=state.offense_score,
            defense_score=state.defense_score,
            drive_number=state.drive_number + 1,
            season=state.season,
            week=state.week,
            season_type=state.season_type,
        )
        return next_state, False

    distance_left = state.distance - yards_gained
    if touchdown or new_yard_line >= 100 or distance_left <= 0:
        next_state = GameState(
            possession_team_id=defense_team_id,
            down=1,
            distance=0,
            yard_line=25,
            quarter=state.quarter,
            clock_seconds=new_clock,
            offense_score=state.offense_score + 6,
            defense_score=state.defense_score,
            drive_number=state.drive_number + 1,
            season=state.season,
            week=state.week,
            season_type=state.season_type,
        )
        return next_state, False

    next_state = GameState(
        possession_team_id=offense_team_id,
        down=state.down + 1,
        distance=max(1, distance_left),
        yard_line=new_yard_line,
        quarter=state.quarter,
        clock_seconds=new_clock,
        offense_score=state.offense_score,
        defense_score=state.defense_score,
        drive_number=state.drive_number,
        season=state.season,
        week=state.week,
        season_type=state.season_type,
    )
    return next_state, False


def _weighted_choice_with_roll(
    items: Sequence[DefensivePlay],
    weights: Sequence[float],
    rng: random.Random,
) -> tuple[DefensivePlay, float, float]:
    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0:
        return rng.choice(list(items)), -1.0, total

    roll = rng.random() * total
    running = 0.0
    for item, weight in zip(items, weights):
        running += max(0.0, weight)
        if running >= roll:
            return item, roll, total
    return items[-1], roll, total


def _weighted_choice(items: Sequence[DefensivePlay], weights: Sequence[float], rng: random.Random) -> DefensivePlay:
    choice, _, _ = _weighted_choice_with_roll(items, weights, rng)
    return choice


def _play_concepts(offensive_play: OffensivePlay) -> set[str]:
    return {concept.lower() for concept in offensive_play.route_concepts}


def _is_man_beater(offensive_play: OffensivePlay) -> bool:
    return bool(_play_concepts(offensive_play) & _MAN_BEATER_CONCEPTS)


def _is_zone_beater(offensive_play: OffensivePlay) -> bool:
    return bool(_play_concepts(offensive_play) & _ZONE_BEATER_CONCEPTS)


def _streak_repeats(recent_results: Sequence[PlayResult], key_getter) -> int:
    if not recent_results:
        return 0

    last_value = key_getter(recent_results[-1])
    streak = 0
    for result in reversed(recent_results):
        if key_getter(result) != last_value:
            break
        streak += 1
    return max(0, streak - 1)


def _observed_offense_snapshot(
    offense_team: Team,
    recent_results: Sequence[PlayResult],
) -> _ObservedOffenseSnapshot:
    sample = list(recent_results[-4:])
    if not sample:
        pass_bias = float(offense_team.tendencies.get("pass_bias", 0.55))
        deep_bias = float(offense_team.tendencies.get("deep_shot_rate", 0.24))
        run_rate = max(0.05, min(0.95, 1.0 - pass_bias))
        short_rate = max(0.25, min(0.75, 0.55 - (deep_bias * 0.25)))
        return _ObservedOffenseSnapshot(
            run_rate=run_rate,
            short_rate=short_rate,
            deep_rate=max(0.08, min(0.55, deep_bias)),
            completion_rate=0.52,
            same_play_repeats=0,
            same_type_repeats=0,
        )

    pass_plays = [result for result in sample if result.offensive_play.play_type == PlayType.PASS]
    run_plays = [result for result in sample if result.offensive_play.play_type == PlayType.RUN]
    if pass_plays:
        short_rate = sum(1 for result in pass_plays if result.offensive_play.target_depth <= 7) / len(pass_plays)
        deep_rate = sum(1 for result in pass_plays if result.offensive_play.target_depth >= 15) / len(pass_plays)
        completion_rate = sum(1 for result in pass_plays if result.complete) / len(pass_plays)
    else:
        short_rate = 0.34
        deep_rate = 0.22
        completion_rate = 0.0

    return _ObservedOffenseSnapshot(
        run_rate=len(run_plays) / len(sample),
        short_rate=short_rate,
        deep_rate=deep_rate,
        completion_rate=completion_rate,
        same_play_repeats=_streak_repeats(sample, lambda result: result.offensive_play.id),
        same_type_repeats=_streak_repeats(sample, lambda result: result.offensive_play.play_type),
    )


def _select_round_defensive_play_with_debug(
    game_state: GameState,
    offense_team: Team,
    defense_team: Team,
    defense_identity: RoundDefenseIdentity,
    recent_results: Sequence[PlayResult],
    rng: random.Random,
) -> tuple[DefensivePlay, dict[str, Any]]:
    if not defense_team.defensive_playbook:
        raise ValueError(f"{defense_team.name} has no defensive plays")

    snapshot = _observed_offense_snapshot(offense_team, recent_results)
    same_play_repeats = snapshot.same_play_repeats
    same_type_repeats = snapshot.same_type_repeats
    adaptation_multiplier = 1.0 + (
        defense_identity.adaptation_rate
        * (
            (same_play_repeats * 0.75)
            + (same_type_repeats * 0.45)
            + (max(0.0, snapshot.run_rate - 0.45) * 1.25)
            + (max(0.0, snapshot.deep_rate - 0.28) * 1.1)
            + (max(0.0, snapshot.short_rate - 0.55) * 0.9)
        )
    )

    weights: list[float] = []
    weighted_calls: list[dict[str, Any]] = []
    for play in defense_team.defensive_playbook:
        weight = 1.0
        front = play.front.lower()

        if defense_identity.front_identity == "run_wall":
            if play.rushers >= 5:
                weight *= 1.7
            if "bear" in front:
                weight *= 1.35
            if snapshot.run_rate >= 0.4:
                weight *= 1.12
            if snapshot.run_rate >= 0.55 and play.rushers >= 5:
                weight *= adaptation_multiplier
        elif defense_identity.front_identity == "pass_rush":
            if play.rushers >= 5:
                weight *= 1.7
            if snapshot.run_rate <= 0.38:
                weight *= 1.15
        elif defense_identity.front_identity == "light_box":
            if play.rushers <= 4:
                weight *= 1.55
            if snapshot.run_rate >= 0.45:
                weight *= 0.72
        else:
            if play.rushers == 4:
                weight *= 1.1

        if defense_identity.coverage_identity == CoverageType.MAN:
            if play.coverage_type == CoverageType.MAN:
                weight *= 1.7
            elif play.coverage_type == CoverageType.ZONE:
                weight *= 0.8

            if snapshot.short_rate >= 0.5 and snapshot.completion_rate >= 0.55 and play.coverage_type == CoverageType.MAN:
                weight *= adaptation_multiplier
        elif defense_identity.coverage_identity == CoverageType.ZONE:
            if play.coverage_type == CoverageType.ZONE:
                weight *= 1.7
            elif play.coverage_type == CoverageType.MAN:
                weight *= 0.8

            if snapshot.deep_rate >= 0.28 and play.coverage_type == CoverageType.ZONE:
                weight *= adaptation_multiplier
        else:
            if play.coverage_type == CoverageType.MIXED:
                weight *= 1.35

        if same_play_repeats >= 1 and play.rushers >= 5:
            weight *= 1.08
        if same_type_repeats >= 1 and play.coverage_type == defense_identity.coverage_identity:
            weight *= 1.05

        if game_state.distance <= 6 and play.rushers >= 5:
            weight *= 1.15
        elif game_state.distance >= 18 and play.coverage_type == CoverageType.ZONE:
            weight *= 1.14
        if snapshot.run_rate <= 0.25 and play.rushers <= 4:
            weight *= 1.08

        final_weight = max(0.05, weight)
        weights.append(final_weight)
        weighted_calls.append(
            {
                "play_id": play.id,
                "play_name": play.name,
                "coverage_type": play.coverage_type.value,
                "rushers": play.rushers,
                "front": play.front,
                "weight": round(final_weight, 4),
            }
        )

    selected_play, selection_roll, total_weight = _weighted_choice_with_roll(defense_team.defensive_playbook, weights, rng)
    defense_debug = {
        "same_play_repeats": same_play_repeats,
        "same_type_repeats": same_type_repeats,
        "observed_run_rate": round(snapshot.run_rate, 4),
        "observed_short_rate": round(snapshot.short_rate, 4),
        "observed_deep_rate": round(snapshot.deep_rate, 4),
        "observed_completion_rate": round(snapshot.completion_rate, 4),
        "adaptation_multiplier": round(adaptation_multiplier, 4),
        "selection_roll": round(selection_roll, 4),
        "selection_total_weight": round(total_weight, 4),
        "selected_defense_call": selected_play.name,
        "weighted_calls": tuple(weighted_calls),
    }
    return selected_play, defense_debug


def select_round_defensive_play(
    game_state: GameState,
    offense_team: Team,
    defense_team: Team,
    defense_identity: RoundDefenseIdentity,
    recent_results: Sequence[PlayResult],
    rng: random.Random,
) -> DefensivePlay:
    selected_play, _ = _select_round_defensive_play_with_debug(
        game_state,
        offense_team,
        defense_team,
        defense_identity,
        recent_results,
        rng,
    )
    return selected_play


def describe_defensive_look(defensive_play: DefensivePlay, defense_identity: RoundDefenseIdentity) -> str:
    front = defensive_play.front.lower()

    if defense_identity.front_identity == "run_wall" or defensive_play.rushers >= 5:
        box_text = "Packed box"
    elif defense_identity.front_identity == "light_box" and defensive_play.rushers <= 4:
        box_text = "Light box"
    else:
        box_text = "Balanced box"

    if "bear" in front:
        front_text = "interior defenders crowd both A-gaps"
    elif defensive_play.rushers >= 5:
        front_text = "pressure threats walked toward the line"
    else:
        front_text = "front stays even before the snap"

    if defensive_play.coverage_type == CoverageType.MAN:
        coverage_text = "corners show man leverage"
    elif defensive_play.coverage_type == CoverageType.ZONE:
        coverage_text = "safeties split high over a zone shell"
    else:
        coverage_text = "secondary rotates late after the snap"

    return f"{box_text}; {front_text}; {coverage_text}"


def simulate_round_play(
    game_state: GameState,
    offense_team: Team,
    defense_team: Team,
    offensive_play: OffensivePlay,
    defense_identity: RoundDefenseIdentity,
    *,
    rng_seed: int | None = None,
    recent_results: Sequence[PlayResult] | None = None,
    model_bundle: ModelBundle | None = None,
    tuning: SimulationTuning | None = None,
) -> RoundSnapResult:
    rng = random.Random(rng_seed)
    defense_play, defense_debug = _select_round_defensive_play_with_debug(
        game_state,
        offense_team,
        defense_team,
        defense_identity,
        tuple(recent_results or ()),
        rng,
    )
    defensive_tell = describe_defensive_look(defense_play, defense_identity)
    play_seed = rng.randint(0, 2_147_483_647)
    play_result = simulate_called_play(
        game_state,
        offense_team,
        defense_team,
        offensive_play,
        defensive_play=defense_play,
        rng_seed=play_seed,
        recent_results=recent_results,
        model_bundle=model_bundle,
        tuning=tuning,
        state_advancer=_advance_round_state,
    )
    return RoundSnapResult(defensive_tell=defensive_tell, play_result=play_result, defense_debug=defense_debug)


def simulate_round(
    config: RoundConfig,
    offense_team: Team,
    defense_team: Team,
    play_calls: Sequence[str | OffensivePlay],
    *,
    rng_seed: int | None = None,
    model_bundle: ModelBundle | None = None,
    tuning: SimulationTuning | None = None,
) -> RoundResult:
    if len(play_calls) < config.play_budget:
        raise ValueError("play_calls must contain at least play_budget calls")

    rng = random.Random(rng_seed)
    state = build_round_start_state(offense_team.id, config.target_yards)
    start_state = state
    snaps: list[RoundSnapResult] = []
    outcome = "PLAY_BUDGET_EXHAUSTED"

    for index in range(config.play_budget):
        play_call = play_calls[index]
        offensive_play = play_call if isinstance(play_call, OffensivePlay) else find_offensive_play(offense_team, play_call)
        snap_seed = rng.randint(0, 2_147_483_647)
        snap = simulate_round_play(
            state,
            offense_team,
            defense_team,
            offensive_play,
            config.defense_identity,
            rng_seed=snap_seed,
            recent_results=tuple(result.play_result for result in snaps),
            model_bundle=model_bundle,
            tuning=tuning,
        )
        snaps.append(snap)
        state = snap.play_result.next_state

        if snap.play_result.touchdown:
            outcome = "TOUCHDOWN"
            break
        if snap.play_result.turnover:
            outcome = "TURNOVER"
            break

    return RoundResult(
        config=config,
        start_state=start_state,
        end_state=state,
        snaps=tuple(snaps),
        outcome=outcome,
        plays_used=len(snaps),
        plays_remaining=max(0, config.play_budget - len(snaps)),
        yards_remaining=max(0, state.distance),
    )
