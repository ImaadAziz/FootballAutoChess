from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, replace
from typing import Sequence, TypeVar

from .ml_models import ModelBundle
from .models import (
    BatchSimulationResult,
    CoverageType,
    DefensivePlay,
    DriveResult,
    GameState,
    OffensivePlay,
    PlayResult,
    PlayType,
    Position,
    Team,
)


@dataclass(frozen=True)
class _OffenseTendencySnapshot:
    short_rate: float
    deep_rate: float
    completion_rate: float
    run_rate: float


T = TypeVar("T")


def _avg_rating(players: tuple, key: str, default: float = 50.0) -> float:
    if not players:
        return default
    return sum(player.rating(key, default) for player in players) / len(players)


def _avg_rating_with_fallback(
    players: tuple,
    primary_key: str,
    fallback_key: str | None = None,
    default: float = 50.0,
) -> float:
    if not players:
        return default

    values: list[float] = []
    for player in players:
        if primary_key in player.ratings:
            values.append(float(player.ratings[primary_key]))
            continue
        if fallback_key and fallback_key in player.ratings:
            values.append(float(player.ratings[fallback_key]))
            continue
        values.append(default)

    return sum(values) / len(values)


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(probability: float) -> float:
    bounded = _clamp(probability, 0.001, 0.999)
    return math.log(bounded / (1.0 - bounded))


def _contest_probability(offense_score: float, defense_score: float, rng: random.Random) -> float:
    noise = rng.gauss(0.0, 6.0)
    contest = ((offense_score - defense_score) + noise) / 12.0
    return _clamp(_sigmoid(contest), 0.01, 0.99)


def _apply_rating_delta(
    baseline_probability: float,
    rating_advantage: float,
    scale: float,
    max_logit_shift: float = 1.35,
) -> tuple[float, float]:
    shift = _clamp(rating_advantage / scale, -max_logit_shift, max_logit_shift)
    blended = _clamp(_sigmoid(_logit(baseline_probability) + shift), 0.001, 0.999)
    return blended, shift


def _resolve_probability(
    model_bundle: ModelBundle | None,
    event: str,
    features: dict[str, float],
    fallback_probability: float,
    rating_advantage: float,
    scale: float,
    max_logit_shift: float = 1.35,
) -> tuple[float, float, float]:
    if model_bundle is None:
        return fallback_probability, fallback_probability, 0.0

    baseline_probability = model_bundle.predict_event_probability(event, features, fallback_probability)
    blended_probability, logit_shift = _apply_rating_delta(
        baseline_probability,
        rating_advantage,
        scale,
        max_logit_shift=max_logit_shift,
    )
    return baseline_probability, blended_probability, logit_shift


def _weighted_choice(items: Sequence[T], weights: Sequence[float], rng: random.Random) -> T:
    if not items:
        raise ValueError("Cannot choose from an empty sequence")

    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0:
        return rng.choice(list(items))

    roll = rng.random() * total
    running = 0.0
    for item, weight in zip(items, weights):
        running += max(0.0, weight)
        if running >= roll:
            return item

    return items[-1]


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _team_tendency(team: Team, key: str, default: float) -> float:
    return float(team.tendencies.get(key, default))


def _goal_to_go(game_state: GameState) -> float:
    yards_to_goal = max(1, 100 - game_state.yard_line)
    return 1.0 if game_state.distance >= yards_to_goal else 0.0


def _base_context_features(game_state: GameState) -> dict[str, float]:
    return {
        "num:down_norm": game_state.down / 4.0,
        "num:distance_norm": min(game_state.distance, 25) / 25.0,
        "num:yardline_norm": min(max(game_state.yard_line, 1), 99) / 100.0,
        "num:quarter_norm": min(max(game_state.quarter, 1), 5) / 5.0,
        "num:score_diff_norm": _clamp(game_state.offense_score - game_state.defense_score, -28.0, 28.0) / 28.0,
        "num:game_clock_norm": min(max(game_state.clock_seconds, 0), 3600) / 3600.0,
        "num:goal_to_go": _goal_to_go(game_state),
        "num:shotgun": 0.5,
        "num:no_huddle": 0.0,
    }


def _recent_offense_tendencies(recent_results: Sequence[PlayResult], lookback: int = 6) -> _OffenseTendencySnapshot:
    sample = list(recent_results[-lookback:])
    if not sample:
        return _OffenseTendencySnapshot(short_rate=0.34, deep_rate=0.26, completion_rate=0.52, run_rate=0.32)

    pass_plays = [result for result in sample if result.offensive_play.play_type == PlayType.PASS]
    run_plays = [result for result in sample if result.offensive_play.play_type == PlayType.RUN]

    if pass_plays:
        short = sum(1 for result in pass_plays if result.offensive_play.target_depth <= 7)
        deep = sum(1 for result in pass_plays if result.offensive_play.target_depth >= 15)
        complete = sum(1 for result in pass_plays if result.complete)
        short_rate = short / len(pass_plays)
        deep_rate = deep / len(pass_plays)
        completion_rate = complete / len(pass_plays)
    else:
        short_rate = 0.34
        deep_rate = 0.26
        completion_rate = 0.52

    run_rate = len(run_plays) / len(sample)

    return _OffenseTendencySnapshot(
        short_rate=short_rate,
        deep_rate=deep_rate,
        completion_rate=completion_rate,
        run_rate=run_rate,
    )


def _select_offensive_play(
    game_state: GameState,
    offense_team: Team,
    recent_results: Sequence[PlayResult],
    rng: random.Random,
    model_bundle: ModelBundle | None,
) -> OffensivePlay:
    playbook = offense_team.offensive_playbook
    tendencies = _recent_offense_tendencies(recent_results)

    aggression = _clamp(_team_tendency(offense_team, "aggressiveness", 0.5), 0.0, 1.0)
    pass_bias = _clamp(_team_tendency(offense_team, "pass_bias", 0.65), 0.05, 0.95)
    deep_bias = _clamp(_team_tendency(offense_team, "deep_shot_rate", 0.3), 0.0, 1.0)

    playcall_context = _base_context_features(game_state)
    playcall_context["num:shotgun"] = pass_bias
    playcall_context["num:no_huddle"] = _clamp(_team_tendency(offense_team, "tempo", 0.5), 0.0, 1.0)

    predicted_pass_bias = pass_bias
    if model_bundle is not None:
        predicted_pass_bias = model_bundle.predict_pass_call_probability(playcall_context, pass_bias)

    effective_pass_bias = _clamp((0.6 * pass_bias) + (0.4 * predicted_pass_bias), 0.05, 0.95)

    desired_depth = float(game_state.distance) * (0.85 + aggression * 0.45)
    if game_state.down <= 2:
        desired_depth *= 0.9
    if game_state.yard_line >= 85:
        desired_depth *= 0.8
    desired_depth = _clamp(desired_depth, 3.0, 22.0)

    weights: list[float] = []
    for play in playbook:
        weight = 1.0

        if play.play_type == PlayType.PASS:
            depth = play.target_depth
            weight *= 0.5 + effective_pass_bias

            depth_fit = 1.3 - min(0.85, abs(depth - desired_depth) / 20.0)
            weight *= max(0.1, depth_fit)

            if game_state.down in (3, 4):
                sticks_gap = game_state.distance - depth
                if sticks_gap > 3:
                    weight *= 0.72
                elif -3 <= sticks_gap <= 3:
                    weight *= 1.3 + (aggression * 0.2)

            if game_state.distance <= 3 and depth > 14:
                weight *= 0.75
            if game_state.distance >= 12 and depth < 8:
                weight *= 0.8

            if depth >= 15:
                weight *= 0.8 + deep_bias
                if tendencies.deep_rate > 0.45:
                    weight *= 0.78
            if depth <= 7 and tendencies.short_rate > 0.55:
                weight *= 0.82

            if game_state.yard_line >= 80 and depth >= 18:
                weight *= 0.6

            if tendencies.run_rate > 0.55:
                weight *= 1.06

        else:
            run_bias = 1.0 - effective_pass_bias
            weight *= 0.5 + run_bias

            if game_state.distance <= 2:
                weight *= 1.35
            if game_state.down <= 2 and game_state.distance <= 6:
                weight *= 1.25
            if game_state.down in (3, 4) and game_state.distance >= 7:
                weight *= 0.55
            if game_state.yard_line >= 80:
                weight *= 1.2
            if game_state.yard_line <= 20:
                weight *= 0.88

            if tendencies.run_rate > 0.52:
                weight *= 0.82
            if tendencies.deep_rate > 0.4:
                weight *= 1.1

        if recent_results and recent_results[-1].offensive_play.id == play.id:
            weight *= 0.7

        weights.append(max(0.05, weight))

    return _weighted_choice(playbook, weights, rng)


def _select_defensive_play(
    game_state: GameState,
    defense_team: Team,
    recent_results: Sequence[PlayResult],
    rng: random.Random,
) -> DefensivePlay:
    playbook = defense_team.defensive_playbook
    tendencies = _recent_offense_tendencies(recent_results)
    blitz_rate = _clamp(_team_tendency(defense_team, "blitz_rate", 0.35), 0.0, 1.0)

    weights: list[float] = []
    for play in playbook:
        weight = 1.0

        if play.rushers >= 5:
            weight *= 0.7 + (blitz_rate * 1.3)
        else:
            weight *= 1.2 - (blitz_rate * 0.35)

        if game_state.distance >= 10:
            if play.coverage_type == CoverageType.ZONE:
                weight *= 1.22
            if play.rushers >= 5:
                weight *= 1.14
        elif game_state.distance <= 3:
            if play.coverage_type == CoverageType.MAN:
                weight *= 1.18
            if play.rushers >= 5:
                weight *= 0.88

        if tendencies.deep_rate >= 0.35:
            if play.coverage_type == CoverageType.ZONE:
                weight *= 1.28
            if play.coverage_type == CoverageType.MAN:
                weight *= 0.9

        if tendencies.short_rate >= 0.5 and tendencies.completion_rate >= 0.55:
            if play.rushers >= 5:
                weight *= 1.22
            if play.coverage_type == CoverageType.MAN:
                weight *= 1.1

        if tendencies.run_rate >= 0.42:
            if play.rushers >= 5:
                weight *= 1.18
            if play.coverage_type == CoverageType.MAN:
                weight *= 1.08

        if game_state.yard_line >= 85:
            if play.coverage_type == CoverageType.ZONE:
                weight *= 1.2
            if play.rushers >= 5:
                weight *= 0.85

        weights.append(max(0.05, weight))

    return _weighted_choice(playbook, weights, rng)


def _advance_state(
    state: GameState,
    yards_gained: int,
    turnover: bool,
    touchdown: bool,
    offense_team_id: str,
    defense_team_id: str,
    play_seconds: int,
) -> tuple[GameState, bool]:
    new_clock = max(0, state.clock_seconds - play_seconds)

    if turnover:
        turnover_spot = max(1, min(100, state.yard_line + yards_gained))
        next_state = replace(
            state,
            possession_team_id=defense_team_id,
            down=1,
            distance=10,
            yard_line=max(1, 100 - turnover_spot),
            drive_number=state.drive_number + 1,
            clock_seconds=new_clock,
        )
        return next_state, False

    new_yard_line = max(1, min(100, state.yard_line + yards_gained))
    distance_left = state.distance - yards_gained
    first_down = distance_left <= 0

    if touchdown or new_yard_line >= 100:
        next_state = replace(
            state,
            possession_team_id=defense_team_id,
            down=1,
            distance=10,
            yard_line=25,
            offense_score=state.offense_score + 6,
            drive_number=state.drive_number + 1,
            clock_seconds=new_clock,
        )
        return next_state, False

    if first_down:
        next_state = replace(
            state,
            possession_team_id=offense_team_id,
            down=1,
            distance=10,
            yard_line=new_yard_line,
            clock_seconds=new_clock,
        )
        return next_state, True

    new_down = state.down + 1
    if new_down > 4:
        next_state = replace(
            state,
            possession_team_id=defense_team_id,
            down=1,
            distance=10,
            yard_line=max(1, 100 - new_yard_line),
            drive_number=state.drive_number + 1,
            clock_seconds=new_clock,
        )
        return next_state, False

    next_state = replace(
        state,
        possession_team_id=offense_team_id,
        down=new_down,
        distance=max(1, distance_left),
        yard_line=new_yard_line,
        clock_seconds=new_clock,
    )
    return next_state, False


def _resolve_pass_play(
    game_state: GameState,
    offensive_play: OffensivePlay,
    defensive_play: DefensivePlay,
    offense_team: Team,
    defense_team: Team,
    rng: random.Random,
    model_bundle: ModelBundle | None,
) -> dict[str, float | bool | int | str | list[str]]:
    qb = offense_team.require_player(Position.QB)
    ol = offense_team.players_by_position(Position.OL)
    receivers = offense_team.players_by_position(Position.WR, Position.TE, Position.RB)

    rushers = defense_team.players_by_position(Position.DL, Position.EDGE, Position.LB)
    coverage_players = defense_team.players_by_position(Position.CB, Position.S, Position.LB)

    protection_offense = 0.65 * _avg_rating_with_fallback(ol, "pass_block", default=55.0) + 0.35 * qb.rating("pocket_awareness")
    if offensive_play.protection_call.lower() in {"max", "max_protect"}:
        protection_offense += 3.0

    pressure_defense = 0.7 * _avg_rating_with_fallback(rushers, "pass_rush_power", default=55.0) + 0.3 * _avg_rating_with_fallback(rushers, "get_off", default=55.0)
    pressure_defense += max(0, defensive_play.rushers - 4) * 2.3

    pressure_prob = 1.0 - _contest_probability(protection_offense, pressure_defense, rng)
    under_pressure = rng.random() < pressure_prob

    separation_offense = (
        0.55 * _avg_rating_with_fallback(receivers, "route_running", default=55.0)
        + 0.3 * _avg_rating_with_fallback(receivers, "release", default=55.0)
        + 0.15 * _avg_rating_with_fallback(receivers, "speed", default=55.0)
    )

    coverage_defense = (
        0.45 * _avg_rating_with_fallback(coverage_players, "man_coverage", default=55.0)
        + 0.4 * _avg_rating_with_fallback(coverage_players, "zone_coverage", default=55.0)
        + 0.15 * _avg_rating_with_fallback(coverage_players, "play_recognition", default=55.0)
    )

    if defensive_play.coverage_type == CoverageType.MAN:
        coverage_defense += _avg_rating_with_fallback(coverage_players, "man_coverage", default=55.0) * 0.06
    elif defensive_play.coverage_type == CoverageType.ZONE:
        coverage_defense += _avg_rating_with_fallback(coverage_players, "zone_coverage", default=55.0) * 0.06

    if offensive_play.target_depth >= 15 and defensive_play.coverage_type == CoverageType.ZONE:
        coverage_defense += 2.0
    if offensive_play.target_depth <= 7 and defensive_play.coverage_type == CoverageType.MAN:
        coverage_defense += 1.5

    separation_prob = _contest_probability(separation_offense, coverage_defense, rng)

    qb_accuracy = (
        qb.rating("accuracy_short") * 0.45
        + qb.rating("accuracy_mid") * 0.35
        + qb.rating("accuracy_deep") * 0.2
    )
    qb_accuracy -= max(0.0, offensive_play.target_depth - 8.0) * 0.85

    qb_decision = qb.rating("decision") - (12.0 if under_pressure else 0.0)
    rating_completion_prob = _contest_probability(qb_accuracy + qb_decision * 0.25, coverage_defense, rng)
    rating_completion_prob *= 0.63 + (0.37 * separation_prob)

    rating_sack_prob = _clamp(pressure_prob * (0.25 + 0.045 * max(0, defensive_play.rushers - 4)), 0.02, 0.55)

    rating_interception_prob = (1.0 - rating_completion_prob) * 0.14
    rating_interception_prob += 0.035 if under_pressure else 0.0
    rating_interception_prob += 0.018 if offensive_play.target_depth >= 15 else 0.0
    rating_interception_prob = _clamp(rating_interception_prob, 0.015, 0.35)

    tackling = _avg_rating_with_fallback(coverage_players, "tackling", default=68.0)
    pursuit = _avg_rating_with_fallback(coverage_players, "pursuit", fallback_key="play_recognition", default=68.0)
    receiver_explosive = (
        0.55 * _avg_rating_with_fallback(receivers, "yac", default=65.0)
        + 0.45 * _avg_rating_with_fallback(receivers, "speed", default=65.0)
    )
    defense_finish = (0.6 * tackling) + (0.4 * pursuit)

    feature_context = _base_context_features(game_state)
    feature_context.update(
        {
            "num:target_depth_norm": _clamp((offensive_play.target_depth + 5.0) / 45.0, 0.0, 1.0),
            "num:rushers_norm": _clamp(defensive_play.rushers / 7.0, 0.0, 1.0),
            "num:pressure_proxy": pressure_prob,
        }
    )

    completion_advantage = ((qb_accuracy + (qb_decision * 0.25)) - coverage_defense) * 0.7
    completion_advantage += (separation_offense - coverage_defense) * 0.3
    sack_advantage = (pressure_defense - protection_offense) + (4.0 if under_pressure else 0.0)
    interception_advantage = coverage_defense - ((0.6 * qb.rating("decision")) + (0.25 * qb_accuracy) + (0.15 * separation_offense))
    interception_advantage += 6.0 if under_pressure else 0.0
    interception_advantage += 4.0 if offensive_play.target_depth >= 15 else 0.0

    sack_baseline_prob, sack_prob, sack_shift = _resolve_probability(
        model_bundle,
        "sack",
        feature_context,
        rating_sack_prob,
        sack_advantage,
        scale=16.0,
    )
    completion_baseline_prob, completion_prob, completion_shift = _resolve_probability(
        model_bundle,
        "completion",
        feature_context,
        rating_completion_prob,
        completion_advantage,
        scale=18.0,
    )
    interception_baseline_prob, interception_prob, interception_shift = _resolve_probability(
        model_bundle,
        "interception",
        feature_context,
        rating_interception_prob,
        interception_advantage,
        scale=20.0,
    )

    events = ["Pocket under pressure" if under_pressure else "Pocket mostly clean"]

    complete = False
    sack = False
    interception = False
    fumble = False
    turnover = False
    yards_gained = 0
    explosive_prob = 0.0
    explosive_baseline_prob = 0.0
    explosive_shift = 0.0

    if rng.random() < sack_prob:
        sack = True
        yards_gained = -rng.randint(4, 10)
        events.append("QB is sacked")
        play_seconds = rng.randint(24, 38)
    else:
        if rng.random() < interception_prob:
            interception = True
            turnover = True
            yards_gained = 0
            events.append("Pass intercepted")
            play_seconds = rng.randint(8, 14)
        elif rng.random() < completion_prob:
            complete = True
            pressure_penalty = 3 if under_pressure else 0
            air_yards = max(1, int(round(offensive_play.target_depth - pressure_penalty + rng.gauss(0, 2.8))))

            yac_mean = (_avg_rating_with_fallback(receivers, "yac", default=65.0) / 17.0) - (tackling / 36.0) + 2.2
            yac = max(0, int(round(yac_mean + rng.gauss(0.0, 1.8))))

            yards_gained = air_yards + yac

            rating_explosive_prob = _clamp(0.06 + ((air_yards - 10.0) / 80.0), 0.02, 0.4)
            explosive_advantage = receiver_explosive - defense_finish
            explosive_baseline_prob, explosive_prob, explosive_shift = _resolve_probability(
                model_bundle,
                "explosive",
                feature_context,
                rating_explosive_prob,
                explosive_advantage,
                scale=18.0,
            )

            if rng.random() < explosive_prob:
                bonus = rng.randint(6, 18)
                yards_gained += bonus
                events.append(f"Explosive after-catch burst (+{bonus})")

            events.append(f"Pass complete for {yards_gained} yards")
            play_seconds = rng.randint(24, 38)
        else:
            yards_gained = 0
            events.append("Pass incomplete")
            play_seconds = rng.randint(6, 12)

    summary = (
        "Interception"
        if interception
        else "Sack"
        if sack
        else "Completion"
        if complete
        else "Incompletion"
    )

    return {
        "complete": complete,
        "sack": sack,
        "interception": interception,
        "fumble": fumble,
        "turnover": turnover,
        "yards_gained": yards_gained,
        "play_seconds": play_seconds,
        "summary": summary,
        "events": events,
        "pressure_prob": round(pressure_prob, 4),
        "separation_prob": round(separation_prob, 4),
        "completion_prob": round(completion_prob, 4),
        "completion_baseline_prob": round(completion_baseline_prob, 4),
        "completion_rating_prob": round(rating_completion_prob, 4),
        "completion_rating_shift": round(completion_shift, 4),
        "sack_prob": round(sack_prob, 4),
        "sack_baseline_prob": round(sack_baseline_prob, 4),
        "sack_rating_prob": round(rating_sack_prob, 4),
        "sack_rating_shift": round(sack_shift, 4),
        "interception_prob": round(interception_prob, 4),
        "interception_baseline_prob": round(interception_baseline_prob, 4),
        "interception_rating_prob": round(rating_interception_prob, 4),
        "interception_rating_shift": round(interception_shift, 4),
        "explosive_prob": round(explosive_prob, 4),
        "explosive_baseline_prob": round(explosive_baseline_prob, 4),
        "explosive_rating_shift": round(explosive_shift, 4),
    }
def _resolve_run_play(
    game_state: GameState,
    offensive_play: OffensivePlay,
    defensive_play: DefensivePlay,
    offense_team: Team,
    defense_team: Team,
    rng: random.Random,
    model_bundle: ModelBundle | None,
) -> dict[str, float | bool | int | str | list[str]]:
    ol = offense_team.players_by_position(Position.OL)
    running_backs = offense_team.players_by_position(Position.RB)
    skill_players = offense_team.players_by_position(Position.WR, Position.TE, Position.RB)

    rb = running_backs[0] if running_backs else (skill_players[0] if skill_players else offense_team.require_player(Position.QB))

    front_seven = defense_team.players_by_position(Position.DL, Position.EDGE, Position.LB)
    second_level = defense_team.players_by_position(Position.LB, Position.S, Position.CB)

    run_block = _avg_rating_with_fallback(ol, "run_block", fallback_key="pass_block", default=56.0)
    run_block += rb.rating("vision", 65.0) * 0.08

    run_defense = (
        0.55 * _avg_rating_with_fallback(front_seven, "run_defense", fallback_key="pass_rush_power", default=58.0)
        + 0.25 * _avg_rating_with_fallback(front_seven, "run_fit", fallback_key="play_recognition", default=58.0)
        + 0.2 * _avg_rating_with_fallback(second_level, "tackling", default=65.0)
    )
    run_defense += max(0, defensive_play.rushers - 4) * 1.8

    line_win_prob = _contest_probability(run_block, run_defense, rng)

    rb_skill = (
        0.4 * rb.rating("vision", 65.0)
        + 0.35 * rb.rating("elusiveness", 65.0)
        + 0.25 * rb.rating("speed", 65.0)
    )

    pursuit = _avg_rating_with_fallback(second_level, "pursuit", fallback_key="play_recognition", default=65.0)
    tackling = _avg_rating_with_fallback(second_level, "tackling", default=66.0)

    evade_prob = _contest_probability(rb_skill, 0.55 * pursuit + 0.45 * tackling, rng)

    feature_context = _base_context_features(game_state)
    feature_context.update(
        {
            "num:defenders_in_box_norm": _clamp((4.0 + defensive_play.rushers) / 9.0, 0.0, 1.0),
            "num:rushers_norm": _clamp(defensive_play.rushers / 7.0, 0.0, 1.0),
            "num:run_strength_proxy": line_win_prob,
            f"cat:run_gap={offensive_play.protection_call}": 1.0,
        }
    )

    rush_advantage = ((run_block - run_defense) * 0.7) + ((rb_skill - ((0.55 * pursuit) + (0.45 * tackling))) * 0.3)
    rating_rush_success_prob = line_win_prob
    rush_success_baseline_prob, rush_success_prob, rush_success_shift = _resolve_probability(
        model_bundle,
        "rush_success",
        feature_context,
        rating_rush_success_prob,
        rush_advantage,
        scale=18.0,
    )

    line_won = rng.random() < rush_success_prob

    if line_won:
        base = int(round(2.2 + rng.gauss(2.8, 2.0)))
    else:
        base = int(round(rng.gauss(0.6, 2.1))) - 1

    extra = 0
    if rng.random() < evade_prob:
        extra += rng.randint(1, 6)

    rating_explosive_prob = _clamp((rb.rating("speed", 65.0) - pursuit) / 120.0 + 0.08 * line_win_prob, 0.01, 0.22)
    explosive_advantage = ((0.6 * rb.rating("speed", 65.0)) + (0.4 * rb.rating("elusiveness", 65.0))) - ((0.6 * pursuit) + (0.4 * tackling))
    explosive_baseline_prob, explosive_prob, explosive_shift = _resolve_probability(
        model_bundle,
        "explosive",
        feature_context,
        rating_explosive_prob,
        explosive_advantage,
        scale=18.0,
    )

    explosive = False
    if line_won and rng.random() < explosive_prob:
        explosive = True
        extra += rng.randint(10, 28)

    yards_gained = max(-4, base + extra)

    ball_security = rb.rating("ball_security", 68.0)
    rating_fumble_prob = _clamp(0.009 + max(0.0, (tackling - ball_security) / 650.0), 0.005, 0.06)
    fumble_advantage = tackling - ball_security
    fumble_baseline_prob, fumble_prob, fumble_shift = _resolve_probability(
        model_bundle,
        "fumble",
        feature_context,
        rating_fumble_prob,
        fumble_advantage,
        scale=24.0,
        max_logit_shift=1.1,
    )

    fumble = rng.random() < fumble_prob
    turnover = fumble

    events = ["Run lane opens" if line_won else "Run lane closes quickly"]
    if explosive:
        events.append("Explosive run lane found")

    if fumble:
        events.append("Ball carrier fumbles")
        summary = "Fumble"
    else:
        events.append(f"Run gains {yards_gained} yards")
        summary = "Run"

    play_seconds = rng.randint(28, 40)

    return {
        "complete": False,
        "sack": False,
        "interception": False,
        "fumble": fumble,
        "turnover": turnover,
        "yards_gained": yards_gained,
        "play_seconds": play_seconds,
        "summary": summary,
        "events": events,
        "line_win_prob": round(line_win_prob, 4),
        "rush_success_prob": round(rush_success_prob, 4),
        "rush_success_baseline_prob": round(rush_success_baseline_prob, 4),
        "rush_success_rating_prob": round(rating_rush_success_prob, 4),
        "rush_success_rating_shift": round(rush_success_shift, 4),
        "evade_prob": round(evade_prob, 4),
        "explosive_prob": round(explosive_prob, 4),
        "explosive_baseline_prob": round(explosive_baseline_prob, 4),
        "explosive_rating_prob": round(rating_explosive_prob, 4),
        "explosive_rating_shift": round(explosive_shift, 4),
        "fumble_prob": round(fumble_prob, 4),
        "fumble_baseline_prob": round(fumble_baseline_prob, 4),
        "fumble_rating_prob": round(rating_fumble_prob, 4),
        "fumble_rating_shift": round(fumble_shift, 4),
    }
def _simulate_down_with_rng(
    game_state: GameState,
    offense_team: Team,
    defense_team: Team,
    rng: random.Random,
    recent_results: Sequence[PlayResult],
    model_bundle: ModelBundle | None,
) -> PlayResult:
    if not offense_team.offensive_playbook:
        raise ValueError(f"{offense_team.name} has no offensive plays")
    if not defense_team.defensive_playbook:
        raise ValueError(f"{defense_team.name} has no defensive plays")

    offensive_play = _select_offensive_play(game_state, offense_team, recent_results, rng, model_bundle)
    defensive_play = _select_defensive_play(game_state, defense_team, recent_results, rng)

    if offensive_play.play_type == PlayType.RUN:
        outcome = _resolve_run_play(game_state, offensive_play, defensive_play, offense_team, defense_team, rng, model_bundle)
    else:
        outcome = _resolve_pass_play(game_state, offensive_play, defensive_play, offense_team, defense_team, rng, model_bundle)

    events: list[str] = [
        f"Offense call: {offensive_play.name}",
        f"Defense call: {defensive_play.name}",
    ]
    events.extend(outcome["events"])

    yards_gained = int(outcome["yards_gained"])
    turnover = bool(outcome["turnover"])

    next_state, first_down = _advance_state(
        game_state,
        yards_gained,
        turnover=turnover,
        touchdown=False,
        offense_team_id=offense_team.id,
        defense_team_id=defense_team.id,
        play_seconds=int(outcome["play_seconds"]),
    )

    touchdown = next_state.offense_score > game_state.offense_score

    if touchdown:
        events.append("Touchdown")
    elif first_down:
        events.append("First down achieved")

    tendencies = _recent_offense_tendencies(recent_results)

    debug: dict[str, float] = {
        "offense_short_tendency": round(tendencies.short_rate, 4),
        "offense_deep_tendency": round(tendencies.deep_rate, 4),
        "offense_completion_tendency": round(tendencies.completion_rate, 4),
        "offense_run_tendency": round(tendencies.run_rate, 4),
    }

    for key, value in outcome.items():
        if key in {"events", "summary", "play_seconds", "yards_gained", "complete", "sack", "interception", "fumble", "turnover"}:
            continue
        debug[key] = float(value)

    return PlayResult(
        offensive_play=offensive_play,
        defensive_play=defensive_play,
        complete=bool(outcome["complete"]),
        yards_gained=yards_gained,
        first_down=first_down,
        touchdown=touchdown,
        turnover=turnover,
        sack=bool(outcome["sack"]),
        interception=bool(outcome["interception"]),
        fumble=bool(outcome["fumble"]),
        summary=str(outcome["summary"]),
        next_state=next_state,
        events=tuple(events),
        debug=debug,
    )


def simulate_down(
    game_state: GameState,
    offense_team: Team,
    defense_team: Team,
    rng_seed: int | None = None,
    recent_results: Sequence[PlayResult] | None = None,
    model_bundle: ModelBundle | None = None,
) -> PlayResult:
    """
    Simulate one down using situation-aware play selection and optional ML probabilities.

    `recent_results` can contain prior downs from the same drive. When omitted, this
    function defaults to neutral tendency assumptions.
    """

    rng = random.Random(rng_seed)
    history = tuple(recent_results or ())
    return _simulate_down_with_rng(game_state, offense_team, defense_team, rng, history, model_bundle)


def simulate_drive(
    game_state: GameState,
    offense_team: Team,
    defense_team: Team,
    rng_seed: int | None = None,
    max_plays: int = 20,
    model_bundle: ModelBundle | None = None,
) -> DriveResult:
    """Simulate one offensive drive as a sequence of downs."""

    rng = random.Random(rng_seed)
    state = game_state
    plays: list[PlayResult] = []
    outcome = "MAX_PLAYS_REACHED"

    for _ in range(max_plays):
        result = _simulate_down_with_rng(state, offense_team, defense_team, rng, plays, model_bundle)
        plays.append(result)
        state = result.next_state

        if result.touchdown:
            outcome = "TOUCHDOWN"
            break
        if result.turnover:
            outcome = "TURNOVER"
            break
        if state.possession_team_id != offense_team.id:
            outcome = "TURNOVER_ON_DOWNS"
            break

    touchdowns = sum(1 for play in plays if play.touchdown)
    turnovers = sum(1 for play in plays if play.turnover)
    if outcome == "TURNOVER_ON_DOWNS":
        turnovers += 1

    return DriveResult(
        start_state=game_state,
        end_state=state,
        plays=tuple(plays),
        outcome=outcome,
        total_yards=sum(play.yards_gained for play in plays),
        first_downs=sum(1 for play in plays if play.first_down),
        touchdowns=touchdowns,
        turnovers=turnovers,
    )


def simulate_many_drives(
    game_state: GameState,
    offense_team: Team,
    defense_team: Team,
    num_drives: int = 1000,
    rng_seed: int | None = None,
    max_plays: int = 20,
    model_bundle: ModelBundle | None = None,
) -> BatchSimulationResult:
    """Run many drives and return aggregate metrics for tuning and balancing."""

    if num_drives <= 0:
        raise ValueError("num_drives must be greater than 0")

    rng = random.Random(rng_seed)
    outcome_counts: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()

    for index in range(num_drives):
        drive_seed = rng.randint(0, 2_147_483_647)
        drive_start = replace(
            game_state,
            possession_team_id=offense_team.id,
            down=1,
            distance=10,
            drive_number=game_state.drive_number + index,
        )

        drive = simulate_drive(
            drive_start,
            offense_team,
            defense_team,
            rng_seed=drive_seed,
            max_plays=max_plays,
            model_bundle=model_bundle,
        )

        outcome_counts[drive.outcome] += 1
        raw_counts["plays"] += len(drive.plays)
        raw_counts["total_yards"] += drive.total_yards
        raw_counts["first_downs"] += drive.first_downs
        raw_counts["touchdowns"] += drive.touchdowns
        raw_counts["turnovers"] += drive.turnovers

        for play in drive.plays:
            if play.offensive_play.play_type == PlayType.PASS:
                raw_counts["passing_plays"] += 1
                raw_counts["passing_yards"] += play.yards_gained
                if play.complete:
                    raw_counts["completions"] += 1
            else:
                raw_counts["running_plays"] += 1
                raw_counts["rushing_yards"] += play.yards_gained

            if play.sack:
                raw_counts["sacks"] += 1
            if play.interception:
                raw_counts["interceptions"] += 1
            if play.fumble:
                raw_counts["fumbles"] += 1
            if play.yards_gained >= 20:
                raw_counts["explosive_plays"] += 1

    total_plays = raw_counts["plays"]
    passing_plays = raw_counts["passing_plays"]
    running_plays = raw_counts["running_plays"]
    turnover_drives = outcome_counts["TURNOVER"] + outcome_counts["TURNOVER_ON_DOWNS"]

    metrics = {
        "plays_per_drive": round(_rate(total_plays, num_drives), 4),
        "yards_per_play": round(_rate(raw_counts["total_yards"], total_plays), 4),
        "yards_per_drive": round(_rate(raw_counts["total_yards"], num_drives), 4),
        "first_downs_per_drive": round(_rate(raw_counts["first_downs"], num_drives), 4),
        "touchdown_rate": round(_rate(outcome_counts["TOUCHDOWN"], num_drives), 4),
        "turnover_drive_rate": round(_rate(turnover_drives, num_drives), 4),
        "completion_rate": round(_rate(raw_counts["completions"], passing_plays), 4),
        "sack_rate_per_pass_play": round(_rate(raw_counts["sacks"], passing_plays), 4),
        "interception_rate_per_pass_play": round(_rate(raw_counts["interceptions"], passing_plays), 4),
        "fumble_rate_per_run_play": round(_rate(raw_counts["fumbles"], running_plays), 4),
        "run_rate": round(_rate(running_plays, total_plays), 4),
        "explosive_play_rate": round(_rate(raw_counts["explosive_plays"], total_plays), 4),
    }

    return BatchSimulationResult(
        num_drives=num_drives,
        total_plays=total_plays,
        max_plays_per_drive=max_plays,
        seed=rng_seed,
        outcome_counts=dict(outcome_counts),
        raw_counts=dict(raw_counts),
        metrics=metrics,
    )

