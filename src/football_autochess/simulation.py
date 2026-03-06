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
    SimulationTuning,
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


_DEFAULT_TUNING = SimulationTuning()


def _logit(probability: float) -> float:
    bounded = _clamp(probability, 0.001, 0.999)
    return math.log(bounded / (1.0 - bounded))


def _contest_probability(offense_score: float, defense_score: float, rng: random.Random) -> float:
    noise = rng.gauss(0.0, 6.0)
    contest = ((offense_score - defense_score) + noise) / 12.0
    return _clamp(_sigmoid(contest), 0.01, 0.99)


def _apply_logit_bias(probability: float, bias: float) -> float:
    return _clamp(_sigmoid(_logit(probability) + bias), 0.001, 0.999)


def _apply_rating_delta(
    baseline_probability: float,
    rating_advantage: float,
    scale: float,
    max_logit_shift: float = 1.35,
) -> tuple[float, float]:
    shift = _clamp(rating_advantage / scale, -max_logit_shift, max_logit_shift)
    blended = _clamp(_sigmoid(_logit(baseline_probability) + shift), 0.001, 0.999)
    return blended, shift


def _blend_probability(fallback_probability: float, model_probability: float, model_weight: float) -> float:
    weight = _clamp(model_weight, 0.0, 1.0)
    return _clamp((fallback_probability * (1.0 - weight)) + (model_probability * weight), 0.001, 0.999)


def _resolve_probability(
    model_bundle: ModelBundle | None,
    event: str,
    features: dict[str, float],
    fallback_probability: float,
    rating_advantage: float,
    scale: float,
    tuning: SimulationTuning | None,
    event_logit_bias: float = 0.0,
    max_logit_shift: float = 1.35,
) -> tuple[float, float, float, float]:
    tuning_values = tuning or _DEFAULT_TUNING
    model_probability = fallback_probability
    if model_bundle is not None:
        model_probability = model_bundle.predict_event_probability(event, features, fallback_probability)

    baseline_probability = _blend_probability(
        fallback_probability,
        model_probability,
        tuning_values.event_model_weight if model_bundle is not None else 0.0,
    )
    baseline_probability = _apply_logit_bias(baseline_probability, event_logit_bias)
    blended_probability, logit_shift = _apply_rating_delta(
        baseline_probability,
        rating_advantage,
        scale,
        max_logit_shift=max_logit_shift,
    )
    return model_probability, baseline_probability, blended_probability, logit_shift


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


def _add_bucket_feature(features: dict[str, float], name: str, bucket: str) -> None:
    features[f"cat:{name}={bucket}"] = 1.0


def _distance_bucket(distance: float) -> str:
    if distance <= 2.0:
        return "very_short"
    if distance <= 4.0:
        return "short"
    if distance <= 7.0:
        return "medium"
    if distance <= 10.0:
        return "long"
    return "very_long"


def _field_zone(yardline_100: float) -> str:
    if yardline_100 <= 10.0:
        return "goal_to_go"
    if yardline_100 <= 20.0:
        return "red_zone"
    if yardline_100 <= 40.0:
        return "scoring_range"
    if yardline_100 <= 60.0:
        return "midfield"
    if yardline_100 <= 80.0:
        return "own_territory"
    return "backed_up"


def _score_state(score_diff: float) -> str:
    if score_diff <= -10.0:
        return "trailing_big"
    if score_diff < -3.0:
        return "trailing"
    if score_diff <= 3.0:
        return "neutral"
    if score_diff < 10.0:
        return "leading"
    return "leading_big"


def _pass_depth_bucket(target_depth: float) -> str:
    if target_depth < 0.0:
        return "behind_los"
    if target_depth <= 7.0:
        return "short"
    if target_depth <= 15.0:
        return "intermediate"
    return "deep"


def _yardline_to_goal(game_state: GameState) -> float:
    return float(max(1, min(99, 100 - game_state.yard_line)))


def _clock_context(game_state: GameState) -> tuple[float, float, float]:
    quarter = max(1, min(5, game_state.quarter))
    quarter_seconds = float(max(0, min(900, game_state.clock_seconds)))

    if quarter >= 5:
        return quarter_seconds, min(quarter_seconds, 900.0), quarter_seconds

    game_seconds = float(((4 - quarter) * 900) + quarter_seconds)
    if quarter in (1, 3):
        half_seconds = float(900 + quarter_seconds)
    else:
        half_seconds = quarter_seconds
    return game_seconds, half_seconds, quarter_seconds


def _base_context_features(
    game_state: GameState,
    *,
    shotgun: float = 0.0,
    no_huddle: float = 0.0,
    play_action: float = 0.0,
) -> dict[str, float]:
    down = max(1, min(4, game_state.down))
    distance = float(max(1, min(25, game_state.distance)))
    yardline_100 = _yardline_to_goal(game_state)
    quarter = max(1, min(5, game_state.quarter))
    season = max(1999, min(2035, game_state.season))
    week = max(1, min(25, game_state.week))
    score_diff = float(game_state.offense_score - game_state.defense_score)
    game_seconds, half_seconds, quarter_seconds = _clock_context(game_state)

    features = {
        "num:down_norm": down / 4.0,
        "num:distance_norm": distance / 25.0,
        "num:yardline_norm": yardline_100 / 100.0,
        "num:quarter_norm": quarter / 5.0,
        "num:season_norm": (season - 1999) / 40.0,
        "num:week_norm": week / 25.0,
        "num:score_diff_norm": _clamp(score_diff, -28.0, 28.0) / 28.0,
        "num:game_clock_norm": game_seconds / 3600.0,
        "num:half_clock_norm": half_seconds / 1800.0,
        "num:quarter_clock_norm": quarter_seconds / 900.0,
        "num:goal_to_go": _goal_to_go(game_state),
        "num:shotgun": _clamp(shotgun, 0.0, 1.0),
        "num:no_huddle": _clamp(no_huddle, 0.0, 1.0),
        "num:play_action": _clamp(play_action, 0.0, 1.0),
        "num:red_zone": 1.0 if yardline_100 <= 20.0 else 0.0,
        "num:fringe_red_zone": 1.0 if yardline_100 <= 30.0 else 0.0,
        "num:backed_up": 1.0 if yardline_100 >= 85.0 else 0.0,
        "num:late_game": 1.0 if game_seconds <= 600.0 else 0.0,
        "num:two_minute": 1.0 if half_seconds <= 120.0 else 0.0,
    }

    _add_bucket_feature(features, "down", str(down))
    _add_bucket_feature(features, "distance", _distance_bucket(distance))
    _add_bucket_feature(features, "field_zone", _field_zone(yardline_100))
    _add_bucket_feature(features, "score_state", _score_state(score_diff))
    _add_bucket_feature(features, "quarter", str(quarter))

    if game_state.season_type:
        _add_bucket_feature(features, "season_type", game_state.season_type)

    return features


def _pass_context_features(
    game_state: GameState,
    offensive_play: OffensivePlay,
    defensive_play: DefensivePlay,
    pressure_prob: float,
) -> dict[str, float]:
    features = _base_context_features(
        game_state,
        shotgun=1.0 if offensive_play.shotgun else 0.0,
        no_huddle=1.0 if offensive_play.no_huddle else 0.0,
        play_action=1.0 if offensive_play.play_action else 0.0,
    )
    air_yards = _clamp(offensive_play.target_depth, -5.0, 40.0)
    rushers = _clamp(defensive_play.rushers, 3.0, 7.0)
    features["num:target_depth_norm"] = (air_yards + 5.0) / 45.0
    features["num:rushers_norm"] = rushers / 7.0
    features["num:depth_behind_los"] = 1.0 if air_yards < 0.0 else 0.0
    features["num:pressure_proxy"] = _clamp(pressure_prob, 0.0, 1.0)
    _add_bucket_feature(features, "pass_depth", _pass_depth_bucket(air_yards))
    if offensive_play.pass_location:
        _add_bucket_feature(features, "pass_location", offensive_play.pass_location)
    return features


def _run_context_features(
    game_state: GameState,
    offensive_play: OffensivePlay,
    defensive_play: DefensivePlay,
    line_win_prob: float,
) -> dict[str, float]:
    features = _base_context_features(
        game_state,
        shotgun=1.0 if offensive_play.shotgun else 0.0,
        no_huddle=1.0 if offensive_play.no_huddle else 0.0,
        play_action=1.0 if offensive_play.play_action else 0.0,
    )
    box_count = 6.0 + max(0.0, float(defensive_play.rushers - 4))
    if game_state.distance <= 2:
        box_count += 1.0
    if defensive_play.coverage_type == CoverageType.MAN:
        box_count += 0.5
    features["num:defenders_in_box_norm"] = _clamp(box_count, 4.0, 9.0) / 9.0
    features["num:rushers_norm"] = _clamp(defensive_play.rushers, 3.0, 7.0) / 7.0
    features["num:run_strength_proxy"] = _clamp(line_win_prob, 0.0, 1.0)
    if offensive_play.run_gap:
        _add_bucket_feature(features, "run_gap", offensive_play.run_gap)
    if offensive_play.run_location:
        _add_bucket_feature(features, "run_location", offensive_play.run_location)
    return features


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
    tuning: SimulationTuning | None,
) -> OffensivePlay:
    playbook = offense_team.offensive_playbook
    tendencies = _recent_offense_tendencies(recent_results)
    tuning_values = tuning or _DEFAULT_TUNING

    aggression = _clamp(_team_tendency(offense_team, "aggressiveness", 0.5), 0.0, 1.0)
    pass_bias = _clamp(_team_tendency(offense_team, "pass_bias", 0.65), 0.05, 0.95)
    deep_bias = _clamp(_team_tendency(offense_team, "deep_shot_rate", 0.3), 0.0, 1.0)
    tempo = _clamp(_team_tendency(offense_team, "tempo", 0.18), 0.0, 1.0)
    shotgun_rate = _clamp(_team_tendency(offense_team, "shotgun_rate", pass_bias), 0.0, 1.0)
    play_action_rate = _clamp(_team_tendency(offense_team, "play_action_rate", 0.12 + ((1.0 - pass_bias) * 0.12)), 0.0, 1.0)

    playcall_context = _base_context_features(
        game_state,
        shotgun=shotgun_rate,
        no_huddle=tempo,
        play_action=play_action_rate,
    )

    predicted_pass_bias = pass_bias
    if model_bundle is not None:
        predicted_pass_bias = model_bundle.predict_pass_call_probability(playcall_context, pass_bias)

    pass_call_baseline = _blend_probability(
        pass_bias,
        predicted_pass_bias,
        tuning_values.playcall_model_weight if model_bundle is not None else 0.0,
    )
    effective_pass_bias = _apply_logit_bias(pass_call_baseline, tuning_values.pass_rate_bias)

    if game_state.distance <= 2:
        effective_pass_bias *= 0.85
    elif game_state.distance >= 8 and game_state.down >= 3:
        effective_pass_bias = _clamp(effective_pass_bias + 0.1, 0.05, 0.95)
    effective_pass_bias = _clamp(effective_pass_bias, 0.05, 0.95)

    pass_plays = [play for play in playbook if play.play_type == PlayType.PASS]
    run_plays = [play for play in playbook if play.play_type == PlayType.RUN]

    if not pass_plays:
        candidate_plays = run_plays
    elif not run_plays:
        candidate_plays = pass_plays
    elif rng.random() < effective_pass_bias:
        candidate_plays = pass_plays
    else:
        candidate_plays = run_plays

    desired_depth = float(game_state.distance) * (0.85 + aggression * 0.45)
    if game_state.down <= 2:
        desired_depth *= 0.9
    if _yardline_to_goal(game_state) <= 20:
        desired_depth *= 0.8
    desired_depth = _clamp(desired_depth, 3.0, 22.0)

    weights: list[float] = []
    for play in candidate_plays:
        weight = 1.0

        if play.play_type == PlayType.PASS:
            depth = play.target_depth
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

            if _yardline_to_goal(game_state) <= 20 and depth >= 18:
                weight *= 0.6
            if tendencies.run_rate > 0.55:
                weight *= 1.06
            if play.play_action:
                weight *= 0.92 + ((1.0 - effective_pass_bias) * 0.35)
            if play.shotgun:
                weight *= 0.9 + (shotgun_rate * 0.25)
        else:
            if game_state.distance <= 2:
                weight *= 1.35
            if game_state.down <= 2 and game_state.distance <= 6:
                weight *= 1.25
            if game_state.down in (3, 4) and game_state.distance >= 7:
                weight *= 0.55
            if _yardline_to_goal(game_state) <= 20:
                weight *= 1.2
            if _yardline_to_goal(game_state) >= 80:
                weight *= 0.88
            if tendencies.run_rate > 0.52:
                weight *= 0.82
            if tendencies.deep_rate > 0.4:
                weight *= 1.1
            if play.play_action:
                weight *= 0.9

        if recent_results and recent_results[-1].offensive_play.id == play.id:
            weight *= 0.7

        weights.append(max(0.05, weight))

    return _weighted_choice(candidate_plays, weights, rng)
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
    tuning: SimulationTuning | None,
) -> dict[str, float | bool | int | str | list[str]]:
    qb = offense_team.require_player(Position.QB)
    ol = offense_team.players_by_position(Position.OL)
    receivers = offense_team.players_by_position(Position.WR, Position.TE, Position.RB)
    rushers = defense_team.players_by_position(Position.DL, Position.EDGE, Position.LB)
    coverage_players = defense_team.players_by_position(Position.CB, Position.S, Position.LB)
    tuning_values = tuning or _DEFAULT_TUNING

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
    rating_completion_prob = _clamp(rating_completion_prob, 0.05, 0.92)

    rating_sack_prob = _clamp(pressure_prob * (0.14 + 0.03 * max(0, defensive_play.rushers - 4)), 0.01, 0.38)

    rating_interception_prob = (1.0 - rating_completion_prob) * 0.045
    rating_interception_prob += 0.012 if under_pressure else 0.0
    rating_interception_prob += 0.006 if offensive_play.target_depth >= 15 else 0.0
    rating_interception_prob -= 0.004 if separation_prob >= 0.62 else 0.0
    rating_interception_prob = _clamp(rating_interception_prob, 0.005, 0.16)

    tackling = _avg_rating_with_fallback(coverage_players, "tackling", default=68.0)
    pursuit = _avg_rating_with_fallback(coverage_players, "pursuit", fallback_key="play_recognition", default=68.0)
    receiver_explosive = (
        0.55 * _avg_rating_with_fallback(receivers, "yac", default=65.0)
        + 0.45 * _avg_rating_with_fallback(receivers, "speed", default=65.0)
    )
    defense_finish = (0.6 * tackling) + (0.4 * pursuit)

    feature_context = _pass_context_features(game_state, offensive_play, defensive_play, pressure_prob)

    completion_advantage = ((qb_accuracy + (qb_decision * 0.25)) - coverage_defense) * 0.7
    completion_advantage += (separation_offense - coverage_defense) * 0.3
    sack_advantage = (pressure_defense - protection_offense) + (4.0 if under_pressure else 0.0)
    interception_advantage = coverage_defense - ((0.6 * qb.rating("decision")) + (0.25 * qb_accuracy) + (0.15 * separation_offense))
    interception_advantage += 4.0 if under_pressure else 0.0
    interception_advantage += 2.0 if offensive_play.target_depth >= 15 else 0.0

    sack_model_prob, sack_baseline_prob, sack_prob, sack_shift = _resolve_probability(
        model_bundle,
        "sack",
        feature_context,
        rating_sack_prob,
        sack_advantage,
        scale=20.0,
        tuning=tuning_values,
        event_logit_bias=tuning_values.sack_logit_bias,
    )
    completion_model_prob, completion_baseline_prob, completion_prob, completion_shift = _resolve_probability(
        model_bundle,
        "completion",
        feature_context,
        rating_completion_prob,
        completion_advantage,
        scale=18.0,
        tuning=tuning_values,
        event_logit_bias=tuning_values.completion_logit_bias,
    )
    interception_model_prob, interception_baseline_prob, interception_prob, interception_shift = _resolve_probability(
        model_bundle,
        "interception",
        feature_context,
        rating_interception_prob,
        interception_advantage,
        scale=30.0,
        tuning=tuning_values,
        event_logit_bias=tuning_values.interception_logit_bias,
        max_logit_shift=0.85,
    )

    events = ["Pocket under pressure" if under_pressure else "Pocket mostly clean"]
    complete = False
    sack = False
    interception = False
    fumble = False
    turnover = False
    yards_gained = 0
    explosive_model_prob = 0.0
    explosive_prob = 0.0
    explosive_baseline_prob = 0.0
    explosive_shift = 0.0
    rating_explosive_prob = 0.0

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

            rating_explosive_prob = _clamp(0.05 + ((air_yards - 10.0) / 90.0), 0.015, 0.35)
            explosive_advantage = receiver_explosive - defense_finish
            explosive_model_prob, explosive_baseline_prob, explosive_prob, explosive_shift = _resolve_probability(
                model_bundle,
                "explosive",
                feature_context,
                rating_explosive_prob,
                explosive_advantage,
                scale=22.0,
                tuning=tuning_values,
                event_logit_bias=tuning_values.explosive_logit_bias,
                max_logit_shift=1.0,
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
        "completion_model_prob": round(completion_model_prob, 4),
        "completion_baseline_prob": round(completion_baseline_prob, 4),
        "completion_rating_prob": round(rating_completion_prob, 4),
        "completion_rating_shift": round(completion_shift, 4),
        "sack_prob": round(sack_prob, 4),
        "sack_model_prob": round(sack_model_prob, 4),
        "sack_baseline_prob": round(sack_baseline_prob, 4),
        "sack_rating_prob": round(rating_sack_prob, 4),
        "sack_rating_shift": round(sack_shift, 4),
        "interception_prob": round(interception_prob, 4),
        "interception_model_prob": round(interception_model_prob, 4),
        "interception_baseline_prob": round(interception_baseline_prob, 4),
        "interception_rating_prob": round(rating_interception_prob, 4),
        "interception_rating_shift": round(interception_shift, 4),
        "explosive_prob": round(explosive_prob, 4),
        "explosive_model_prob": round(explosive_model_prob, 4),
        "explosive_baseline_prob": round(explosive_baseline_prob, 4),
        "explosive_rating_prob": round(rating_explosive_prob, 4),
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
    tuning: SimulationTuning | None,
) -> dict[str, float | bool | int | str | list[str]]:
    ol = offense_team.players_by_position(Position.OL)
    running_backs = offense_team.players_by_position(Position.RB)
    skill_players = offense_team.players_by_position(Position.WR, Position.TE, Position.RB)
    rb = running_backs[0] if running_backs else (skill_players[0] if skill_players else offense_team.require_player(Position.QB))
    front_seven = defense_team.players_by_position(Position.DL, Position.EDGE, Position.LB)
    second_level = defense_team.players_by_position(Position.LB, Position.S, Position.CB)
    tuning_values = tuning or _DEFAULT_TUNING

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

    feature_context = _run_context_features(game_state, offensive_play, defensive_play, line_win_prob)
    rush_advantage = ((run_block - run_defense) * 0.7) + ((rb_skill - ((0.55 * pursuit) + (0.45 * tackling))) * 0.3)
    rating_rush_success_prob = line_win_prob
    rush_success_model_prob, rush_success_baseline_prob, rush_success_prob, rush_success_shift = _resolve_probability(
        model_bundle,
        "rush_success",
        feature_context,
        rating_rush_success_prob,
        rush_advantage,
        scale=18.0,
        tuning=tuning_values,
        event_logit_bias=tuning_values.rush_success_logit_bias,
    )

    line_won = rng.random() < rush_success_prob
    if line_won:
        base = int(round(2.2 + rng.gauss(2.8, 2.0)))
    else:
        base = int(round(rng.gauss(0.6, 2.1))) - 1

    extra = 0
    if rng.random() < evade_prob:
        extra += rng.randint(1, 6)

    rating_explosive_prob = _clamp((rb.rating("speed", 65.0) - pursuit) / 140.0 + 0.07 * line_win_prob, 0.008, 0.18)
    explosive_advantage = ((0.6 * rb.rating("speed", 65.0)) + (0.4 * rb.rating("elusiveness", 65.0))) - ((0.6 * pursuit) + (0.4 * tackling))
    explosive_model_prob, explosive_baseline_prob, explosive_prob, explosive_shift = _resolve_probability(
        model_bundle,
        "explosive",
        feature_context,
        rating_explosive_prob,
        explosive_advantage,
        scale=22.0,
        tuning=tuning_values,
        event_logit_bias=tuning_values.explosive_logit_bias,
        max_logit_shift=1.0,
    )

    explosive = False
    if line_won and rng.random() < explosive_prob:
        explosive = True
        extra += rng.randint(10, 28)

    yards_gained = max(-4, base + extra)

    ball_security = rb.rating("ball_security", 68.0)
    rating_fumble_prob = _clamp(0.004 + max(0.0, (tackling - ball_security) / 900.0), 0.002, 0.03)
    fumble_advantage = tackling - ball_security
    fumble_model_prob, fumble_baseline_prob, fumble_prob, fumble_shift = _resolve_probability(
        model_bundle,
        "fumble",
        feature_context,
        rating_fumble_prob,
        fumble_advantage,
        scale=36.0,
        tuning=tuning_values,
        event_logit_bias=tuning_values.fumble_logit_bias,
        max_logit_shift=0.75,
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
        "rush_success_model_prob": round(rush_success_model_prob, 4),
        "rush_success_baseline_prob": round(rush_success_baseline_prob, 4),
        "rush_success_rating_prob": round(rating_rush_success_prob, 4),
        "rush_success_rating_shift": round(rush_success_shift, 4),
        "evade_prob": round(evade_prob, 4),
        "explosive_prob": round(explosive_prob, 4),
        "explosive_model_prob": round(explosive_model_prob, 4),
        "explosive_baseline_prob": round(explosive_baseline_prob, 4),
        "explosive_rating_prob": round(rating_explosive_prob, 4),
        "explosive_rating_shift": round(explosive_shift, 4),
        "fumble_prob": round(fumble_prob, 4),
        "fumble_model_prob": round(fumble_model_prob, 4),
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
    tuning: SimulationTuning | None,
) -> PlayResult:
    if not offense_team.offensive_playbook:
        raise ValueError(f"{offense_team.name} has no offensive plays")
    if not defense_team.defensive_playbook:
        raise ValueError(f"{defense_team.name} has no defensive plays")

    offensive_play = _select_offensive_play(game_state, offense_team, recent_results, rng, model_bundle, tuning)
    defensive_play = _select_defensive_play(game_state, defense_team, recent_results, rng)

    if offensive_play.play_type == PlayType.RUN:
        outcome = _resolve_run_play(game_state, offensive_play, defensive_play, offense_team, defense_team, rng, model_bundle, tuning)
    else:
        outcome = _resolve_pass_play(game_state, offensive_play, defensive_play, offense_team, defense_team, rng, model_bundle, tuning)

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
    tuning: SimulationTuning | None = None,
) -> PlayResult:
    """
    Simulate one down using situation-aware play selection and optional ML probabilities.

    `recent_results` can contain prior downs from the same drive. When omitted, this
    function defaults to neutral tendency assumptions.
    """

    rng = random.Random(rng_seed)
    history = tuple(recent_results or ())
    return _simulate_down_with_rng(game_state, offense_team, defense_team, rng, history, model_bundle, tuning)
def simulate_drive(
    game_state: GameState,
    offense_team: Team,
    defense_team: Team,
    rng_seed: int | None = None,
    max_plays: int = 20,
    model_bundle: ModelBundle | None = None,
    tuning: SimulationTuning | None = None,
) -> DriveResult:
    """Simulate one offensive drive as a sequence of downs."""

    rng = random.Random(rng_seed)
    state = game_state
    plays: list[PlayResult] = []
    outcome = "MAX_PLAYS_REACHED"

    for _ in range(max_plays):
        result = _simulate_down_with_rng(state, offense_team, defense_team, rng, plays, model_bundle, tuning)
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
    tuning: SimulationTuning | None = None,
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
            tuning=tuning,
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

