from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Mapping

from .models import BatchSimulationResult, GameState, SimulationTuning, Team
from .simulation import simulate_many_drives


@dataclass(frozen=True)
class CalibrationConfig:
    iterations: int = 40
    drives_per_iteration: int = 300
    team_step_size: float = 0.08
    tuning_step_size: float = 0.18
    seed: int | None = None


@dataclass(frozen=True)
class CalibrationResult:
    offense_team: Team
    defense_team: Team
    tuning: SimulationTuning
    best_metrics: dict[str, float]
    best_loss: float
    history: tuple[dict[str, float], ...]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _objective(metrics: Mapping[str, float], targets: Mapping[str, float]) -> float:
    loss = 0.0
    for key, target in targets.items():
        metric = metrics.get(key)
        if metric is None:
            continue
        denom = max(0.01, abs(target))
        error = (metric - target) / denom
        loss += error * error
    return loss


def _mutate_team_tendencies(
    offense_team: Team,
    defense_team: Team,
    rng: random.Random,
    step_size: float,
) -> tuple[Team, Team]:
    offense = dict(offense_team.tendencies)
    defense = dict(defense_team.tendencies)

    offense_bounds = {
        "pass_bias": (0.25, 0.9),
        "aggressiveness": (0.1, 0.9),
        "deep_shot_rate": (0.05, 0.6),
    }
    defense_bounds = {
        "blitz_rate": (0.1, 0.75),
    }

    for key, (low, high) in offense_bounds.items():
        base = float(offense.get(key, (low + high) / 2.0))
        offense[key] = _clamp(base + rng.uniform(-step_size, step_size), low, high)

    for key, (low, high) in defense_bounds.items():
        base = float(defense.get(key, (low + high) / 2.0))
        defense[key] = _clamp(base + rng.uniform(-step_size, step_size), low, high)

    return replace(offense_team, tendencies=offense), replace(defense_team, tendencies=defense)


def _mutate_tuning(tuning: SimulationTuning, rng: random.Random, step_size: float) -> SimulationTuning:
    values = tuning.to_dict()
    bounds = {
        "playcall_model_weight": (0.0, 1.0),
        "event_model_weight": (0.0, 1.0),
        "pass_rate_bias": (-0.35, 0.35),
        "completion_logit_bias": (-1.0, 1.0),
        "sack_logit_bias": (-1.0, 1.0),
        "interception_logit_bias": (-1.5, 0.5),
        "rush_success_logit_bias": (-1.0, 1.0),
        "explosive_logit_bias": (-0.8, 0.8),
        "fumble_logit_bias": (-1.5, 0.4),
    }

    for key, (low, high) in bounds.items():
        base = float(values[key])
        values[key] = _clamp(base + rng.uniform(-step_size, step_size), low, high)

    return SimulationTuning.from_dict(values)


def calibrate_simulation_tendencies(
    game_state: GameState,
    offense_team: Team,
    defense_team: Team,
    targets: Mapping[str, float],
    config: CalibrationConfig | None = None,
    model_bundle_path: str | None = None,
    initial_tuning: SimulationTuning | None = None,
) -> CalibrationResult:
    """
    Random-search calibration over team identity plus league-level probability tuning.

    Team tendencies shape style. Simulation tuning shapes league-wide event calibration.
    That separation is more stable for an auto-chess project than trying to solve both with
    roster knobs alone.
    """

    from .ml_models import ModelBundle

    calibration = config or CalibrationConfig()
    rng = random.Random(calibration.seed)

    model_bundle = ModelBundle.load_json(model_bundle_path) if model_bundle_path else None
    best_tuning = initial_tuning or SimulationTuning()

    baseline = simulate_many_drives(
        game_state,
        offense_team,
        defense_team,
        num_drives=calibration.drives_per_iteration,
        rng_seed=rng.randint(0, 2_147_483_647),
        model_bundle=model_bundle,
        tuning=best_tuning,
    )
    best_loss = _objective(baseline.metrics, targets)
    best_offense = offense_team
    best_defense = defense_team
    best_metrics = dict(baseline.metrics)

    history: list[dict[str, float]] = [
        {
            "iteration": 0.0,
            "loss": best_loss,
        }
    ]

    for iteration in range(1, calibration.iterations + 1):
        cand_offense, cand_defense = _mutate_team_tendencies(
            best_offense,
            best_defense,
            rng,
            calibration.team_step_size,
        )
        cand_tuning = _mutate_tuning(best_tuning, rng, calibration.tuning_step_size)

        batch: BatchSimulationResult = simulate_many_drives(
            game_state,
            cand_offense,
            cand_defense,
            num_drives=calibration.drives_per_iteration,
            rng_seed=rng.randint(0, 2_147_483_647),
            model_bundle=model_bundle,
            tuning=cand_tuning,
        )

        loss = _objective(batch.metrics, targets)
        if loss <= best_loss:
            best_loss = loss
            best_offense = cand_offense
            best_defense = cand_defense
            best_tuning = cand_tuning
            best_metrics = dict(batch.metrics)

        history.append(
            {
                "iteration": float(iteration),
                "loss": loss,
                "best_loss": best_loss,
            }
        )

    return CalibrationResult(
        offense_team=best_offense,
        defense_team=best_defense,
        tuning=best_tuning,
        best_metrics=best_metrics,
        best_loss=best_loss,
        history=tuple(history),
    )
