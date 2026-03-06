from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Position(str, Enum):
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    OL = "OL"
    DL = "DL"
    EDGE = "EDGE"
    LB = "LB"
    CB = "CB"
    S = "S"


class PlayType(str, Enum):
    PASS = "PASS"
    RUN = "RUN"


class CoverageType(str, Enum):
    MAN = "MAN"
    ZONE = "ZONE"
    MIXED = "MIXED"


@dataclass(frozen=True)
class Player:
    id: str
    name: str
    position: Position
    ratings: dict[str, float]
    fatigue: float = 0.0

    def rating(self, key: str, default: float = 50.0) -> float:
        return float(self.ratings.get(key, default))


@dataclass(frozen=True)
class OffensivePlay:
    id: str
    name: str
    play_type: PlayType
    target_depth: float
    route_concepts: tuple[str, ...] = ()
    protection_call: str = "base"
    shotgun: bool = False
    no_huddle: bool = False
    play_action: bool = False
    pass_location: str = ""
    run_gap: str = ""
    run_location: str = ""


@dataclass(frozen=True)
class DefensivePlay:
    id: str
    name: str
    coverage_type: CoverageType
    rushers: int
    front: str = "even"


@dataclass(frozen=True)
class Team:
    id: str
    name: str
    roster: tuple[Player, ...]
    offensive_playbook: tuple[OffensivePlay, ...]
    defensive_playbook: tuple[DefensivePlay, ...]
    tendencies: dict[str, float] = field(default_factory=dict)

    def players_by_position(self, *positions: Position) -> tuple[Player, ...]:
        wanted = set(positions)
        return tuple(player for player in self.roster if player.position in wanted)

    def require_player(self, position: Position) -> Player:
        for player in self.roster:
            if player.position == position:
                return player
        raise ValueError(f"{self.name} has no player at required position: {position.value}")


@dataclass(frozen=True)
class GameState:
    possession_team_id: str
    down: int
    distance: int
    yard_line: int
    quarter: int = 1
    clock_seconds: int = 900
    offense_score: int = 0
    defense_score: int = 0
    drive_number: int = 1
    season: int = 2025
    week: int = 10
    season_type: str = "REG"


@dataclass(frozen=True)
class PlayResult:
    offensive_play: OffensivePlay
    defensive_play: DefensivePlay
    complete: bool
    yards_gained: int
    first_down: bool
    touchdown: bool
    turnover: bool
    sack: bool
    interception: bool
    fumble: bool
    summary: str
    next_state: GameState
    events: tuple[str, ...] = ()
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DriveResult:
    start_state: GameState
    end_state: GameState
    plays: tuple[PlayResult, ...]
    outcome: str
    total_yards: int
    first_downs: int
    touchdowns: int
    turnovers: int


@dataclass(frozen=True)
class BatchSimulationResult:
    num_drives: int
    total_plays: int
    max_plays_per_drive: int
    seed: int | None
    outcome_counts: dict[str, int]
    raw_counts: dict[str, int]
    metrics: dict[str, float]


@dataclass(frozen=True)
class SimulationTuning:
    playcall_model_weight: float = 0.4
    event_model_weight: float = 0.75
    pass_rate_bias: float = 0.0
    completion_logit_bias: float = 0.0
    sack_logit_bias: float = 0.0
    interception_logit_bias: float = 0.0
    rush_success_logit_bias: float = 0.0
    explosive_logit_bias: float = 0.0
    fumble_logit_bias: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "playcall_model_weight": self.playcall_model_weight,
            "event_model_weight": self.event_model_weight,
            "pass_rate_bias": self.pass_rate_bias,
            "completion_logit_bias": self.completion_logit_bias,
            "sack_logit_bias": self.sack_logit_bias,
            "interception_logit_bias": self.interception_logit_bias,
            "rush_success_logit_bias": self.rush_success_logit_bias,
            "explosive_logit_bias": self.explosive_logit_bias,
            "fumble_logit_bias": self.fumble_logit_bias,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> SimulationTuning:
        if not payload:
            return cls()
        return cls(
            playcall_model_weight=float(payload.get("playcall_model_weight", 0.4)),
            event_model_weight=float(payload.get("event_model_weight", 0.75)),
            pass_rate_bias=float(payload.get("pass_rate_bias", 0.0)),
            completion_logit_bias=float(payload.get("completion_logit_bias", 0.0)),
            sack_logit_bias=float(payload.get("sack_logit_bias", 0.0)),
            interception_logit_bias=float(payload.get("interception_logit_bias", 0.0)),
            rush_success_logit_bias=float(payload.get("rush_success_logit_bias", 0.0)),
            explosive_logit_bias=float(payload.get("explosive_logit_bias", 0.0)),
            fumble_logit_bias=float(payload.get("fumble_logit_bias", 0.0)),
        )
