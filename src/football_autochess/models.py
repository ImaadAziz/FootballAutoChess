from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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
