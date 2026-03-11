from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .ml_models import ModelBundle
from .models import GameState, OffensivePlay, PlayResult, PlayType, Position, SimulationTuning, Team
from .rounds import RoundConfig, RoundSnapResult, build_round_start_state, simulate_round_play

FIELD_LENGTH = 100.0
FIELD_WIDTH = 53.3
FIELD_CENTER_Y = FIELD_WIDTH / 2.0
ROUND_REPLAY_VERSION = 1

PlayPicker = Callable[[GameState, Sequence[PlayResult]], OffensivePlay]

_TEAM_PALETTES: dict[str, dict[str, str]] = {
    "PP": {"primary": "#204E78", "secondary": "#E0F2FE", "accent": "#F6AE2D", "outline": "#0B132B"},
    "GC": {"primary": "#43552C", "secondary": "#F1F7ED", "accent": "#D4A373", "outline": "#1B1B1B"},
    "VA": {"primary": "#7B1E3A", "secondary": "#FFF6F9", "accent": "#F4D35E", "outline": "#160F29"},
    "RWZ": {"primary": "#364156", "secondary": "#EEF0F2", "accent": "#D66853", "outline": "#11151C"},
    "PRM": {"primary": "#3D315B", "secondary": "#F4F1DE", "accent": "#E07A5F", "outline": "#1E1B18"},
    "LBM": {"primary": "#005F73", "secondary": "#E9F5F3", "accent": "#EE9B00", "outline": "#001219"},
    "LBZ": {"primary": "#355070", "secondary": "#F7F7FF", "accent": "#B56576", "outline": "#1D1E2C"},
    "BMX": {"primary": "#2F4858", "secondary": "#F6F4F3", "accent": "#FF9F1C", "outline": "#1A1A1A"},
}

_OFFENSE_SLOTS = {
    "wr_left": (-0.5, 8.5),
    "slot_left": (-1.0, 16.0),
    "ol_left": (-0.4, 21.8),
    "ol_center_left": (-0.2, 24.3),
    "ol_center": (0.0, FIELD_CENTER_Y),
    "ol_center_right": (-0.2, 29.0),
    "ol_right": (-0.4, 31.6),
    "te_right": (-0.6, 34.0),
    "slot_right": (-1.0, 38.2),
    "wr_right": (-0.5, 44.8),
    "qb": (-4.2, FIELD_CENTER_Y),
    "rb": (-7.3, FIELD_CENTER_Y),
}


@dataclass(frozen=True)
class ReplayPoint:
    x: float
    y: float


@dataclass(frozen=True)
class ReplayActor:
    id: str
    team_side: str
    role: str
    label: str
    sprite_archetype_key: str
    start_slot: str
    start_position: ReplayPoint
    palette_tokens: dict[str, str]
    facing: str = "right"
    scale: float = 1.0


@dataclass(frozen=True)
class ReplayPath:
    id: str
    actor_id: str
    kind: str
    points: tuple[ReplayPoint, ...]
    start: float
    duration: float
    easing: str = "sine_in_out"


@dataclass(frozen=True)
class ReplayBeat:
    id: str
    kind: str
    start: float
    duration: float
    actor_ids: tuple[str, ...] = ()
    path_ids: tuple[str, ...] = ()
    banner_text: str = ""


@dataclass(frozen=True)
class ReplayCameraBeat:
    start: float
    duration: float
    focus_target: str
    zoom: float
    pan_target: ReplayPoint | None = None
    hold: float = 0.0


@dataclass(frozen=True)
class ReplaySnap:
    index: int
    pre_snap: dict[str, Any]
    outcome: dict[str, Any]
    beats: tuple[ReplayBeat, ...]
    actors: tuple[ReplayActor, ...]
    paths: tuple[ReplayPath, ...]
    camera: tuple[ReplayCameraBeat, ...]
    ui: dict[str, Any]


@dataclass(frozen=True)
class RoundReplay:
    meta: dict[str, Any]
    teams: dict[str, Any]
    field: dict[str, Any]
    snaps: tuple[ReplaySnap, ...]


def build_round_replay(
    config: RoundConfig,
    offense_team: Team,
    defense_team: Team,
    play_picker: PlayPicker,
    seed: int,
    *,
    model_bundle: ModelBundle | None = None,
    tuning: SimulationTuning | None = None,
) -> RoundReplay:
    rng = random.Random(seed)
    state = build_round_start_state(offense_team.id, config.target_yards)
    history: list[PlayResult] = []
    snaps: list[ReplaySnap] = []
    outcome = "PLAY_BUDGET_EXHAUSTED"

    for snap_index in range(config.play_budget):
        selected_play = play_picker(state, tuple(history))
        round_snap = simulate_round_play(
            state,
            offense_team,
            defense_team,
            selected_play,
            config.defense_identity,
            rng_seed=rng.randint(0, 2_147_483_647),
            recent_results=tuple(history),
            model_bundle=model_bundle,
            tuning=tuning,
        )
        replay_snap = _build_replay_snap(
            snap_index=snap_index,
            config=config,
            state=state,
            snap=round_snap,
            plays_left=config.play_budget - snap_index,
            offense_team=offense_team,
            defense_team=defense_team,
        )
        snaps.append(replay_snap)
        result = round_snap.play_result
        history.append(result)
        state = result.next_state

        if result.touchdown:
            outcome = "TOUCHDOWN"
            break
        if result.turnover:
            outcome = "TURNOVER"
            break

    replay = RoundReplay(
        meta={
            "version": ROUND_REPLAY_VERSION,
            "seed": seed,
            "round_name": config.name,
            "target_yards": config.target_yards,
            "play_budget": config.play_budget,
            "outcome": outcome,
            "plays_used": len(snaps),
        },
        teams={
            "offense": _team_payload(offense_team, "offense"),
            "defense": _team_payload(defense_team, "defense"),
        },
        field={
            "length": FIELD_LENGTH,
            "width": FIELD_WIDTH,
            "line_of_scrimmage": float(build_round_start_state(offense_team.id, config.target_yards).yard_line),
            "end_zone_direction": "right",
        },
        snaps=tuple(snaps),
    )
    validate_round_replay(replay)
    return replay


def save_round_replay(replay: RoundReplay, output_path: str | Path) -> Path:
    payload = _to_payload(replay)
    validate_round_replay(payload)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def validate_round_replay(replay: RoundReplay | Mapping[str, Any]) -> None:
    payload = _to_payload(replay)
    required_top_keys = {"meta", "teams", "field", "snaps"}
    missing = required_top_keys - payload.keys()
    if missing:
        raise ValueError(f"Round replay missing top-level keys: {sorted(missing)}")

    if not payload["snaps"]:
        raise ValueError("Round replay must contain at least one snap")

    field = payload["field"]
    for field_key in ("length", "width", "line_of_scrimmage", "end_zone_direction"):
        if field_key not in field:
            raise ValueError(f"Round replay field missing '{field_key}'")

    for snap_index, snap in enumerate(payload["snaps"]):
        for section in ("pre_snap", "outcome", "beats", "actors", "paths", "camera", "ui"):
            if section not in snap:
                raise ValueError(f"Snap {snap_index} missing '{section}'")
        if not snap["actors"]:
            raise ValueError(f"Snap {snap_index} must contain actor definitions")
        if not snap["paths"]:
            raise ValueError(f"Snap {snap_index} must contain movement paths")
        if not snap["beats"]:
            raise ValueError(f"Snap {snap_index} must contain animation beats")

        actor_ids = {actor["id"] for actor in snap["actors"]}
        for actor in snap["actors"]:
            for actor_key in ("id", "team_side", "role", "sprite_archetype_key", "start_slot", "start_position", "palette_tokens"):
                if actor_key not in actor:
                    raise ValueError(f"Snap {snap_index} actor missing '{actor_key}'")
            _validate_point(actor["start_position"], f"snap {snap_index} actor {actor['id']} start_position")

        for path in snap["paths"]:
            for path_key in ("id", "actor_id", "kind", "points", "start", "duration"):
                if path_key not in path:
                    raise ValueError(f"Snap {snap_index} path missing '{path_key}'")
            if path["actor_id"] not in actor_ids:
                raise ValueError(f"Snap {snap_index} path '{path['id']}' references unknown actor '{path['actor_id']}'")
            if not path["points"]:
                raise ValueError(f"Snap {snap_index} path '{path['id']}' must include at least one point")
            for point_index, point in enumerate(path["points"]):
                _validate_point(point, f"snap {snap_index} path {path['id']} point {point_index}")

        for beat in snap["beats"]:
            for beat_key in ("id", "kind", "start", "duration"):
                if beat_key not in beat:
                    raise ValueError(f"Snap {snap_index} beat missing '{beat_key}'")
            for actor_id in beat.get("actor_ids", []):
                if actor_id not in actor_ids:
                    raise ValueError(f"Snap {snap_index} beat '{beat['id']}' references unknown actor '{actor_id}'")

        for camera_beat in snap["camera"]:
            for camera_key in ("start", "duration", "focus_target", "zoom"):
                if camera_key not in camera_beat:
                    raise ValueError(f"Snap {snap_index} camera beat missing '{camera_key}'")
            if camera_beat.get("pan_target") is not None:
                _validate_point(camera_beat["pan_target"], f"snap {snap_index} camera pan target")


def _build_replay_snap(
    *,
    snap_index: int,
    config: RoundConfig,
    state: GameState,
    snap: RoundSnapResult,
    plays_left: int,
    offense_team: Team,
    defense_team: Team,
) -> ReplaySnap:
    result = snap.play_result
    recipe = _pick_recipe(result)
    geometry = _build_snap_geometry(state, result, offense_team, defense_team, snap.defensive_tell)
    beats = _build_beats(recipe, result, geometry["actors"])
    camera = _build_camera_beats(recipe, result, geometry)
    ui = _build_ui_payload(state, plays_left, snap, recipe)

    return ReplaySnap(
        index=snap_index,
        pre_snap={
            "down": state.down,
            "distance": state.distance,
            "yard_line": state.yard_line,
            "plays_left": plays_left,
            "line_of_scrimmage": float(state.yard_line),
            "offensive_call": result.offensive_play.name,
            "defensive_call": result.defensive_play.name,
            "tell_text": snap.defensive_tell,
            "round_tell": config.defense_identity.tell,
        },
        outcome={
            "summary": result.summary,
            "yards_gained": result.yards_gained,
            "touchdown": result.touchdown,
            "turnover": result.turnover,
            "complete": result.complete,
            "sack": result.sack,
            "interception": result.interception,
            "fumble": result.fumble,
            "explosive": _is_explosive(result),
            "recipe": recipe,
            "next_down": result.next_state.down,
            "next_distance": result.next_state.distance,
            "next_yard_line": result.next_state.yard_line,
        },
        beats=tuple(beats),
        actors=tuple(geometry["actors"]),
        paths=tuple(geometry["paths"]),
        camera=tuple(camera),
        ui=ui,
    )


def _build_ui_payload(state: GameState, plays_left: int, snap: RoundSnapResult, recipe: str) -> dict[str, Any]:
    debug = snap.play_result.debug
    return {
        "scoreboard": {
            "quarter": state.quarter,
            "clock_seconds": state.clock_seconds,
            "down_text": f"{state.down} & {state.distance}",
            "yards_remaining": state.distance,
            "plays_left": plays_left,
        },
        "play_banner": f"{snap.play_result.offensive_play.name} vs {snap.play_result.defensive_play.name}",
        "result_banner": snap.play_result.summary,
        "recipe": recipe,
        "debug_panel": {
            "visible_by_default": False,
            "probabilities": _debug_probability_panel(snap.play_result),
        },
    }


def _debug_probability_panel(result: PlayResult) -> dict[str, float]:
    debug = result.debug
    if result.offensive_play.play_type == PlayType.PASS:
        return {
            "sack": float(debug.get("snap_sack_prob", 0.0)),
            "interception": float(debug.get("snap_interception_prob", 0.0)),
            "completion": float(debug.get("snap_completion_prob", 0.0)),
            "explosive": float(debug.get("snap_explosive_prob", 0.0)),
        }
    return {
        "rush_success": float(debug.get("snap_rush_success_prob", 0.0)),
        "evade": float(debug.get("snap_evade_prob", 0.0)),
        "explosive": float(debug.get("snap_explosive_prob", 0.0)),
        "fumble": float(debug.get("snap_fumble_prob", 0.0)),
    }


def _pick_recipe(result: PlayResult) -> str:
    play = result.offensive_play
    if result.interception:
        return "interception"
    if result.sack:
        return "sack"
    if play.play_type == PlayType.RUN:
        is_edge = play.run_location in {"left", "right"}
        if result.yards_gained <= 1:
            return "edge_run_stuff" if is_edge else "inside_run_stuff"
        return "edge_run_success" if is_edge else "inside_run_success"
    if play.target_depth <= 7:
        return "quick_pass_complete" if result.complete else "quick_pass_incomplete"
    if play.target_depth <= 14:
        return "intermediate_complete" if result.complete else "intermediate_incomplete"
    return "deep_complete" if result.complete else "deep_incomplete"


def _build_beats(recipe: str, result: PlayResult, actors: Sequence[ReplayActor]) -> list[ReplayBeat]:
    actor_ids = tuple(actor.id for actor in actors)
    finish_banner = "Touchdown" if result.touchdown else result.summary
    beats = [
        ReplayBeat(id=f"{recipe}_presnap", kind="pre_snap_hold", start=0.0, duration=0.8, actor_ids=actor_ids),
        ReplayBeat(id=f"{recipe}_snap", kind="snap_release", start=0.8, duration=1.0, actor_ids=actor_ids),
        ReplayBeat(id=f"{recipe}_resolve", kind=recipe, start=1.8, duration=1.6, actor_ids=actor_ids, banner_text=result.summary),
    ]
    if result.touchdown:
        beats.append(
            ReplayBeat(
                id=f"{recipe}_touchdown",
                kind="touchdown_finish",
                start=3.4,
                duration=1.2,
                actor_ids=actor_ids,
                banner_text=finish_banner,
            )
        )
    else:
        beats.append(
            ReplayBeat(
                id=f"{recipe}_finish",
                kind="finish",
                start=3.4,
                duration=0.9,
                actor_ids=actor_ids,
                banner_text=finish_banner,
            )
        )
    return beats


def _build_camera_beats(recipe: str, result: PlayResult, geometry: dict[str, Any]) -> list[ReplayCameraBeat]:
    focus_actor = geometry["focus_actor_id"]
    finish_point = geometry["focus_point"]
    camera = [
        ReplayCameraBeat(start=0.0, duration=0.8, focus_target="line_of_scrimmage", zoom=0.92, pan_target=geometry["line_of_scrimmage_point"]),
        ReplayCameraBeat(start=0.8, duration=1.2, focus_target=focus_actor, zoom=1.02),
        ReplayCameraBeat(start=2.0, duration=1.6, focus_target=focus_actor, zoom=1.1, pan_target=finish_point),
    ]
    camera.append(
        ReplayCameraBeat(
            start=3.6,
            duration=0.8,
            focus_target=focus_actor if result.touchdown else "line_of_scrimmage",
            zoom=1.18 if result.touchdown else 0.98,
            pan_target=finish_point,
            hold=0.2,
        )
    )
    return camera


def _build_snap_geometry(
    state: GameState,
    result: PlayResult,
    offense_team: Team,
    defense_team: Team,
    defensive_tell: str,
) -> dict[str, Any]:
    los = float(state.yard_line)
    offense_palette = _palette_for_team(offense_team)
    defense_palette = _palette_for_team(defense_team)
    offense_positions = _offense_positions(los)
    defense_positions = _defense_positions(los, result, defensive_tell)
    target_key, target_y = _target_slot(result)

    qb = offense_team.require_player(Position.QB)
    rb = offense_team.players_by_position(Position.RB)[0]
    wrs = offense_team.players_by_position(Position.WR)
    tes = offense_team.players_by_position(Position.TE)
    ol = offense_team.players_by_position(Position.OL)
    edges = defense_team.players_by_position(Position.EDGE)
    dls = defense_team.players_by_position(Position.DL)
    lbs = defense_team.players_by_position(Position.LB)
    cbs = defense_team.players_by_position(Position.CB)
    safeties = defense_team.players_by_position(Position.S)

    wr_left = wrs[0] if wrs else tes[0]
    wr_right = wrs[1] if len(wrs) > 1 else wr_left
    te_right = tes[0] if tes else wr_right
    ol_left = ol[0] if ol else qb
    ol_right = ol[-1] if ol else qb
    edge_left = edges[0] if edges else dls[0]
    edge_right = edges[1] if len(edges) > 1 else edge_left
    linebacker = lbs[0] if lbs else safeties[0]
    corner = cbs[0] if cbs else safeties[0]
    safety = safeties[0] if safeties else linebacker

    actor_specs = [
        (qb, "offense", "qb", "toy_qb", "qb", offense_positions["qb"], "right", 1.0),
        (rb, "offense", "ball_carrier", "toy_skill", "rb", offense_positions["rb"], "right", 0.98),
        (wr_left, "offense", "receiver", "toy_skill", "wr_left", offense_positions["wr_left"], "right", 0.96),
        (wr_right, "offense", "receiver", "toy_skill", "wr_right", offense_positions["wr_right"], "right", 0.96),
        (te_right, "offense", "receiver", "toy_heavy", "te_right", offense_positions["te_right"], "right", 1.02),
        (ol_left, "offense", "lineman", "toy_heavy", "ol_left", offense_positions["ol_left"], "right", 1.08),
        (ol_right, "offense", "lineman", "toy_heavy", "ol_right", offense_positions["ol_right"], "right", 1.08),
        (edge_left, "defense", "edge", "toy_front_seven", "def_edge_left", defense_positions["def_edge_left"], "left", 1.03),
        (edge_right, "defense", "edge", "toy_front_seven", "def_edge_right", defense_positions["def_edge_right"], "left", 1.03),
        (linebacker, "defense", "linebacker", "toy_front_seven", "def_lb", defense_positions["def_lb"], "left", 1.0),
        (corner, "defense", "corner", "toy_db", "def_cb", defense_positions["def_cb"], "left", 0.94),
        (safety, "defense", "safety", "toy_db", "def_safety", defense_positions["def_safety"], "left", 0.95),
    ]
    actors = [
        ReplayActor(
            id=player.id,
            team_side=team_side,
            role=role,
            label=player.name,
            sprite_archetype_key=archetype,
            start_slot=slot_name,
            start_position=position,
            palette_tokens=offense_palette if team_side == "offense" else defense_palette,
            facing=facing,
            scale=scale,
        )
        for player, team_side, role, archetype, slot_name, position, facing, scale in actor_specs
    ]
    ball_point = offense_positions["qb"]
    actors.append(
        ReplayActor(
            id="ball",
            team_side="neutral",
            role="ball",
            label="Ball",
            sprite_archetype_key="football",
            start_slot="ball",
            start_position=ball_point,
            palette_tokens={"primary": "#8D5524", "secondary": "#F8E9D2", "accent": "#F8E9D2", "outline": "#3E2723"},
            facing="right",
            scale=0.55,
        )
    )

    actor_lookup = {actor.id: actor for actor in actors}
    target_actor_id = _target_actor_id(result, wr_left.id, wr_right.id, te_right.id, rb.id)
    target_start = actor_lookup[target_actor_id].start_position
    line_point = ReplayPoint(x=los, y=FIELD_CENTER_Y)

    paths = _build_paths(
        state=state,
        result=result,
        target_y=target_y,
        target_actor_id=target_actor_id,
        actor_lookup=actor_lookup,
        target_start=target_start,
        line_of_scrimmage=line_point,
    )
    finish_point = _path_finish_point(paths, target_actor_id)
    if result.sack:
        finish_point = _path_finish_point(paths, qb.id)
    elif result.interception:
        finish_point = _path_finish_point(paths, linebacker.id if target_actor_id == rb.id else safety.id)

    return {
        "actors": actors,
        "paths": paths,
        "focus_actor_id": target_actor_id if not result.sack else qb.id,
        "focus_point": finish_point,
        "line_of_scrimmage_point": line_point,
    }


def _build_paths(
    *,
    state: GameState,
    result: PlayResult,
    target_y: float,
    target_actor_id: str,
    actor_lookup: Mapping[str, ReplayActor],
    target_start: ReplayPoint,
    line_of_scrimmage: ReplayPoint,
) -> list[ReplayPath]:
    play = result.offensive_play
    qb_id = next(actor_id for actor_id, actor in actor_lookup.items() if actor.role == "qb")
    rb_id = next(actor_id for actor_id, actor in actor_lookup.items() if actor.role == "ball_carrier")
    edge_ids = [actor.id for actor in actor_lookup.values() if actor.role == "edge"]
    defender_ids = [actor.id for actor in actor_lookup.values() if actor.team_side == "defense" and actor.role in {"linebacker", "corner", "safety"}]

    paths: list[ReplayPath] = []
    qb_start = actor_lookup[qb_id].start_position
    rb_start = actor_lookup[rb_id].start_position
    target_x = min(FIELD_LENGTH, max(0.0, line_of_scrimmage.x + max(0, result.yards_gained)))

    qb_drop = ReplayPoint(x=max(0.0, qb_start.x - 2.1), y=qb_start.y)
    paths.append(_path(f"{qb_id}_drop", qb_id, "dropback", 0.8, 0.7, qb_start, qb_drop))

    for edge_id in edge_ids:
        start = actor_lookup[edge_id].start_position
        paths.append(
            _path(
                f"{edge_id}_rush",
                edge_id,
                "rush_arc",
                0.8,
                1.0,
                start,
                ReplayPoint(x=qb_drop.x + 1.0, y=qb_drop.y + (2.4 if start.y < FIELD_CENTER_Y else -2.4)),
            )
        )

    if play.play_type == PlayType.RUN:
        lane_y = target_y
        handoff_point = ReplayPoint(x=rb_start.x + 1.4, y=rb_start.y)
        run_mid = ReplayPoint(x=line_of_scrimmage.x + 2.4, y=lane_y)
        run_end = ReplayPoint(x=min(FIELD_LENGTH, line_of_scrimmage.x + max(-3, result.yards_gained)), y=lane_y)
        paths.extend(
            [
                _path("ball_handoff", "ball", "handoff", 0.85, 0.35, qb_drop, handoff_point),
                _path(f"{rb_id}_run", rb_id, "run_lane", 0.85, 2.1, rb_start, handoff_point, run_mid, run_end),
                _path("ball_run", "ball", "ball_run", 1.0, 1.9, handoff_point, run_mid, run_end),
            ]
        )
        for defender_id in defender_ids:
            start = actor_lookup[defender_id].start_position
            finish = ReplayPoint(x=run_end.x - 0.4, y=run_end.y + (0.8 if start.y < run_end.y else -0.8))
            paths.append(_path(f"{defender_id}_fit", defender_id, "pursuit", 1.4, 1.3, start, finish))
        return paths

    route_end = ReplayPoint(x=min(FIELD_LENGTH, line_of_scrimmage.x + max(play.target_depth, 4)), y=target_y)
    catch_x = line_of_scrimmage.x + max(2.0, float(result.debug.get("air_yards", play.target_depth or 4)))
    catch_point = ReplayPoint(x=min(FIELD_LENGTH, catch_x), y=target_y)
    end_x = min(FIELD_LENGTH, catch_point.x + max(0.0, float(result.debug.get("yac", max(0, result.yards_gained - int(catch_point.x - line_of_scrimmage.x))))))
    finish_point = ReplayPoint(x=end_x, y=target_y)

    paths.append(_path(f"{target_actor_id}_route", target_actor_id, "route", 0.8, 1.4, target_start, route_end))

    cover_roles = {"corner": "coverage", "safety": "coverage", "linebacker": "hook_drop"}
    for actor in actor_lookup.values():
        if actor.team_side != "defense" or actor.role not in cover_roles:
            continue
        finish = ReplayPoint(x=min(FIELD_LENGTH, route_end.x - 1.4), y=route_end.y + (1.2 if actor.start_position.y < route_end.y else -1.2))
        start_time = 0.9 if actor.role == "corner" else 1.2
        paths.append(_path(f"{actor.id}_coverage", actor.id, cover_roles[actor.role], start_time, 1.5, actor.start_position, finish))

    if result.sack:
        sack_point = ReplayPoint(x=max(0.0, line_of_scrimmage.x + result.yards_gained), y=qb_drop.y)
        paths.extend(
            [
                _path(f"{qb_id}_sack", qb_id, "sack_escape", 1.8, 1.2, qb_drop, sack_point),
                _path("ball_sack", "ball", "ball_carry", 0.8, 2.2, qb_start, qb_drop, sack_point),
            ]
        )
        return paths

    if result.interception:
        interceptor_id = next((actor.id for actor in actor_lookup.values() if actor.role == "safety"), target_actor_id)
        pick_point = ReplayPoint(x=catch_point.x, y=target_y - 0.8)
        return_point = ReplayPoint(x=max(0.0, pick_point.x - 6.0), y=pick_point.y - 1.0)
        paths.extend(
            [
                _path("ball_flight_pick", "ball", "ball_flight", 1.6, 0.85, qb_drop, pick_point),
                _path(f"{interceptor_id}_pick", interceptor_id, "interception_return", 1.8, 1.4, pick_point, return_point),
                _path("ball_return", "ball", "ball_return", 1.8, 1.4, pick_point, return_point),
            ]
        )
        return paths

    flight_end = catch_point if result.complete else route_end
    paths.append(_path("ball_flight", "ball", "ball_flight", 1.6, 0.85, qb_drop, flight_end))
    if result.complete:
        paths.append(_path(f"{target_actor_id}_catch", target_actor_id, "catch_and_run", 1.65, 1.5, catch_point, finish_point))
        paths.append(_path("ball_after_catch", "ball", "ball_after_catch", 2.0, 1.15, catch_point, finish_point))
        for defender_id in defender_ids:
            start = actor_lookup[defender_id].start_position
            converge = ReplayPoint(x=finish_point.x - 0.5, y=finish_point.y + (0.8 if start.y < finish_point.y else -0.8))
            paths.append(_path(f"{defender_id}_tackle", defender_id, "tackle_converge", 2.0, 1.2, start, converge))
    return paths


def _path(path_id: str, actor_id: str, kind: str, start: float, duration: float, *points: ReplayPoint) -> ReplayPath:
    return ReplayPath(id=path_id, actor_id=actor_id, kind=kind, points=tuple(points), start=start, duration=duration)


def _target_slot(result: PlayResult) -> tuple[str, float]:
    play = result.offensive_play
    if play.play_type == PlayType.RUN:
        if play.run_location == "left":
            return "rb", 18.0
        if play.run_location == "right":
            return "rb", 35.2
        return "rb", FIELD_CENTER_Y

    if play.pass_location == "left":
        return "wr_left", 11.5 if play.target_depth <= 10 else 8.8
    if play.pass_location == "right":
        if play.target_depth <= 10:
            return "te_right", 33.8
        return "wr_right", 43.2
    if play.target_depth <= 8:
        return "te_right", 29.4
    return "wr_left", 23.5 if play.target_depth <= 14 else 20.8


def _target_actor_id(result: PlayResult, wr_left_id: str, wr_right_id: str, te_right_id: str, rb_id: str) -> str:
    slot, _ = _target_slot(result)
    if slot == "wr_left":
        return wr_left_id
    if slot == "wr_right":
        return wr_right_id
    if slot == "te_right":
        return te_right_id
    return rb_id


def _offense_positions(line_of_scrimmage: float) -> dict[str, ReplayPoint]:
    return {slot: _point(line_of_scrimmage + offset_x, y) for slot, (offset_x, y) in _OFFENSE_SLOTS.items()}


def _defense_positions(line_of_scrimmage: float, result: PlayResult, defensive_tell: str) -> dict[str, ReplayPoint]:
    packed_box = "Packed box" in defensive_tell
    light_box = "Light box" in defensive_tell
    man_look = "man leverage" in defensive_tell

    linebacker_x = line_of_scrimmage + (2.7 if packed_box else 5.0 if light_box else 3.6)
    safety_x = line_of_scrimmage + (13.5 if man_look else 16.0 if light_box else 14.8)
    corner_x = line_of_scrimmage + (1.8 if man_look else 4.6)
    edge_x = line_of_scrimmage + (1.6 if packed_box else 1.1)

    target_slot, target_y = _target_slot(result)
    corner_y = target_y if target_slot in {"wr_left", "wr_right"} else 37.8
    return {
        "def_edge_left": _point(edge_x, 21.0),
        "def_edge_right": _point(edge_x, 32.8),
        "def_lb": _point(linebacker_x, 26.9),
        "def_cb": _point(corner_x, corner_y),
        "def_safety": _point(safety_x, 22.8 if target_y <= FIELD_CENTER_Y else 31.5),
    }


def _path_finish_point(paths: Sequence[ReplayPath], actor_id: str) -> ReplayPoint:
    matching = [path for path in paths if path.actor_id == actor_id]
    if not matching:
        return ReplayPoint(x=50.0, y=FIELD_CENTER_Y)
    return matching[-1].points[-1]


def _palette_for_team(team: Team) -> dict[str, str]:
    return dict(_TEAM_PALETTES.get(team.id, {"primary": "#1F3A5F", "secondary": "#F8F9FA", "accent": "#FFB703", "outline": "#101820"}))


def _team_payload(team: Team, side: str) -> dict[str, Any]:
    return {
        "id": team.id,
        "name": team.name,
        "side": side,
        "palette_tokens": _palette_for_team(team),
    }


def _is_explosive(result: PlayResult) -> bool:
    return bool(result.debug.get("explosive_bonus", 0)) or result.yards_gained >= 15


def _point(x: float, y: float) -> ReplayPoint:
    return ReplayPoint(x=round(max(0.0, min(FIELD_LENGTH, x)), 3), y=round(max(0.0, min(FIELD_WIDTH, y)), 3))


def _validate_point(point: Mapping[str, Any], label: str) -> None:
    if "x" not in point or "y" not in point:
        raise ValueError(f"{label} must contain 'x' and 'y'")
    x = float(point["x"])
    y = float(point["y"])
    if not 0.0 <= x <= FIELD_LENGTH:
        raise ValueError(f"{label} x={x} is outside 0..{FIELD_LENGTH}")
    if not 0.0 <= y <= FIELD_WIDTH:
        raise ValueError(f"{label} y={y} is outside 0..{FIELD_WIDTH}")


def _to_payload(value: Any) -> Any:
    if is_dataclass(value):
        return _to_payload(asdict(value))
    if isinstance(value, dict):
        return {key: _to_payload(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_payload(inner) for inner in value]
    return value
