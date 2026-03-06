from __future__ import annotations

from football_autochess import (
    CoverageType,
    DefensivePlay,
    OffensivePlay,
    PlayType,
    Player,
    Position,
    Team,
)


def build_offense() -> Team:
    roster = (
        Player(
            id="qb1",
            name="Alex Cannon",
            position=Position.QB,
            ratings={
                "accuracy_short": 83,
                "accuracy_mid": 79,
                "accuracy_deep": 72,
                "decision": 81,
                "pocket_awareness": 78,
            },
        ),
        Player(
            id="rb1",
            name="Kai Trent",
            position=Position.RB,
            ratings={
                "route_running": 65,
                "release": 70,
                "speed": 84,
                "yac": 82,
                "vision": 80,
                "elusiveness": 83,
                "ball_security": 76,
            },
        ),
        Player(id="wr1", name="Jalen Pike", position=Position.WR, ratings={"route_running": 80, "release": 77, "speed": 84, "yac": 79}),
        Player(id="wr2", name="Miles King", position=Position.WR, ratings={"route_running": 76, "release": 75, "speed": 81, "yac": 74}),
        Player(id="te1", name="Noah Vale", position=Position.TE, ratings={"route_running": 72, "release": 68, "speed": 69, "yac": 71, "run_block": 71}),
        Player(id="ol1", name="LT Mason", position=Position.OL, ratings={"pass_block": 79, "run_block": 74}),
        Player(id="ol2", name="LG Briggs", position=Position.OL, ratings={"pass_block": 76, "run_block": 77}),
        Player(id="ol3", name="C Rowan", position=Position.OL, ratings={"pass_block": 78, "run_block": 79}),
        Player(id="ol4", name="RG Dax", position=Position.OL, ratings={"pass_block": 74, "run_block": 75}),
        Player(id="ol5", name="RT Flynn", position=Position.OL, ratings={"pass_block": 75, "run_block": 73}),
    )
    playbook = (
        OffensivePlay(id="o1", name="Inside Zone", play_type=PlayType.RUN, target_depth=0, protection_call="run_base"),
        OffensivePlay(id="o2", name="Duo", play_type=PlayType.RUN, target_depth=0, protection_call="run_gap"),
        OffensivePlay(id="o3", name="Toss Crack", play_type=PlayType.RUN, target_depth=1, protection_call="run_edge"),
        OffensivePlay(id="o4", name="Quick Slants", play_type=PlayType.PASS, target_depth=6, route_concepts=("slant", "flat")),
        OffensivePlay(id="o5", name="Stick Spacing", play_type=PlayType.PASS, target_depth=5, route_concepts=("stick", "flat")),
        OffensivePlay(id="o6", name="Dagger", play_type=PlayType.PASS, target_depth=12, route_concepts=("clear", "dig")),
        OffensivePlay(id="o7", name="Sail", play_type=PlayType.PASS, target_depth=14, route_concepts=("go", "out")),
        OffensivePlay(id="o8", name="Shot Post", play_type=PlayType.PASS, target_depth=20, route_concepts=("post", "go")),
    )
    return Team(
        id="OFF",
        name="City Hawks",
        roster=roster,
        offensive_playbook=playbook,
        defensive_playbook=(),
        tendencies={
            "aggressiveness": 0.58,
            "pass_bias": 0.67,
            "deep_shot_rate": 0.26,
        },
    )


def build_defense() -> Team:
    roster = (
        Player(id="dl1", name="A. Stone", position=Position.DL, ratings={"pass_rush_power": 80, "get_off": 78, "run_defense": 81, "run_fit": 77}),
        Player(id="dl2", name="B. Holt", position=Position.DL, ratings={"pass_rush_power": 77, "get_off": 75, "run_defense": 79, "run_fit": 75}),
        Player(id="edge1", name="C. Vale", position=Position.EDGE, ratings={"pass_rush_power": 82, "get_off": 81, "run_defense": 78, "run_fit": 74}),
        Player(id="edge2", name="D. Knox", position=Position.EDGE, ratings={"pass_rush_power": 79, "get_off": 80, "run_defense": 76, "run_fit": 73}),
        Player(id="lb1", name="E. Shaw", position=Position.LB, ratings={"pass_rush_power": 71, "get_off": 73, "zone_coverage": 74, "play_recognition": 76, "tackling": 78, "run_fit": 79, "pursuit": 80}),
        Player(id="lb2", name="J. Moss", position=Position.LB, ratings={"zone_coverage": 72, "play_recognition": 75, "tackling": 77, "run_fit": 78, "pursuit": 79}),
        Player(id="cb1", name="F. Drew", position=Position.CB, ratings={"man_coverage": 82, "zone_coverage": 76, "play_recognition": 75, "tackling": 73, "pursuit": 72}),
        Player(id="cb2", name="G. Lane", position=Position.CB, ratings={"man_coverage": 79, "zone_coverage": 78, "play_recognition": 77, "tackling": 74, "pursuit": 73}),
        Player(id="s1", name="H. Wells", position=Position.S, ratings={"man_coverage": 71, "zone_coverage": 80, "play_recognition": 83, "tackling": 79, "pursuit": 81}),
        Player(id="s2", name="I. Poe", position=Position.S, ratings={"man_coverage": 70, "zone_coverage": 79, "play_recognition": 81, "tackling": 78, "pursuit": 80}),
    )
    playbook = (
        DefensivePlay(id="d1", name="Cover 1 Man", coverage_type=CoverageType.MAN, rushers=4),
        DefensivePlay(id="d2", name="Cover 2 Zone", coverage_type=CoverageType.ZONE, rushers=4),
        DefensivePlay(id="d3", name="Nickel Pressure", coverage_type=CoverageType.MIXED, rushers=5),
        DefensivePlay(id="d4", name="Bear Front", coverage_type=CoverageType.MAN, rushers=5),
    )
    return Team(
        id="DEF",
        name="Metro Knights",
        roster=roster,
        offensive_playbook=(),
        defensive_playbook=playbook,
        tendencies={"blitz_rate": 0.38},
    )
