from __future__ import annotations

from typing import Callable

from football_autochess import CoverageType, DefensivePlay, OffensivePlay, PlayType, Player, Position, RoundDefenseIdentity, Team


def build_precision_pass_offense() -> Team:
    roster = (
        Player(
            id="qb_pp_1",
            name="Micah Vale",
            position=Position.QB,
            ratings={
                "accuracy_short": 89,
                "accuracy_mid": 86,
                "accuracy_deep": 78,
                "decision": 87,
                "pocket_awareness": 85,
            },
        ),
        Player(
            id="rb_pp_1",
            name="Trey Rivers",
            position=Position.RB,
            ratings={
                "route_running": 68,
                "release": 72,
                "speed": 81,
                "yac": 79,
                "vision": 73,
                "elusiveness": 76,
                "ball_security": 80,
            },
        ),
        Player(id="wr_pp_1", name="Nico Hart", position=Position.WR, ratings={"route_running": 84, "release": 82, "speed": 81, "yac": 77}),
        Player(id="wr_pp_2", name="Evan Pike", position=Position.WR, ratings={"route_running": 80, "release": 79, "speed": 79, "yac": 74}),
        Player(id="te_pp_1", name="Jonah Cross", position=Position.TE, ratings={"route_running": 76, "release": 74, "speed": 72, "yac": 70, "run_block": 67}),
        Player(id="ol_pp_1", name="LT Boone", position=Position.OL, ratings={"pass_block": 83, "run_block": 70}),
        Player(id="ol_pp_2", name="LG Mercer", position=Position.OL, ratings={"pass_block": 81, "run_block": 69}),
        Player(id="ol_pp_3", name="C Hollis", position=Position.OL, ratings={"pass_block": 84, "run_block": 72}),
        Player(id="ol_pp_4", name="RG Voss", position=Position.OL, ratings={"pass_block": 80, "run_block": 68}),
        Player(id="ol_pp_5", name="RT Flynn", position=Position.OL, ratings={"pass_block": 82, "run_block": 69}),
    )
    playbook = (
        OffensivePlay(
            id="inside_zone",
            name="Inside Zone",
            play_type=PlayType.RUN,
            target_depth=0,
            protection_call="run_base",
            shotgun=True,
            run_gap="guard",
            run_location="middle",
        ),
        OffensivePlay(
            id="toss_crack",
            name="Toss Crack",
            play_type=PlayType.RUN,
            target_depth=1,
            protection_call="run_edge",
            run_gap="tackle",
            run_location="left",
        ),
        OffensivePlay(
            id="quick_slants",
            name="Quick Slants",
            play_type=PlayType.PASS,
            target_depth=6,
            route_concepts=("slant", "flat"),
            shotgun=True,
            pass_location="middle",
        ),
        OffensivePlay(
            id="stick_spacing",
            name="Stick Spacing",
            play_type=PlayType.PASS,
            target_depth=5,
            route_concepts=("stick", "flat"),
            shotgun=True,
            pass_location="right",
        ),
        OffensivePlay(
            id="mesh_rail",
            name="Mesh Rail",
            play_type=PlayType.PASS,
            target_depth=7,
            route_concepts=("drag", "flat"),
            shotgun=True,
            pass_location="middle",
        ),
        OffensivePlay(
            id="dagger",
            name="Dagger",
            play_type=PlayType.PASS,
            target_depth=12,
            route_concepts=("clear", "dig"),
            shotgun=True,
            pass_location="middle",
        ),
        OffensivePlay(
            id="sail",
            name="Sail",
            play_type=PlayType.PASS,
            target_depth=14,
            route_concepts=("go", "out", "sail"),
            pass_location="right",
        ),
        OffensivePlay(
            id="shot_post",
            name="Shot Post",
            play_type=PlayType.PASS,
            target_depth=20,
            route_concepts=("post", "go"),
            play_action=True,
            pass_location="middle",
        ),
    )
    return Team(
        id="PP",
        name="Precision Pass",
        roster=roster,
        offensive_playbook=playbook,
        defensive_playbook=(),
        tendencies={
            "aggressiveness": 0.61,
            "pass_bias": 0.72,
            "deep_shot_rate": 0.22,
            "tempo": 0.16,
            "shotgun_rate": 0.74,
            "play_action_rate": 0.1,
        },
    )


def build_ground_control_offense() -> Team:
    roster = (
        Player(
            id="qb_gc_1",
            name="Reed Mercer",
            position=Position.QB,
            ratings={
                "accuracy_short": 79,
                "accuracy_mid": 75,
                "accuracy_deep": 69,
                "decision": 78,
                "pocket_awareness": 76,
            },
        ),
        Player(
            id="rb_gc_1",
            name="Malik Boone",
            position=Position.RB,
            ratings={
                "route_running": 63,
                "release": 67,
                "speed": 87,
                "yac": 85,
                "vision": 90,
                "elusiveness": 88,
                "ball_security": 84,
            },
        ),
        Player(id="wr_gc_1", name="Tanner Hale", position=Position.WR, ratings={"route_running": 75, "release": 74, "speed": 78, "yac": 74}),
        Player(id="wr_gc_2", name="Bryce Cole", position=Position.WR, ratings={"route_running": 73, "release": 72, "speed": 77, "yac": 72}),
        Player(id="te_gc_1", name="Mason Pike", position=Position.TE, ratings={"route_running": 70, "release": 68, "speed": 70, "yac": 69, "run_block": 81}),
        Player(id="ol_gc_1", name="LT Rios", position=Position.OL, ratings={"pass_block": 77, "run_block": 84}),
        Player(id="ol_gc_2", name="LG Vann", position=Position.OL, ratings={"pass_block": 76, "run_block": 86}),
        Player(id="ol_gc_3", name="C Doran", position=Position.OL, ratings={"pass_block": 78, "run_block": 87}),
        Player(id="ol_gc_4", name="RG Pierce", position=Position.OL, ratings={"pass_block": 75, "run_block": 85}),
        Player(id="ol_gc_5", name="RT Keene", position=Position.OL, ratings={"pass_block": 76, "run_block": 83}),
    )
    playbook = (
        OffensivePlay(
            id="gc_inside_zone",
            name="Inside Zone",
            play_type=PlayType.RUN,
            target_depth=0,
            protection_call="run_base",
            shotgun=True,
            run_gap="guard",
            run_location="middle",
        ),
        OffensivePlay(
            id="gc_duo",
            name="Duo",
            play_type=PlayType.RUN,
            target_depth=0,
            protection_call="run_gap",
            run_gap="guard",
            run_location="middle",
        ),
        OffensivePlay(
            id="gc_counter",
            name="Counter Bash",
            play_type=PlayType.RUN,
            target_depth=0,
            protection_call="run_gap",
            run_gap="tackle",
            run_location="right",
        ),
        OffensivePlay(
            id="gc_toss",
            name="Toss Crack",
            play_type=PlayType.RUN,
            target_depth=1,
            protection_call="run_edge",
            run_gap="tackle",
            run_location="left",
        ),
        OffensivePlay(
            id="gc_stick",
            name="Stick Spacing",
            play_type=PlayType.PASS,
            target_depth=5,
            route_concepts=("stick", "flat"),
            shotgun=True,
            pass_location="right",
        ),
        OffensivePlay(
            id="gc_boot",
            name="Boot Flood",
            play_type=PlayType.PASS,
            target_depth=11,
            route_concepts=("out", "flat", "sail"),
            play_action=True,
            pass_location="right",
        ),
        OffensivePlay(
            id="gc_cross",
            name="Glance Cross",
            play_type=PlayType.PASS,
            target_depth=9,
            route_concepts=("slant", "dig"),
            play_action=True,
            pass_location="middle",
        ),
        OffensivePlay(
            id="gc_shot",
            name="Yankee Shot",
            play_type=PlayType.PASS,
            target_depth=18,
            route_concepts=("post", "go"),
            play_action=True,
            pass_location="middle",
        ),
    )
    return Team(
        id="GC",
        name="Ground Control",
        roster=roster,
        offensive_playbook=playbook,
        defensive_playbook=(),
        tendencies={
            "aggressiveness": 0.49,
            "pass_bias": 0.42,
            "deep_shot_rate": 0.18,
            "tempo": 0.12,
            "shotgun_rate": 0.36,
            "play_action_rate": 0.24,
        },
    )


def build_vertical_attack_offense() -> Team:
    roster = (
        Player(
            id="qb_va_1",
            name="Zane Cannon",
            position=Position.QB,
            ratings={
                "accuracy_short": 76,
                "accuracy_mid": 80,
                "accuracy_deep": 87,
                "decision": 79,
                "pocket_awareness": 77,
            },
        ),
        Player(
            id="rb_va_1",
            name="Kian Frost",
            position=Position.RB,
            ratings={
                "route_running": 66,
                "release": 69,
                "speed": 82,
                "yac": 78,
                "vision": 72,
                "elusiveness": 77,
                "ball_security": 78,
            },
        ),
        Player(id="wr_va_1", name="Jett Rowe", position=Position.WR, ratings={"route_running": 83, "release": 84, "speed": 92, "yac": 86}),
        Player(id="wr_va_2", name="Silas West", position=Position.WR, ratings={"route_running": 77, "release": 78, "speed": 85, "yac": 78}),
        Player(id="te_va_1", name="Noah Vale", position=Position.TE, ratings={"route_running": 71, "release": 70, "speed": 73, "yac": 71, "run_block": 66}),
        Player(id="ol_va_1", name="LT Wynn", position=Position.OL, ratings={"pass_block": 80, "run_block": 70}),
        Player(id="ol_va_2", name="LG Brody", position=Position.OL, ratings={"pass_block": 78, "run_block": 69}),
        Player(id="ol_va_3", name="C Kerr", position=Position.OL, ratings={"pass_block": 79, "run_block": 71}),
        Player(id="ol_va_4", name="RG Slate", position=Position.OL, ratings={"pass_block": 77, "run_block": 68}),
        Player(id="ol_va_5", name="RT Knox", position=Position.OL, ratings={"pass_block": 78, "run_block": 69}),
    )
    playbook = (
        OffensivePlay(
            id="va_inside_zone",
            name="Inside Zone",
            play_type=PlayType.RUN,
            target_depth=0,
            protection_call="run_base",
            shotgun=True,
            run_gap="guard",
            run_location="middle",
        ),
        OffensivePlay(
            id="va_toss",
            name="Speed Toss",
            play_type=PlayType.RUN,
            target_depth=1,
            protection_call="run_edge",
            run_gap="tackle",
            run_location="right",
        ),
        OffensivePlay(
            id="va_mesh",
            name="Mesh Rail",
            play_type=PlayType.PASS,
            target_depth=7,
            route_concepts=("drag", "flat"),
            shotgun=True,
            pass_location="middle",
        ),
        OffensivePlay(
            id="va_sail",
            name="Sail",
            play_type=PlayType.PASS,
            target_depth=14,
            route_concepts=("go", "out", "sail"),
            pass_location="right",
        ),
        OffensivePlay(
            id="va_fade",
            name="Slot Fade",
            play_type=PlayType.PASS,
            target_depth=16,
            route_concepts=("go", "flat"),
            shotgun=True,
            pass_location="right",
        ),
        OffensivePlay(
            id="va_dagger",
            name="Dagger",
            play_type=PlayType.PASS,
            target_depth=12,
            route_concepts=("clear", "dig"),
            shotgun=True,
            pass_location="middle",
        ),
        OffensivePlay(
            id="va_shot_post",
            name="Shot Post",
            play_type=PlayType.PASS,
            target_depth=20,
            route_concepts=("post", "go"),
            play_action=True,
            pass_location="middle",
        ),
        OffensivePlay(
            id="va_double_move",
            name="Double Move",
            play_type=PlayType.PASS,
            target_depth=18,
            route_concepts=("go", "post"),
            shotgun=True,
            pass_location="left",
        ),
    )
    return Team(
        id="VA",
        name="Vertical Attack",
        roster=roster,
        offensive_playbook=playbook,
        defensive_playbook=(),
        tendencies={
            "aggressiveness": 0.7,
            "pass_bias": 0.74,
            "deep_shot_rate": 0.42,
            "tempo": 0.19,
            "shotgun_rate": 0.7,
            "play_action_rate": 0.14,
        },
    )


def _defender(player_id: str, name: str, position: Position, **ratings: float) -> Player:
    return Player(id=player_id, name=name, position=position, ratings=dict(ratings))


def build_run_wall_zone_defense() -> Team:
    roster = (
        _defender("dl_rw_1", "Axel Stone", Position.DL, pass_rush_power=79, get_off=77, run_defense=88, run_fit=85),
        _defender("dl_rw_2", "Grant Holt", Position.DL, pass_rush_power=76, get_off=74, run_defense=86, run_fit=83),
        _defender("edge_rw_1", "Cole March", Position.EDGE, pass_rush_power=80, get_off=79, run_defense=82, run_fit=79),
        _defender("edge_rw_2", "Drew Knox", Position.EDGE, pass_rush_power=78, get_off=76, run_defense=81, run_fit=78),
        _defender("lb_rw_1", "Eli Shaw", Position.LB, zone_coverage=77, play_recognition=83, tackling=84, run_fit=86, pursuit=82),
        _defender("lb_rw_2", "Jace Moss", Position.LB, zone_coverage=76, play_recognition=81, tackling=83, run_fit=84, pursuit=81),
        _defender("cb_rw_1", "Finn Drew", Position.CB, man_coverage=72, zone_coverage=83, play_recognition=81, tackling=74, pursuit=75),
        _defender("cb_rw_2", "Gabe Lane", Position.CB, man_coverage=71, zone_coverage=82, play_recognition=80, tackling=75, pursuit=76),
        _defender("s_rw_1", "Hale Wells", Position.S, man_coverage=70, zone_coverage=85, play_recognition=86, tackling=81, pursuit=83),
        _defender("s_rw_2", "Ian Poe", Position.S, man_coverage=69, zone_coverage=84, play_recognition=84, tackling=80, pursuit=82),
    )
    playbook = (
        DefensivePlay(id="rwz_1", name="Bear Buzz", coverage_type=CoverageType.ZONE, rushers=5, front="bear"),
        DefensivePlay(id="rwz_2", name="Tampa 2", coverage_type=CoverageType.ZONE, rushers=4, front="even"),
        DefensivePlay(id="rwz_3", name="Quarters Sink", coverage_type=CoverageType.ZONE, rushers=4, front="over"),
        DefensivePlay(id="rwz_4", name="Robber Press", coverage_type=CoverageType.MAN, rushers=5, front="bear"),
    )
    return Team(
        id="RWZ",
        name="Run Wall Zone",
        roster=roster,
        offensive_playbook=(),
        defensive_playbook=playbook,
        tendencies={"blitz_rate": 0.26},
    )


def build_pass_rush_man_defense() -> Team:
    roster = (
        _defender("dl_pr_1", "Rex Talon", Position.DL, pass_rush_power=90, get_off=89, run_defense=73, run_fit=71),
        _defender("dl_pr_2", "Bran Ives", Position.DL, pass_rush_power=87, get_off=86, run_defense=72, run_fit=70),
        _defender("edge_pr_1", "Zane Crowe", Position.EDGE, pass_rush_power=92, get_off=91, run_defense=74, run_fit=72),
        _defender("edge_pr_2", "Micah Thorn", Position.EDGE, pass_rush_power=89, get_off=88, run_defense=73, run_fit=71),
        _defender("lb_pr_1", "Tobin Quill", Position.LB, man_coverage=74, zone_coverage=72, play_recognition=77, tackling=79, run_fit=73, pursuit=77),
        _defender("lb_pr_2", "Cade Wynn", Position.LB, man_coverage=75, zone_coverage=71, play_recognition=76, tackling=78, run_fit=72, pursuit=76),
        _defender("cb_pr_1", "Kellan Frost", Position.CB, man_coverage=86, zone_coverage=74, play_recognition=77, tackling=71, pursuit=74),
        _defender("cb_pr_2", "Luca Spence", Position.CB, man_coverage=84, zone_coverage=73, play_recognition=76, tackling=72, pursuit=74),
        _defender("s_pr_1", "Owen Slate", Position.S, man_coverage=80, zone_coverage=76, play_recognition=78, tackling=76, pursuit=78),
        _defender("s_pr_2", "Jett Mercer", Position.S, man_coverage=79, zone_coverage=75, play_recognition=77, tackling=75, pursuit=77),
    )
    playbook = (
        DefensivePlay(id="prm_1", name="Double Mug Press", coverage_type=CoverageType.MAN, rushers=6, front="double-mug"),
        DefensivePlay(id="prm_2", name="Nickel Fire", coverage_type=CoverageType.MAN, rushers=5, front="over"),
        DefensivePlay(id="prm_3", name="Cross Dog 1", coverage_type=CoverageType.MAN, rushers=5, front="even"),
        DefensivePlay(id="prm_4", name="Press Bail", coverage_type=CoverageType.MIXED, rushers=4, front="wide"),
    )
    return Team(
        id="PRM",
        name="Pass Rush Man",
        roster=roster,
        offensive_playbook=(),
        defensive_playbook=playbook,
        tendencies={"blitz_rate": 0.41},
    )


def build_light_box_man_defense() -> Team:
    roster = (
        _defender("dl_lbm_1", "Noah Vale", Position.DL, pass_rush_power=74, get_off=74, run_defense=66, run_fit=65),
        _defender("dl_lbm_2", "Pierce Hale", Position.DL, pass_rush_power=73, get_off=72, run_defense=65, run_fit=64),
        _defender("edge_lbm_1", "Jace Corbin", Position.EDGE, pass_rush_power=78, get_off=79, run_defense=68, run_fit=66),
        _defender("edge_lbm_2", "Milo Soren", Position.EDGE, pass_rush_power=77, get_off=78, run_defense=67, run_fit=65),
        _defender("lb_lbm_1", "Arlo Kent", Position.LB, man_coverage=79, zone_coverage=74, play_recognition=79, tackling=74, run_fit=68, pursuit=75),
        _defender("lb_lbm_2", "Beck Rowan", Position.LB, man_coverage=78, zone_coverage=73, play_recognition=78, tackling=73, run_fit=67, pursuit=74),
        _defender("cb_lbm_1", "Nate Sterling", Position.CB, man_coverage=91, zone_coverage=78, play_recognition=82, tackling=71, pursuit=76),
        _defender("cb_lbm_2", "Ty Roman", Position.CB, man_coverage=89, zone_coverage=77, play_recognition=81, tackling=72, pursuit=77),
        _defender("s_lbm_1", "Cruz Banner", Position.S, man_coverage=84, zone_coverage=79, play_recognition=81, tackling=75, pursuit=79),
        _defender("s_lbm_2", "Dane Keller", Position.S, man_coverage=83, zone_coverage=78, play_recognition=80, tackling=74, pursuit=78),
    )
    playbook = (
        DefensivePlay(id="lbm_1", name="Cover 1 Shade", coverage_type=CoverageType.MAN, rushers=4, front="nickel"),
        DefensivePlay(id="lbm_2", name="Two-Man", coverage_type=CoverageType.MAN, rushers=4, front="split"),
        DefensivePlay(id="lbm_3", name="Cone Bracket", coverage_type=CoverageType.MAN, rushers=4, front="even"),
        DefensivePlay(id="lbm_4", name="Match Carry", coverage_type=CoverageType.MIXED, rushers=4, front="nickel"),
    )
    return Team(
        id="LBM",
        name="Light Box Man",
        roster=roster,
        offensive_playbook=(),
        defensive_playbook=playbook,
        tendencies={"blitz_rate": 0.14},
    )


def build_light_box_zone_defense() -> Team:
    roster = (
        _defender("dl_lbz_1", "Eli Mercer", Position.DL, pass_rush_power=72, get_off=72, run_defense=64, run_fit=64),
        _defender("dl_lbz_2", "Shawn Kade", Position.DL, pass_rush_power=71, get_off=71, run_defense=63, run_fit=63),
        _defender("edge_lbz_1", "Rory Pike", Position.EDGE, pass_rush_power=75, get_off=76, run_defense=66, run_fit=65),
        _defender("edge_lbz_2", "Vince Sato", Position.EDGE, pass_rush_power=74, get_off=75, run_defense=65, run_fit=64),
        _defender("lb_lbz_1", "Omar Wynn", Position.LB, man_coverage=72, zone_coverage=84, play_recognition=85, tackling=74, run_fit=68, pursuit=78),
        _defender("lb_lbz_2", "Gavin Marsh", Position.LB, man_coverage=71, zone_coverage=83, play_recognition=84, tackling=73, run_fit=67, pursuit=77),
        _defender("cb_lbz_1", "Troy Vale", Position.CB, man_coverage=75, zone_coverage=88, play_recognition=86, tackling=72, pursuit=78),
        _defender("cb_lbz_2", "Mason Reed", Position.CB, man_coverage=74, zone_coverage=87, play_recognition=85, tackling=73, pursuit=78),
        _defender("s_lbz_1", "Cole Denton", Position.S, man_coverage=76, zone_coverage=90, play_recognition=88, tackling=77, pursuit=81),
        _defender("s_lbz_2", "Brady Shaw", Position.S, man_coverage=75, zone_coverage=89, play_recognition=87, tackling=76, pursuit=80),
    )
    playbook = (
        DefensivePlay(id="lbz_1", name="Cover 2 Sink", coverage_type=CoverageType.ZONE, rushers=4, front="split"),
        DefensivePlay(id="lbz_2", name="Quarters Read", coverage_type=CoverageType.ZONE, rushers=4, front="nickel"),
        DefensivePlay(id="lbz_3", name="Palms", coverage_type=CoverageType.ZONE, rushers=4, front="even"),
        DefensivePlay(id="lbz_4", name="Zone Dog", coverage_type=CoverageType.MIXED, rushers=5, front="nickel"),
    )
    return Team(
        id="LBZ",
        name="Light Box Zone",
        roster=roster,
        offensive_playbook=(),
        defensive_playbook=playbook,
        tendencies={"blitz_rate": 0.12},
    )


def build_balanced_mixed_defense() -> Team:
    roster = (
        _defender("dl_bm_1", "Quinn Voss", Position.DL, pass_rush_power=79, get_off=78, run_defense=78, run_fit=77),
        _defender("dl_bm_2", "Dax Mercer", Position.DL, pass_rush_power=78, get_off=77, run_defense=77, run_fit=76),
        _defender("edge_bm_1", "Rhett Cole", Position.EDGE, pass_rush_power=81, get_off=80, run_defense=76, run_fit=75),
        _defender("edge_bm_2", "Kai Banner", Position.EDGE, pass_rush_power=80, get_off=79, run_defense=75, run_fit=74),
        _defender("lb_bm_1", "Tate Hollis", Position.LB, man_coverage=76, zone_coverage=79, play_recognition=81, tackling=80, run_fit=79, pursuit=80),
        _defender("lb_bm_2", "Liam Ross", Position.LB, man_coverage=75, zone_coverage=78, play_recognition=80, tackling=79, run_fit=78, pursuit=79),
        _defender("cb_bm_1", "Silas Boone", Position.CB, man_coverage=81, zone_coverage=81, play_recognition=80, tackling=75, pursuit=77),
        _defender("cb_bm_2", "Rowan Pike", Position.CB, man_coverage=80, zone_coverage=80, play_recognition=79, tackling=74, pursuit=76),
        _defender("s_bm_1", "Wes Nolan", Position.S, man_coverage=79, zone_coverage=82, play_recognition=83, tackling=78, pursuit=80),
        _defender("s_bm_2", "Jude Arnett", Position.S, man_coverage=78, zone_coverage=81, play_recognition=82, tackling=77, pursuit=79),
    )
    playbook = (
        DefensivePlay(id="bm_1", name="Match Buzz", coverage_type=CoverageType.MIXED, rushers=4, front="even"),
        DefensivePlay(id="bm_2", name="Robber Rotate", coverage_type=CoverageType.MIXED, rushers=5, front="over"),
        DefensivePlay(id="bm_3", name="Cover 3 Match", coverage_type=CoverageType.ZONE, rushers=4, front="even"),
        DefensivePlay(id="bm_4", name="Man Slice", coverage_type=CoverageType.MAN, rushers=5, front="under"),
    )
    return Team(
        id="BMX",
        name="Balanced Mixed",
        roster=roster,
        offensive_playbook=(),
        defensive_playbook=playbook,
        tendencies={"blitz_rate": 0.22},
    )


STARTER_OFFENSE_BUILDERS = {
    "precision-pass": build_precision_pass_offense,
    "ground-control": build_ground_control_offense,
    "vertical-attack": build_vertical_attack_offense,
}


def build_starter_offense(archetype: str) -> Team:
    normalized = archetype.strip().lower()
    if normalized not in STARTER_OFFENSE_BUILDERS:
        known = ", ".join(sorted(STARTER_OFFENSE_BUILDERS))
        raise ValueError(f"Unknown starter archetype: {archetype}. Expected one of: {known}")
    return STARTER_OFFENSE_BUILDERS[normalized]()


DEFENSE_TEAM_BUILDERS: dict[str, Callable[[], Team]] = {
    "run-wall-zone": build_run_wall_zone_defense,
    "pass-rush-man": build_pass_rush_man_defense,
    "light-box-man": build_light_box_man_defense,
    "light-box-zone": build_light_box_zone_defense,
    "balanced-mixed": build_balanced_mixed_defense,
}


DEFENSE_IDENTITIES = {
    "run-wall-zone": RoundDefenseIdentity(
        name="Run Wall + Zone Shell",
        front_identity="run_wall",
        coverage_identity=CoverageType.ZONE,
        tell="Packed box with safeties sitting high.",
        strength_note="Crushes interior runs and closes deep zone windows.",
        weakness_note="Can be baited into giving up edges and underneath spacing.",
        adaptation_rate=0.42,
    ),
    "pass-rush-man": RoundDefenseIdentity(
        name="Pass Rush + Man",
        front_identity="pass_rush",
        coverage_identity=CoverageType.MAN,
        tell="Wide rush alignments with corners crowding leverage outside.",
        strength_note="Fast pressure and sticky man coverage squeeze long-developing concepts.",
        weakness_note="Light boxes and vacated underneath grass can be hit with quick game and runs.",
        adaptation_rate=0.46,
    ),
    "light-box-man": RoundDefenseIdentity(
        name="Light Box + Man",
        front_identity="light_box",
        coverage_identity=CoverageType.MAN,
        tell="Two-high safeties with corners pressed and only six bodies near the box.",
        strength_note="Elite man defenders erase first reads and isolate top targets.",
        weakness_note="Dares the offense to run and can be stressed by play-action.",
        adaptation_rate=0.34,
    ),
    "light-box-zone": RoundDefenseIdentity(
        name="Light Box + Zone",
        front_identity="light_box",
        coverage_identity=CoverageType.ZONE,
        tell="Soft two-high shell with linebackers sitting off the ball.",
        strength_note="Caps explosives and rallies cleanly to underneath throws.",
        weakness_note="Invites a patient run game and can get leaned on inside.",
        adaptation_rate=0.31,
    ),
    "balanced-mixed": RoundDefenseIdentity(
        name="Balanced + Mixed",
        front_identity="balanced",
        coverage_identity=CoverageType.MIXED,
        tell="Balanced front with safeties creeping and rotating late.",
        strength_note="Muddies pre-snap reads and stays sound against most concepts.",
        weakness_note="Does not dominate any single phase if the offense stays patient and varied.",
        adaptation_rate=0.38,
    ),
}


def build_defense_matchup(identity_slug: str) -> tuple[Team, RoundDefenseIdentity]:
    normalized = identity_slug.strip().lower()
    if normalized not in DEFENSE_TEAM_BUILDERS:
        known = ", ".join(sorted(DEFENSE_TEAM_BUILDERS))
        raise ValueError(f"Unknown defense identity: {identity_slug}. Expected one of: {known}")
    return DEFENSE_TEAM_BUILDERS[normalized](), DEFENSE_IDENTITIES[normalized]
