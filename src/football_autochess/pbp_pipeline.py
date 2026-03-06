from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Iterable

from .ml_models import FeatureVector, TrainingExample


EVENT_NAMES: tuple[str, ...] = (
    "completion",
    "sack",
    "interception",
    "rush_success",
    "explosive",
    "fumble",
)


def _to_float(raw: str | None, default: float = 0.0) -> float:
    if raw is None:
        return default
    text = str(raw).strip()
    if not text:
        return default
    if text.lower() in {"na", "nan", "none"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _to_int(raw: str | None, default: int = 0) -> int:
    return int(round(_to_float(raw, float(default))))


def _to_bool(raw: str | None) -> bool:
    return _to_int(raw, 0) == 1


def _iter_rows(path: Path, max_rows: int | None = None) -> Iterable[dict[str, object]]:
    if path.suffix == ".parquet":
        try:
            import polars as pl
        except ImportError as exc:
            raise RuntimeError(
                "Reading parquet PBP files requires `polars`. Install it directly or via `nflreadpy`."
            ) from exc

        frame = pl.read_parquet(path)
        if max_rows is not None:
            frame = frame.head(max_rows)
        for row in frame.iter_rows(named=True):
            yield row
        return

    if path.suffix == ".gz":
        handle = gzip.open(path, mode="rt", encoding="utf-8", newline="")
    else:
        handle = path.open(mode="r", encoding="utf-8", newline="")

    with handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            if max_rows is not None and index > max_rows:
                break
            yield row


def _base_features(row: dict[str, object]) -> FeatureVector:
    down = max(1, min(4, _to_int(row.get("down"), 1)))
    distance = max(1.0, min(25.0, _to_float(row.get("ydstogo"), 10.0)))
    yardline_100 = max(1.0, min(99.0, _to_float(row.get("yardline_100"), 50.0)))
    qtr = max(1, min(5, _to_int(row.get("qtr"), 1)))

    score_diff = _to_float(row.get("score_differential"), 0.0)
    game_seconds_remaining = max(0.0, min(3600.0, _to_float(row.get("game_seconds_remaining"), 1800.0)))

    features: FeatureVector = {
        "num:down_norm": down / 4.0,
        "num:distance_norm": distance / 25.0,
        "num:yardline_norm": yardline_100 / 100.0,
        "num:quarter_norm": qtr / 5.0,
        "num:score_diff_norm": max(-28.0, min(28.0, score_diff)) / 28.0,
        "num:game_clock_norm": game_seconds_remaining / 3600.0,
        "num:goal_to_go": 1.0 if _to_bool(row.get("goal_to_go")) else 0.0,
        "num:shotgun": 1.0 if _to_bool(row.get("shotgun")) else 0.0,
        "num:no_huddle": 1.0 if _to_bool(row.get("no_huddle")) else 0.0,
    }

    season_type = str(row.get("season_type", "") or "")
    if season_type:
        features[f"cat:season_type={season_type}"] = 1.0

    roof = str(row.get("roof", "") or "")
    if roof:
        features[f"cat:roof={roof}"] = 1.0

    return features


def _pass_features(row: dict[str, object], base: FeatureVector) -> FeatureVector:
    features = dict(base)
    air_yards = max(-5.0, min(40.0, _to_float(row.get("air_yards"), 8.0)))
    features["num:target_depth_norm"] = (air_yards + 5.0) / 45.0

    rushers = max(3.0, min(7.0, _to_float(row.get("number_of_pass_rushers"), 4.0)))
    features["num:rushers_norm"] = rushers / 7.0

    return features


def _run_features(row: dict[str, object], base: FeatureVector) -> FeatureVector:
    features = dict(base)
    box = max(4.0, min(9.0, _to_float(row.get("defenders_in_box"), 6.0)))
    features["num:defenders_in_box_norm"] = box / 9.0

    run_gap = str(row.get("run_gap", "") or "")
    if run_gap:
        features[f"cat:run_gap={run_gap}"] = 1.0

    return features


def _play_type_flags(row: dict[str, object]) -> tuple[bool, bool]:
    is_pass = _to_bool(row.get("pass")) or _to_bool(row.get("qb_dropback"))
    is_run = _to_bool(row.get("rush"))
    return is_pass, is_run


def _rush_success_label(down: int, ydstogo: float, yards_gained: float) -> int:
    if down == 1:
        return int(yards_gained >= 0.4 * ydstogo)
    if down == 2:
        return int(yards_gained >= 0.6 * ydstogo)
    return int(yards_gained >= ydstogo)


def build_training_examples_from_pbp(
    pbp_csv_path: str | Path,
    max_rows: int | None = None,
    max_samples_per_event: int = 250000,
) -> tuple[dict[str, list[TrainingExample]], list[TrainingExample], dict[str, int]]:
    """
    Build event-specific training sets and pass/run play-call examples from local nflverse PBP data.

    Supported inputs: `.csv`, `.csv.gz`, and `.parquet` with nflverse-style columns.
    """

    path = Path(pbp_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"PBP file not found: {path}")

    event_examples: dict[str, list[TrainingExample]] = {name: [] for name in EVENT_NAMES}
    playcall_examples: list[TrainingExample] = []

    stats = {
        "rows_read": 0,
        "scrimmage_rows": 0,
        "pass_rows": 0,
        "run_rows": 0,
    }

    for row in _iter_rows(path, max_rows=max_rows):
        stats["rows_read"] += 1

        is_pass, is_run = _play_type_flags(row)
        if not is_pass and not is_run:
            continue

        if _to_bool(row.get("qb_kneel")) or _to_bool(row.get("qb_spike")):
            continue

        stats["scrimmage_rows"] += 1
        down = max(1, min(4, _to_int(row.get("down"), 1)))
        ydstogo = max(1.0, _to_float(row.get("ydstogo"), 10.0))
        yards_gained = _to_float(row.get("yards_gained"), 0.0)

        base = _base_features(row)

        if is_pass or is_run:
            if len(playcall_examples) < max_samples_per_event:
                features = dict(base)
                features["num:is_goal_to_go"] = base["num:goal_to_go"]
                playcall_examples.append((features, int(is_pass)))

        if is_pass:
            stats["pass_rows"] += 1
            pass_features = _pass_features(row, base)
            sack = int(_to_bool(row.get("sack")))
            interception = int(_to_bool(row.get("interception")))
            complete_pass = int(_to_bool(row.get("complete_pass")))
            explosive = int(yards_gained >= 20)

            if len(event_examples["sack"]) < max_samples_per_event:
                event_examples["sack"].append((dict(pass_features), sack))

            if len(event_examples["explosive"]) < max_samples_per_event:
                event_examples["explosive"].append((dict(pass_features), explosive))

            if sack == 0:
                if len(event_examples["completion"]) < max_samples_per_event:
                    event_examples["completion"].append((dict(pass_features), complete_pass))

                if len(event_examples["interception"]) < max_samples_per_event:
                    event_examples["interception"].append((dict(pass_features), interception))

        if is_run:
            stats["run_rows"] += 1
            run_features = _run_features(row, base)
            explosive = int(yards_gained >= 10)
            fumble = int(_to_bool(row.get("fumble_lost")))
            success = _rush_success_label(down, ydstogo, yards_gained)

            if len(event_examples["rush_success"]) < max_samples_per_event:
                event_examples["rush_success"].append((dict(run_features), success))

            if len(event_examples["fumble"]) < max_samples_per_event:
                event_examples["fumble"].append((dict(run_features), fumble))

            if len(event_examples["explosive"]) < max_samples_per_event:
                event_examples["explosive"].append((dict(run_features), explosive))

    return event_examples, playcall_examples, stats


def compute_target_metrics_from_pbp(
    pbp_csv_path: str | Path,
    max_rows: int | None = None,
) -> dict[str, float]:
    path = Path(pbp_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"PBP file not found: {path}")

    counts = {
        "plays": 0,
        "pass_plays": 0,
        "run_plays": 0,
        "yards": 0.0,
        "completions": 0,
        "sacks": 0,
        "interceptions": 0,
        "fumbles": 0,
        "explosive": 0,
    }

    for row in _iter_rows(path, max_rows=max_rows):
        is_pass, is_run = _play_type_flags(row)
        if not is_pass and not is_run:
            continue
        if _to_bool(row.get("qb_kneel")) or _to_bool(row.get("qb_spike")):
            continue

        yards = _to_float(row.get("yards_gained"), 0.0)
        counts["plays"] += 1
        counts["yards"] += yards

        if is_pass:
            counts["pass_plays"] += 1
            counts["completions"] += int(_to_bool(row.get("complete_pass")))
            counts["sacks"] += int(_to_bool(row.get("sack")))
            counts["interceptions"] += int(_to_bool(row.get("interception")))
            counts["explosive"] += int(yards >= 20)

        if is_run:
            counts["run_plays"] += 1
            counts["fumbles"] += int(_to_bool(row.get("fumble_lost")))
            counts["explosive"] += int(yards >= 10)

    plays = max(1, counts["plays"])
    pass_plays = max(1, counts["pass_plays"])
    run_plays = max(1, counts["run_plays"])

    return {
        "yards_per_play": counts["yards"] / plays,
        "completion_rate": counts["completions"] / pass_plays,
        "sack_rate_per_pass_play": counts["sacks"] / pass_plays,
        "interception_rate_per_pass_play": counts["interceptions"] / pass_plays,
        "fumble_rate_per_run_play": counts["fumbles"] / run_plays,
        "run_rate": counts["run_plays"] / plays,
        "explosive_play_rate": counts["explosive"] / plays,
    }


def save_target_metrics(targets: dict[str, float], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(targets, indent=2), encoding="utf-8")


def load_target_metrics(target_path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(target_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Target metrics file must contain a JSON object")

    targets: dict[str, float] = {}
    for key, value in payload.items():
        targets[str(key)] = float(value)
    return targets
