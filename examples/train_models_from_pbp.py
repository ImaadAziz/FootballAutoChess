from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from football_autochess import (  # noqa: E402
    LogisticTrainingConfig,
    ModelBundle,
    SparseLogisticModel,
    basic_classification_metrics,
    build_training_examples_from_pbp,
)


def _split_train_eval(examples, eval_ratio: float = 0.2):
    cut = int(len(examples) * (1.0 - eval_ratio))
    cut = max(1, min(cut, len(examples) - 1))
    return examples[:cut], examples[cut:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train event models from nflverse PBP CSV")
    parser.add_argument("pbp", help="Path to play-by-play file (.csv or .csv.gz)")
    parser.add_argument("--output", default="artifacts/model_bundle.json", help="Output model bundle JSON")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick runs")
    parser.add_argument("--max-samples", type=int, default=200000, help="Max samples per event")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--l2", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-event-samples", type=int, default=200)
    args = parser.parse_args()

    event_examples, playcall_examples, stats = build_training_examples_from_pbp(
        args.pbp,
        max_rows=args.max_rows,
        max_samples_per_event=args.max_samples,
    )

    config = LogisticTrainingConfig(
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2=args.l2,
        seed=args.seed,
    )

    trained_events: dict[str, SparseLogisticModel] = {}
    report: dict[str, dict[str, float]] = {}

    for event_name, examples in event_examples.items():
        if len(examples) < args.min_event_samples:
            continue

        train_set, eval_set = _split_train_eval(examples)
        model = SparseLogisticModel.train(event_name, train_set, config)
        trained_events[event_name] = model
        report[event_name] = basic_classification_metrics(eval_set, model)

    playcall_model = None
    if len(playcall_examples) >= args.min_event_samples:
        train_set, eval_set = _split_train_eval(playcall_examples)
        playcall_model = SparseLogisticModel.train("pass_call", train_set, config)
        report["pass_call"] = basic_classification_metrics(eval_set, playcall_model)

    bundle = ModelBundle(
        event_models=trained_events,
        playcall_model=playcall_model,
        metadata={
            "source": str(Path(args.pbp).resolve()),
            "rows_read": stats["rows_read"],
            "scrimmage_rows": stats["scrimmage_rows"],
            "pass_rows": stats["pass_rows"],
            "run_rows": stats["run_rows"],
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "seed": args.seed,
            "min_event_samples": args.min_event_samples,
        },
    )
    bundle.save_json(args.output)

    print("Training complete.")
    print(f"Model output: {Path(args.output).resolve()}")
    print("\nMetrics:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
