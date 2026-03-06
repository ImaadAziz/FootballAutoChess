from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    simulated: float
    target: float
    abs_error: float
    rel_error: float
    within_tolerance: bool


@dataclass(frozen=True)
class ValidationReport:
    comparisons: tuple[MetricComparison, ...]
    mean_abs_error: float
    mean_rel_error: float
    pass_rate: float


def compare_metrics_to_targets(
    simulated: Mapping[str, float],
    targets: Mapping[str, float],
    tolerance: float = 0.1,
) -> ValidationReport:
    rows: list[MetricComparison] = []

    for metric, target in targets.items():
        if metric not in simulated:
            continue

        sim_value = float(simulated[metric])
        target_value = float(target)
        abs_error = abs(sim_value - target_value)
        denom = max(0.01, abs(target_value))
        rel_error = abs_error / denom
        rows.append(
            MetricComparison(
                metric=metric,
                simulated=sim_value,
                target=target_value,
                abs_error=abs_error,
                rel_error=rel_error,
                within_tolerance=rel_error <= tolerance,
            )
        )

    if not rows:
        return ValidationReport(
            comparisons=tuple(),
            mean_abs_error=0.0,
            mean_rel_error=0.0,
            pass_rate=0.0,
        )

    mae = sum(row.abs_error for row in rows) / len(rows)
    mre = sum(row.rel_error for row in rows) / len(rows)
    pass_rate = sum(1 for row in rows if row.within_tolerance) / len(rows)

    return ValidationReport(
        comparisons=tuple(rows),
        mean_abs_error=mae,
        mean_rel_error=mre,
        pass_rate=pass_rate,
    )
