"""Evaluate baseline and reference implementations on TA (Taillard, 1993).

Baseline is pure-python and independent from `job_shop_lib`.
Reference uses `job_shop_lib` + OR-Tools.
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


FAMILY_PREFIX = "ta"
FAMILY_NAME = "TA (Taillard, 1993)"


@dataclass
class InstanceResult:
    name: str
    optimum: int | None
    lower_bound: int | None
    upper_bound: int | None
    baseline_makespan: int
    baseline_elapsed_s: float
    reference_makespan: int | None
    reference_elapsed_s: float | None
    reference_error: str | None


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _score(target: int | None, makespan: int | None) -> float | None:
    if target is None or makespan is None or makespan <= 0:
        return None
    return min(100.0, 100.0 * float(target) / float(makespan))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def _fmt_int(value: int | None) -> str:
    return "-" if value is None else str(value)


def _fmt_float(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _select_instances(
    all_instances: list[dict],
    names: list[str] | None,
    max_instances: int | None,
) -> list[dict]:
    selected = all_instances
    if names:
        by_name = {ins["name"]: ins for ins in selected}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(
                f"Unknown instance(s): {missing}. "
                f"Known prefix={FAMILY_PREFIX}."
            )
        selected = [by_name[name] for name in names]
    if max_instances is not None:
        selected = selected[: max(max_instances, 0)]
    return selected


def evaluate_instances(
    instances: list[dict],
    reference_time_limit: float,
    baseline_mod: ModuleType,
    reference_mod: ModuleType,
) -> list[InstanceResult]:
    results: list[InstanceResult] = []

    reference_map = {
        ins.name: ins
        for ins in reference_mod.load_family_instances()
    }

    for instance in instances:
        meta = instance["metadata"]
        optimum = meta.get("optimum")
        lower_bound = meta.get("lower_bound")
        upper_bound = meta.get("upper_bound")

        start = time.perf_counter()
        baseline_result = baseline_mod.solve_instance(instance)
        baseline_elapsed = time.perf_counter() - start
        baseline_makespan = int(baseline_result["makespan"])

        reference_makespan: int | None = None
        reference_elapsed: float | None = None
        reference_error: str | None = None

        try:
            ref_instance = reference_map[instance["name"]]
            start = time.perf_counter()
            ref_schedule = reference_mod.solve_instance(
                ref_instance,
                max_time_in_seconds=reference_time_limit,
            )
            reference_elapsed = time.perf_counter() - start
            reference_makespan = ref_schedule.makespan()
        except Exception as exc:  # pragma: no cover - environment dependent
            reference_error = str(exc)

        results.append(
            InstanceResult(
                name=instance["name"],
                optimum=optimum,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                baseline_makespan=baseline_makespan,
                baseline_elapsed_s=baseline_elapsed,
                reference_makespan=reference_makespan,
                reference_elapsed_s=reference_elapsed,
                reference_error=reference_error,
            )
        )

    return results


def print_report(results: list[InstanceResult]) -> None:
    if not results:
        print("No instances selected.")
        return

    print(f"Family: {FAMILY_NAME} ({FAMILY_PREFIX})")
    print(
        "Columns: instance | baseline_ms | reference_ms | optimum | lower_bound | "
        "best_score(b/r) | lb_score(b/r)"
    )

    baseline_best_scores: list[float] = []
    reference_best_scores: list[float] = []
    baseline_lb_scores: list[float] = []
    reference_lb_scores: list[float] = []
    baseline_opt_gaps: list[float] = []
    reference_opt_gaps: list[float] = []

    for row in results:
        target = row.optimum if row.optimum is not None else row.upper_bound

        b_best = _score(target, row.baseline_makespan)
        r_best = _score(target, row.reference_makespan)
        b_lb = _score(row.lower_bound, row.baseline_makespan)
        r_lb = _score(row.lower_bound, row.reference_makespan)

        if b_best is not None:
            baseline_best_scores.append(b_best)
        if r_best is not None:
            reference_best_scores.append(r_best)
        if b_lb is not None:
            baseline_lb_scores.append(b_lb)
        if r_lb is not None:
            reference_lb_scores.append(r_lb)

        if row.optimum is not None:
            baseline_opt_gaps.append(
                100.0 * (row.baseline_makespan - row.optimum) / row.optimum
            )
            if row.reference_makespan is not None:
                reference_opt_gaps.append(
                    100.0 * (row.reference_makespan - row.optimum) / row.optimum
                )

        print(
            f"{row.name:8} | "
            f"{row.baseline_makespan:11} | "
            f"{_fmt_int(row.reference_makespan):11} | "
            f"{_fmt_int(row.optimum):7} | "
            f"{_fmt_int(row.lower_bound):11} | "
            f"{_fmt_float(b_best):>6}/{_fmt_float(r_best):<6} | "
            f"{_fmt_float(b_lb):>6}/{_fmt_float(r_lb):<6}"
        )

    reference_failures = [r for r in results if r.reference_error is not None]

    print("\nSummary")
    print(f"- instances: {len(results)}")
    print(f"- reference failures: {len(reference_failures)}")
    print(
        f"- avg baseline runtime (s): "
        f"{_fmt_float(_mean([r.baseline_elapsed_s for r in results]), 4)}"
    )
    print(
        f"- avg reference runtime (s): "
        f"{_fmt_float(_mean([r.reference_elapsed_s for r in results if r.reference_elapsed_s is not None]), 4)}"
    )
    print(
        f"- avg best-known score   (baseline/reference): "
        f"{_fmt_float(_mean(baseline_best_scores))} / "
        f"{_fmt_float(_mean(reference_best_scores))}"
    )
    print(
        f"- avg lower-bound score  (baseline/reference): "
        f"{_fmt_float(_mean(baseline_lb_scores))} / "
        f"{_fmt_float(_mean(reference_lb_scores))}"
    )
    print(
        f"- avg optimality gap %   (baseline/reference, known optimum only): "
        f"{_fmt_float(_mean(baseline_opt_gaps))} / "
        f"{_fmt_float(_mean(reference_opt_gaps))}"
    )
    print("- theoretical score ceiling under score_lb formula: 100.00")

    if reference_failures:
        print("\nReference solver errors:")
        for err in reference_failures[:5]:
            print(f"- {err.name}: {err.reference_error}")


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            f"Evaluate baseline and reference implementations for "
            f"{FAMILY_NAME} ({FAMILY_PREFIX})."
        )
    )
    parser.add_argument(
        "--instances",
        nargs="*",
        default=None,
        help="Optional explicit instance names.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Evaluate only the first N selected instances.",
    )
    parser.add_argument(
        "--reference-time-limit",
        type=float,
        default=10.0,
        help="Time limit in seconds per instance for reference solver.",
    )
    args = parser.parse_args()

    family_dir = Path(__file__).resolve().parents[1]
    baseline_mod = _load_module(
        f"baseline_{FAMILY_PREFIX}",
        family_dir / "baseline" / "init.py",
    )
    reference_mod = _load_module(
        f"reference_{FAMILY_PREFIX}",
        family_dir / "verification" / "reference.py",
    )

    all_instances = baseline_mod.load_family_instances()
    selected = _select_instances(all_instances, args.instances, args.max_instances)
    results = evaluate_instances(
        selected,
        args.reference_time_limit,
        baseline_mod,
        reference_mod,
    )
    print_report(results)


if __name__ == "__main__":
    _cli()
