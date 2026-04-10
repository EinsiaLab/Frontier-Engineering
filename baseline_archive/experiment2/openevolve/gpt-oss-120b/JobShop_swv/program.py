# EVOLVE-BLOCK-START
"""Simple greedy baseline for SWV (Storer, Wu & Vaccari, 1992).

Baseline constraints:
- Pure Python implementation.
- Standard library only.
- No `job_shop_lib` import and no external solver usage.
"""

from __future__ import annotations

import argparse
import os
import json
import re
import time
from pathlib import Path
from typing import Any

# Cache for benchmark JSON to avoid repeated I/O
_benchmark_json_cache: dict[str, dict[str, Any]] | None = None

FAMILY_PREFIX = "swv"
FAMILY_NAME = "SWV (Storer, Wu & Vaccari, 1992)"


def _natural_key(name: str) -> list[object]:
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]


def _benchmark_json_path() -> Path:
    env_path = str(os.environ.get("JOBSHOP_BENCHMARK_JSON", "")).strip()
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"JOBSHOP_BENCHMARK_JSON points to a missing file: {candidate}"
        )

    candidates = [
        Path(__file__).resolve().parents[2] / "data" / "benchmark_instances.json",
        Path(__file__).resolve().parents[1] / "data" / "benchmark_instances.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "benchmark_instances.json not found under JobShop/data. "
        "Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def load_benchmark_json() -> dict[str, dict[str, Any]]:
    """Load the benchmark JSON file, caching the result after the first read."""
    global _benchmark_json_cache
    if _benchmark_json_cache is None:
        with _benchmark_json_path().open("r", encoding="utf-8") as f:
            _benchmark_json_cache = json.load(f)
    return _benchmark_json_cache


def load_family_instances() -> list[dict[str, Any]]:
    data = load_benchmark_json()
    selected = [
        value
        for name, value in data.items()
        if name.startswith(FAMILY_PREFIX)
    ]
    return sorted(selected, key=lambda x: _natural_key(x["name"]))


def load_instance_by_name(name: str) -> dict[str, Any]:
    data = load_benchmark_json()
    if name not in data:
        raise KeyError(f"Unknown instance: {name}")
    return data[name]


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Greedy EST+SPT scheduler with an optional critical‑path rule.

    The function builds **two** schedules:
      * plain EST + SPT
      * EST + SPT + critical‑path (prefers the job with the largest
        remaining processing time when earliest‑start times tie)

    The schedule with the smaller makespan is returned.  This mirrors the
    high‑performing variant used in earlier experiments and improves the
    overall fitness while keeping the same public API.
    """
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    total_operations = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines) + 1

    # ------------------------------------------------------------
    # Pre‑compute remaining work for each job/operation.
    # remaining_work[j][k] = sum(durations[j][k:]) (including current op)
    # ------------------------------------------------------------
    remaining_work: list[list[int]] = []
    for job_durs in durations:
        cum = [0] * (len(job_durs) + 1)          # cum[i] = sum(job_durs[i:])
        total = 0
        for i in range(len(job_durs) - 1, -1, -1):
            total += job_durs[i]
            cum[i] = total
        remaining_work.append(cum[:-1])          # drop the extra trailing zero

    # -----------------------------------------------------------------
    # Helper that builds a schedule; ``use_critical`` toggles the extra
    # tie‑breaking rule based on remaining work.
    # -----------------------------------------------------------------
    def _run(use_critical: bool) -> tuple[int, list[list[dict[str, int]]]]:
        next_op = [0] * num_jobs
        job_ready = [0] * num_jobs
        machine_ready = [0] * num_machines
        machine_schedules: list[list[dict[str, int]]] = [
            [] for _ in range(num_machines)
        ]

        scheduled = 0
        while scheduled < total_operations:
            candidates: list[tuple[int, int, int, int, int, int]] = []
            # (earliest_start, tie_key, duration, job_id, op_idx, machine_id)
            # ``tie_key`` is either the remaining work (negative, so bigger work wins)
            # or the duration itself when ``use_critical`` is False.
            for job_id in range(num_jobs):
                op_idx = next_op[job_id]
                if op_idx >= len(durations[job_id]):
                    continue

                machine_id = machines[job_id][op_idx]
                duration = durations[job_id][op_idx]
                est = max(job_ready[job_id], machine_ready[machine_id])

                if use_critical:
                    # Larger remaining work should be chosen first → use -remaining.
                    tie_key = -remaining_work[job_id][op_idx]
                else:
                    tie_key = duration

                candidates.append((est, tie_key, duration, job_id, op_idx, machine_id))

            if not candidates:
                raise RuntimeError("No schedulable operation found.")

            # Sorting key: earliest start, then tie_key, then duration, then job id.
            est, _tk, duration, job_id, op_idx, machine_id = min(
                candidates,
                key=lambda x: (x[0], x[1], x[2], x[3]),
            )
            end = est + duration

            machine_schedules[machine_id].append(
                {
                    "job_id": job_id,
                    "operation_index": op_idx,
                    "start_time": est,
                    "end_time": end,
                    "duration": duration,
                }
            )

            next_op[job_id] += 1
            job_ready[job_id] = end
            machine_ready[machine_id] = end
            scheduled += 1

        makespan = max(job_ready) if job_ready else 0
        return makespan, machine_schedules

    # -----------------------------------------------------------------
    # Additional heuristic: EST + LPT (Longest Processing Time)
    # -----------------------------------------------------------------
    def _run_lpt() -> tuple[int, list[list[dict[str, int]]]]:
        """Schedule using EST tie‑breaks that favor longer operations."""
        next_op = [0] * num_jobs
        job_ready = [0] * num_jobs
        machine_ready = [0] * num_machines
        machine_schedules: list[list[dict[str, int]]] = [
            [] for _ in range(num_machines)
        ]

        scheduled = 0
        while scheduled < total_operations:
            candidates: list[tuple[int, int, int, int, int, int]] = []
            # tie_key = -duration implements LPT (larger duration gets priority)
            for job_id in range(num_jobs):
                op_idx = next_op[job_id]
                if op_idx >= len(durations[job_id]):
                    continue

                machine_id = machines[job_id][op_idx]
                duration = durations[job_id][op_idx]
                est = max(job_ready[job_id], machine_ready[machine_id])
                tie_key = -duration
                candidates.append((est, tie_key, duration, job_id, op_idx, machine_id))

            if not candidates:
                raise RuntimeError("No schedulable operation found.")

            est, _tk, duration, job_id, op_idx, machine_id = min(
                candidates,
                key=lambda x: (x[0], x[1], x[2], x[3]),
            )
            end = est + duration

            machine_schedules[machine_id].append(
                {
                    "job_id": job_id,
                    "operation_index": op_idx,
                    "start_time": est,
                    "end_time": end,
                    "duration": duration,
                }
            )
            next_op[job_id] += 1
            job_ready[job_id] = end
            machine_ready[machine_id] = end
            scheduled += 1

        makespan = max(job_ready) if job_ready else 0
        return makespan, machine_schedules

    # Run both variants and keep the better one.
    makespan_plain, schedule_plain = _run(False)
    makespan_crit,  schedule_crit  = _run(True)
    makespan_lpt,   schedule_lpt   = _run_lpt()

    # -----------------------------------------------------------------
    # Additional heuristic: EST + Critical‑Ratio (remaining work / duration)
    # -----------------------------------------------------------------
    def _run_critical_ratio() -> tuple[int, list[list[dict[str, int]]]]:
        """EST schedule where ties are broken by the largest
        remaining_work / duration ratio (i.e. work‑intensity)."""
        next_op = [0] * num_jobs
        job_ready = [0] * num_jobs
        machine_ready = [0] * num_machines
        machine_schedules: list[list[dict[str, int]]] = [
            [] for _ in range(num_machines)
        ]

        scheduled = 0
        while scheduled < total_operations:
            candidates: list[tuple[int, float, int, int, int, int]] = []
            # (earliest_start, -ratio, duration, job_id, op_idx, machine_id)
            for job_id in range(num_jobs):
                op_idx = next_op[job_id]
                if op_idx >= len(durations[job_id]):
                    continue

                machine_id = machines[job_id][op_idx]
                duration = durations[job_id][op_idx]
                est = max(job_ready[job_id], machine_ready[machine_id])

                # avoid division by zero (duration is always >0 in valid data)
                ratio = remaining_work[job_id][op_idx] / duration
                # larger ratio should win → use negative for min‑heap ordering
                candidates.append((est, -ratio, duration, job_id, op_idx, machine_id))

            if not candidates:
                raise RuntimeError("No schedulable operation found.")

            est, _neg_ratio, duration, job_id, op_idx, machine_id = min(
                candidates,
                key=lambda x: (x[0], x[1], x[2], x[3]),
            )
            end = est + duration

            machine_schedules[machine_id].append(
                {
                    "job_id": job_id,
                    "operation_index": op_idx,
                    "start_time": est,
                    "end_time": end,
                    "duration": duration,
                }
            )

            next_op[job_id] += 1
            job_ready[job_id] = end
            machine_ready[machine_id] = end
            scheduled += 1

        makespan = max(job_ready) if job_ready else 0
        return makespan, machine_schedules

    makespan_cr, schedule_cr = _run_critical_ratio()

    # Choose the schedule with the smallest makespan among the four variants.
    best_makespan, best_schedule = min(
        (makespan_plain, schedule_plain),
        (makespan_crit,  schedule_crit),
        (makespan_lpt,   schedule_lpt),
        (makespan_cr,    schedule_cr),
        key=lambda pair: pair[0],
    )

    return {
        "name": instance["name"],
        "makespan": best_makespan,
        "machine_schedules": best_schedule,
        "solved_by": "GreedyESTSPTBaseline",
        "family": FAMILY_PREFIX,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=f"Run pure-python baseline on {FAMILY_NAME}."
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Instance name. If omitted, run the first N family instances.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=3,
        help="How many family instances to run when --instance is omitted.",
    )
    args = parser.parse_args()

    if args.instance:
        instances = [load_instance_by_name(args.instance)]
    else:
        instances = load_family_instances()[: max(args.max_instances, 1)]

    for instance in instances:
        start = time.perf_counter()
        result = solve_instance(instance)
        elapsed = time.perf_counter() - start
        print(
            f"[{FAMILY_PREFIX}] {instance['name']}: "
            f"makespan={result['makespan']} elapsed={elapsed:.4f}s"
        )


if __name__ == "__main__":
    _cli()
# EVOLVE-BLOCK-END
