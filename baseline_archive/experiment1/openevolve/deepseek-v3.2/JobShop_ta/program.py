# EVOLVE-BLOCK-START
"""Simple greedy baseline for TA (Taillard, 1993).

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

FAMILY_PREFIX = "ta"
FAMILY_NAME = "TA (Taillard, 1993)"


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
    with _benchmark_json_path().open("r", encoding="utf-8") as f:
        return json.load(f)


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
    """Greedy EST+MOR scheduler on raw benchmark matrices.
    MOR: Most Operations Remaining (prioritize jobs with more remaining operations)
    Priority order: EST → Most Operations Remaining → Machine Readiness → SPT → Job ID.

    Input:
        instance dict with keys:
        - name
        - duration_matrix
        - machines_matrix
        - metadata

    Output:
        dict with at least:
        - name
        - makespan
        - machine_schedules
    """
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    total_operations = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines) + 1

    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    # Precompute total remaining operations per job (number of operations left)
    total_job_ops = [len(job) for job in durations]
    # We'll track remaining operations by subtracting as we schedule
    remaining_ops = total_job_ops[:]

    machine_schedules: list[list[dict[str, int]]] = [
        [] for _ in range(num_machines)
    ]

    scheduled = 0
    while scheduled < total_operations:
        candidates: list[tuple[int, int, int, int, int, int, int]] = []
        # (earliest_start, -remaining_ops, machine_ready, duration, job_id, op_idx, machine_id)
        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= len(durations[job_id]):
                continue

            machine_id = machines[job_id][op_idx]
            duration = durations[job_id][op_idx]
            est = max(job_ready[job_id], machine_ready[machine_id])
            # Note: remaining_ops[job_id] includes current operation
            candidates.append((est, -remaining_ops[job_id], machine_ready[machine_id], duration, job_id, op_idx, machine_id))

        if not candidates:
            raise RuntimeError("No schedulable operation found.")

        # Select based on est, then most operations remaining (largest remaining_ops),
        # then machine readiness (lower is better), then duration (SPT), then job_id for tie-breaking
        est, neg_ops, mach_ready, duration, job_id, op_idx, machine_id = min(
            candidates,
            key=lambda x: (x[0], x[1], x[2], x[3], x[4]),
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
        # Update remaining operations: subtract 1
        remaining_ops[job_id] -= 1
        scheduled += 1

    makespan = max(job_ready) if job_ready else 0
    return {
        "name": instance["name"],
        "makespan": makespan,
        "machine_schedules": machine_schedules,
        "solved_by": "GreedyESTMORPriorityBaseline",
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
