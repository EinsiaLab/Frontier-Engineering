# EVOLVE-BLOCK-START
"""Simple greedy baseline for ABZ (Adams, Balas & Zawack, 1988).

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

FAMILY_PREFIX = "abz"
FAMILY_NAME = "ABZ (Adams, Balas & Zawack, 1988)"


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
    """Greedy EST+SPT scheduler with lookahead, machine load awareness, and slack time consideration."""
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    total_ops = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines) + 1

    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    machine_loads = [0] * num_machines  # Track total processing time per machine
    job_remaining_total = [sum(job) for job in durations]  # Precompute total remaining time per job
    
    # Calculate slack time for each job (difference between latest possible start and earliest start)
    job_slack = [float('inf')] * num_jobs
    machine_schedules: list[list[dict[str, int]]] = [[] for _ in range(num_machines)]

    scheduled = 0
    while scheduled < total_ops:
        candidates: list[tuple[int, int, int, int, int, int, int, int, float]] = []
        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= len(durations[job_id]):
                continue
            machine_id = machines[job_id][op_idx]
            duration = durations[job_id][op_idx]
            est = max(job_ready[job_id], machine_ready[machine_id])
            # Lookahead: estimate remaining work for this job (including current operation)
            remaining = sum(durations[job_id][k] for k in range(op_idx, len(durations[job_id])))
            # Criticality: how many operations left for this job
            criticality = len(durations[job_id]) - op_idx
            # Calculate slack time based on job's remaining time and machine constraints
            slack = job_slack[job_id] if job_slack[job_id] != float('inf') else remaining
            candidates.append((est, duration, job_id, op_idx, machine_id, remaining, criticality, machine_loads[machine_id], slack))

        if not candidates:
            raise RuntimeError("No schedulable operation found.")

        # Improved selection: prioritize operations with lower EST, but consider remaining work,
        # machine load, and slack time to balance the schedule
        est, duration, job_id, op_idx, machine_id, remaining, criticality, machine_load, _ = min(
            candidates,
            key=lambda x: (
                x[0] +  # Earliest start time
                x[5] / (x[6] * 3) +  # Remaining work normalized by criticality
                x[7] * 0.1 -  # Machine load factor with smaller weight (penalize overloaded machines)
                x[8] * 0.05,  # Slack time factor with small weight (prefer operations with less flexibility)
            ),
        )
        end = est + duration
        
        # Update machine load tracking
        machine_loads[machine_id] += duration
        machine_schedules[machine_id].append({
            "job_id": job_id, "operation_index": op_idx,
            "start_time": est, "end_time": end, "duration": duration,
        })
        next_op[job_id] += 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end
        scheduled += 1

    makespan = max(job_ready) if job_ready else 0
    return {"name": instance["name"], "makespan": makespan,
            "machine_schedules": machine_schedules, "solved_by": "GreedyESTSPTBaseline",
            "family": FAMILY_PREFIX}


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
