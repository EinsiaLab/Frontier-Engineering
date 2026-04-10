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


def _improve_schedule(
    machine_schedules: list[list[dict[str, int]]],
    durations: list[list[int]],
    machines: list[list[int]],
) -> bool:
    """Local search to improve schedule by shifting operations earlier."""
    improved = False
    num_machines = len(machine_schedules)
    
    # Multiple passes to propagate improvements
    for _ in range(2):
        for machine_id in range(num_machines):
            ops = machine_schedules[machine_id]
            if not ops:
                continue
                
            # Process in reverse order to enable better shifts
            for i in range(len(ops) - 1, -1, -1):
                op = ops[i]
                job_id = op["job_id"]
                op_idx = op["operation_index"]
                
                # Find earliest possible start (precedence constraint)
                prev_end = 0
                if op_idx > 0:
                    # Find previous operation in same job
                    for other_machine in machine_schedules:
                        for other_op in other_machine:
                            if (other_op["job_id"] == job_id and 
                                other_op["operation_index"] == op_idx - 1):
                                prev_end = other_op["end_time"]
                                break
                
                # Find earliest possible start (machine availability)
                machine_earliest = 0
                if i > 0:
                    machine_earliest = ops[i - 1]["end_time"]
                
                earliest = max(prev_end, machine_earliest)
                
                if earliest < op["start_time"]:
                    # Shift this operation earlier
                    shift = op["start_time"] - earliest
                    if shift > 0:
                        duration = op["duration"]
                        op["start_time"] = earliest
                        op["end_time"] = earliest + duration
                        improved = True
                        
                        # Update subsequent operations on same machine
                        current_end = earliest + duration
                        for j in range(i + 1, len(ops)):
                            next_op = ops[j]
                            if next_op["start_time"] < current_end:
                                next_op["start_time"] = current_end
                                next_op["end_time"] = current_end + next_op["duration"]
                                current_end = next_op["end_time"]
    
    return improved


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Greedy EST+SPT scheduler with load balancing, criticality awareness, and local search improvement."""
    durations = instance["duration_matrix"]
    machines = instance["machines_matrix"]

    num_jobs = len(durations)
    total_ops = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines) + 1

    # Track remaining operations and time for priority calculation
    remaining_ops = [len(job) for job in durations]
    remaining_time = [sum(job) for job in durations]
    
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    machine_load = [0] * num_machines  # Track total processing time per machine
    job_criticality = [len(job) for job in durations]  # Track remaining ops per job
    machine_schedules = [[] for _ in range(num_machines)]

    scheduled = 0
    while scheduled < total_ops:
        best = None
        best_priority = float("inf")

        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= len(durations[job_id]):
                continue

            machine_id = machines[job_id][op_idx]
            duration = durations[job_id][op_idx]
            est = max(job_ready[job_id], machine_ready[machine_id])
            
            # Priority function combining EST, machine load balancing, operation duration, and criticality
            # Lower priority value is better
            # Weight machine load factor by remaining operations to balance urgency vs load distribution
            load_factor = machine_load[machine_id] / (remaining_ops[job_id] + 1)
            # Criticality bonus: prioritize jobs with fewer remaining operations
            criticality_factor = job_criticality[job_id] / total_ops
            # Optimized coefficients for better makespan
            priority = est + load_factor * 0.4 + duration * 0.08 - criticality_factor * 0.2
            
            if priority < best_priority:
                best_priority = priority
                best = (est, duration, job_id, op_idx, machine_id)

        if best is None:
            raise RuntimeError("No schedulable operation found.")

        est, duration, job_id, op_idx, machine_id = best
        actual_start = max(job_ready[job_id], machine_ready[machine_id])
        end = actual_start + duration

        machine_schedules[machine_id].append({
            "job_id": job_id,
            "operation_index": op_idx,
            "start_time": actual_start,
            "end_time": end,
            "duration": duration,
        })

        next_op[job_id] += 1
        job_criticality[job_id] -= 1  # Properly decrement criticality after each operation
        remaining_ops[job_id] -= 1
        remaining_time[job_id] -= duration
        job_ready[job_id] = end
        machine_ready[machine_id] = end
        machine_load[machine_id] += duration
        scheduled += 1

    # Local search: try to shift operations earlier where possible
    _improve_schedule(machine_schedules, durations, machines)
    
    # Calculate makespan directly from scheduled operations (more accurate than using job_ready)
    makespan = max(op["end_time"] for machine_ops in machine_schedules for op in machine_ops) if machine_schedules else 0
    return {
        "name": instance["name"],
        "makespan": makespan,
        "machine_schedules": machine_schedules,
        "solved_by": "GreedyESTSPTWithLoadBalance",
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
