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
    """Greedy EST+SPT scheduler on raw benchmark matrices.
    
    Uses earliest start time with look-ahead criticality and job balance for tie-breaking.
    """
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    total_ops = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines) + 1

    # Track remaining operations per job for balance calculation
    remaining_ops = [len(job) for job in durations]
    
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    machine_schedules: list[list[dict[str, int]]] = [[] for _ in range(num_machines)]

    scheduled = 0
    while scheduled < total_ops:
        best = None
        best_key = (float('inf'), float('inf'), float('inf'), float('inf'))
        
        # Calculate balance factor for each job
        max_remaining = max(remaining_ops) if remaining_ops else 1
        
        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= len(durations[job_id]):
                continue

            machine_id = machines[job_id][op_idx]
            duration = durations[job_id][op_idx]
            est = max(job_ready[job_id], machine_ready[machine_id])
            
            # Look ahead: consider next operation's constraints
            next_op_idx = op_idx + 1
            criticality = 0
            if next_op_idx < len(durations[job_id]):
                next_machine = machines[job_id][next_op_idx]
                next_duration = durations[job_id][next_op_idx]
                next_est = max(est + duration, machine_ready[next_machine])
                # Criticality based on how constrained the next operation is
                # Simplified: just consider the time until next operation can start
                criticality = next_est - est
            
            # Job balance: prefer jobs with fewer remaining operations
            balance_factor = (max_remaining - remaining_ops[job_id]) / max_remaining
            
            # Criticality weighting based on job progress
            job_progress = (op_idx + 1) / len(durations[job_id])
            weighted_criticality = criticality * (1 + job_progress * 0.5)
            
            # Key: (earliest_start, duration, -weighted_criticality, -balance_factor)
            # Prefer operations with lower earliest start, shorter duration, less criticality,
            # and jobs that are behind in schedule
            key = (est, duration, -weighted_criticality, -balance_factor)
            
            if key < best_key:
                best_key = key
                best = (est, duration, job_id, op_idx, machine_id)

        if best is None:
            raise RuntimeError("No schedulable operation found.")

        est, duration, job_id, op_idx, machine_id = best
        end = est + duration

        machine_schedules[machine_id].append({
            "job_id": job_id,
            "operation_index": op_idx,
            "start_time": est,
            "end_time": end,
            "duration": duration,
        })

        next_op[job_id] += 1
        remaining_ops[job_id] -= 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end
        scheduled += 1

    # Apply local search improvement: try to shift operations to reduce makespan
    improved = _local_search_improvement(
        machine_schedules, durations, machines, job_ready, machine_ready
    )
    
    makespan = max(job_ready) if job_ready else 0
    return {
        "name": instance["name"],
        "makespan": makespan,
        "machine_schedules": machine_schedules,
        "solved_by": "GreedyESTSPTBaseline",
        "family": FAMILY_PREFIX,
    }


def _local_search_improvement(
    machine_schedules: list[list[dict[str, int]]], 
    durations: list[list[int]], 
    machines: list[list[int]],
    job_ready: list[int],
    machine_ready: list[int]
) -> bool:
    """Simple local search to improve schedule by shifting operations."""
    improved = False
    num_machines = len(machine_schedules)
    
    # Try to shift operations earlier where possible
    for machine_id in range(num_machines):
        machine_ops = machine_schedules[machine_id]
        for i, op in enumerate(machine_ops):
            job_id = op["job_id"]
            op_idx = op["operation_index"]
            
            # Calculate earliest possible start time
            prev_op_end = op["start_time"] - op["duration"]
            if op_idx > 0:
                prev_op_end = max(prev_op_end, job_ready[job_id] - durations[job_id][op_idx])
            
            next_op_start = op["end_time"]
            if op_idx + 1 < len(durations[job_id]):
                next_machine = machines[job_id][op_idx + 1]
                next_op_start = min(next_op_start, 
                                   next((o["start_time"] for o in machine_schedules[next_machine] 
                                        if o["job_id"] == job_id and o["operation_index"] == op_idx + 1),
                                       float('inf')))
            
            # Find earliest possible start considering machine and job constraints
            earliest_start = max(
                job_ready[job_id] - durations[job_id][op_idx],  # Job constraint
                # Machine constraint: find gap before this operation
                next((o["end_time"] for o in reversed(machine_ops[:i]) 
                     if o["end_time"] <= op["start_time"]), 0)
            )
            
            # If we can shift earlier without causing conflict
            if earliest_start < op["start_time"]:
                duration = op["duration"]
                new_start = earliest_start
                new_end = new_start + duration
                
                # Check no conflicts with neighbors
                conflict = False
                for other in machine_ops:
                    if other is op:
                        continue
                    if (new_start < other["end_time"] and new_end > other["start_time"]):
                        conflict = True
                        break
                
                if not conflict:
                    op["start_time"] = new_start
                    op["end_time"] = new_end
                    improved = True
    
    return improved


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

    instances = (
        [load_instance_by_name(args.instance)]
        if args.instance
        else load_family_instances()[: max(args.max_instances, 1)]
    )

    for instance in instances:
        start = time.perf_counter()
        result = solve_instance(instance)
        elapsed = time.perf_counter() - start
        print(f"[{FAMILY_PREFIX}] {instance['name']}: "
              f"makespan={result['makespan']} elapsed={elapsed:.4f}s")


if __name__ == "__main__":
    _cli()
# EVOLVE-BLOCK-END
