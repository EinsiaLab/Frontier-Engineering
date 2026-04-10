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
import random
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


def _compute_makespan_from_perm(durations, machines, num_machines, job_order):
    """Compute makespan by scheduling jobs in given operation order (permutation list)."""
    num_jobs = len(durations)
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines

    for job_id in job_order:
        op_idx = next_op[job_id]
        machine_id = machines[job_id][op_idx]
        duration = durations[job_id][op_idx]
        est = job_ready[job_id]
        mr = machine_ready[machine_id]
        if mr > est:
            est = mr
        end = est + duration
        job_ready[job_id] = end
        machine_ready[machine_id] = end
        next_op[job_id] = op_idx + 1

    return max(job_ready)


def _build_schedule_from_perm(durations, machines, num_machines, job_order):
    """Build full schedule from job_order permutation."""
    num_jobs = len(durations)
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    machine_schedules = [[] for _ in range(num_machines)]

    for job_id in job_order:
        op_idx = next_op[job_id]
        machine_id = machines[job_id][op_idx]
        duration = durations[job_id][op_idx]
        est = job_ready[job_id]
        mr = machine_ready[machine_id]
        if mr > est:
            est = mr
        end = est + duration

        machine_schedules[machine_id].append({
            "job_id": job_id,
            "operation_index": op_idx,
            "start_time": est,
            "end_time": end,
            "duration": duration,
        })

        job_ready[job_id] = end
        machine_ready[machine_id] = end
        next_op[job_id] = op_idx + 1

    makespan = max(job_ready)
    return makespan, machine_schedules


def _greedy_dispatch(durations, machines, num_machines, priority_key):
    """Generic greedy dispatcher. Returns (makespan, job_order)."""
    num_jobs = len(durations)
    total_operations = sum(len(job) for job in durations)
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    job_order = []

    for _ in range(total_operations):
        best = None
        best_key = None
        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= len(durations[job_id]):
                continue
            machine_id = machines[job_id][op_idx]
            duration = durations[job_id][op_idx]
            est = job_ready[job_id]
            mr = machine_ready[machine_id]
            if mr > est:
                est = mr
            key = priority_key(est, duration, job_id, op_idx, machine_id)
            if best_key is None or key < best_key:
                best_key = key
                best = (job_id, op_idx, machine_id, duration, est)

        job_id, op_idx, machine_id, duration, est = best
        end = est + duration
        job_order.append(job_id)
        next_op[job_id] = op_idx + 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end

    makespan = max(job_ready)
    return makespan, job_order


def _compute_remaining_work(durations):
    """For each job, compute remaining work from each operation onwards."""
    num_jobs = len(durations)
    remaining = [None] * num_jobs
    for j in range(num_jobs):
        ops = durations[j]
        n = len(ops)
        rem = [0] * (n + 1)
        for k in range(n - 1, -1, -1):
            rem[k] = rem[k + 1] + ops[k]
        remaining[j] = rem
    return remaining


def _critical_path_lower_bound(durations, machines, num_machines):
    """Compute a simple lower bound: max of job lengths and machine loads."""
    lb = 0
    # Job length lower bound
    for j in range(len(durations)):
        s = sum(durations[j])
        if s > lb:
            lb = s
    # Machine load lower bound
    machine_load = [0] * num_machines
    for j in range(len(durations)):
        for k in range(len(durations[j])):
            machine_load[machines[j][k]] += durations[j][k]
    ml = max(machine_load)
    if ml > lb:
        lb = ml
    return lb


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    num_machines = max(max(row) for row in machines) + 1
    total_operations = sum(len(job) for job in durations)

    remaining_work = _compute_remaining_work(durations)

    # Try multiple dispatch rules and keep the best
    priority_rules = [
        # EST + SPT
        lambda est, dur, jid, oidx, mid: (est, dur, jid),
        # EST + LPT
        lambda est, dur, jid, oidx, mid: (est, -dur, jid),
        # EST + MWKR (most work remaining)
        lambda est, dur, jid, oidx, mid: (est, -remaining_work[jid][oidx], jid),
        # MWKR first
        lambda est, dur, jid, oidx, mid: (-remaining_work[jid][oidx], est, jid),
        # EST + MWKR combined
        lambda est, dur, jid, oidx, mid: (est - remaining_work[jid][oidx], jid),
        # SPT only
        lambda est, dur, jid, oidx, mid: (dur, est, jid),
        # LPT only  
        lambda est, dur, jid, oidx, mid: (-dur, est, jid),
    ]

    best_makespan = float('inf')
    best_order = None

    for rule in priority_rules:
        ms, order = _greedy_dispatch(durations, machines, num_machines, rule)
        if ms < best_makespan:
            best_makespan = ms
            best_order = order

    # Now apply local search (simulated annealing on the job_order permutation)
    # Use a time budget
    time_limit = 25.0  # seconds per instance
    start_time = time.perf_counter()

    current_order = list(best_order)
    current_ms = best_makespan
    best_ms_global = best_makespan
    best_order_global = list(best_order)

    rng = random.Random(42)
    
    # Precompute for speed
    dur_flat = durations
    mach_flat = machines
    n_ops = total_operations

    # Temperature for SA
    T = current_ms * 0.05
    T_min = current_ms * 0.0001
    
    lb = _critical_path_lower_bound(durations, machines, num_machines)
    
    iterations = 0
    cooling_rate = 0.99995
    
    # For large instances, use block moves
    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed > time_limit:
            break
        if best_ms_global <= lb:
            break
            
        # Generate neighbor by swapping two adjacent operations of different jobs
        # or swapping two random positions
        if rng.random() < 0.5:
            # Swap two random positions
            i = rng.randint(0, n_ops - 1)
            j = rng.randint(0, n_ops - 2)
            if j >= i:
                j += 1
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
            new_ms = _compute_makespan_from_perm(dur_flat, mach_flat, num_machines, current_order)
            
            delta = new_ms - current_ms
            if delta <= 0 or (T > T_min and rng.random() < _fast_exp(-delta / T)):
                current_ms = new_ms
                if new_ms < best_ms_global:
                    best_ms_global = new_ms
                    best_order_global = list(current_order)
            else:
                current_order[i], current_order[j] = current_order[j], current_order[i]
        else:
            # Block shift: take a random element and insert it elsewhere
            i = rng.randint(0, n_ops - 1)
            j = rng.randint(0, n_ops - 2)
            if j >= i:
                j += 1
            val = current_order[i]
            current_order.pop(i)
            current_order.insert(j, val)
            
            new_ms = _compute_makespan_from_perm(dur_flat, mach_flat, num_machines, current_order)
            
            delta = new_ms - current_ms
            if delta <= 0 or (T > T_min and rng.random() < _fast_exp(-delta / T)):
                current_ms = new_ms
                if new_ms < best_ms_global:
                    best_ms_global = new_ms
                    best_order_global = list(current_order)
            else:
                # Undo
                current_order.pop(j if j < i else j)
                current_order.insert(i, val)
        
        T *= cooling_rate
        if T < T_min:
            # Reheat
            T = best_ms_global * 0.02
            current_order = list(best_order_global)
            current_ms = best_ms_global
        
        iterations += 1

    # Build final schedule
    makespan, machine_schedules = _build_schedule_from_perm(
        durations, machines, num_machines, best_order_global
    )

    return {
        "name": instance["name"],
        "makespan": makespan,
        "machine_schedules": machine_schedules,
        "solved_by": "SA_LocalSearch",
        "family": FAMILY_PREFIX,
    }


import math

def _fast_exp(x):
    if x < -700:
        return 0.0
    return math.exp(x)


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
