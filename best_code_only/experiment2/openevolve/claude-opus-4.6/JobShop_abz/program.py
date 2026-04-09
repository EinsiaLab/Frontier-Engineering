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


import random


def _greedy_schedule(durations, machines, num_jobs, num_machines, total_operations, rule="spt"):
    """Build a schedule using a greedy dispatching rule. Returns (makespan, job_order)."""
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    job_order = []

    # Precompute remaining work for MWR rule
    remaining_work = [sum(durations[j]) for j in range(num_jobs)]

    scheduled = 0
    while scheduled < total_operations:
        # Find minimum EST among all candidates
        min_est = float('inf')
        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= len(durations[job_id]):
                continue
            machine_id = machines[job_id][op_idx]
            est = max(job_ready[job_id], machine_ready[machine_id])
            if est < min_est:
                min_est = est

        # Collect candidates within a window of min_est
        candidates = []
        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= len(durations[job_id]):
                continue
            machine_id = machines[job_id][op_idx]
            duration = durations[job_id][op_idx]
            est = max(job_ready[job_id], machine_ready[machine_id])
            if est <= min_est + duration:  # lookahead window
                candidates.append((est, duration, job_id, op_idx, machine_id))

        if not candidates:
            raise RuntimeError("No schedulable operation found.")

        if rule == "spt":
            best = min(candidates, key=lambda x: (x[1], x[0], x[2]))
        elif rule == "lpt":
            best = min(candidates, key=lambda x: (-x[1], x[0], x[2]))
        elif rule == "mwr":
            best = min(candidates, key=lambda x: (-remaining_work[x[2]], x[0], x[2]))
        elif rule == "lor":
            # Least operations remaining
            best = min(candidates, key=lambda x: (len(durations[x[2]]) - next_op[x[2]], x[0], x[2]))
        elif rule == "est":
            best = min(candidates, key=lambda x: (x[0], x[1], x[2]))
        elif rule == "est_spt":
            best = min(candidates, key=lambda x: (x[0], x[1], x[2]))
        else:
            best = min(candidates, key=lambda x: (x[0], x[1], x[2]))

        est, duration, job_id, op_idx, machine_id = best
        end = est + duration

        next_op[job_id] += 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end
        remaining_work[job_id] -= duration
        job_order.append(job_id)
        scheduled += 1

    makespan = max(job_ready) if job_ready else 0
    return makespan, job_order


def _evaluate_order(job_order, durations, machines, num_jobs, num_machines):
    """Evaluate a job operation order and return (makespan, job_ready, machine_ready, start_times)."""
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    start_times = []

    for job_id in job_order:
        op_idx = next_op[job_id]
        machine_id = machines[job_id][op_idx]
        duration = durations[job_id][op_idx]
        est = max(job_ready[job_id], machine_ready[machine_id])
        end = est + duration
        start_times.append((job_id, op_idx, machine_id, est, end, duration))
        next_op[job_id] += 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end

    makespan = max(job_ready) if job_ready else 0
    return makespan, start_times


def _build_machine_schedules(start_times, num_machines):
    machine_schedules = [[] for _ in range(num_machines)]
    for job_id, op_idx, machine_id, est, end, duration in start_times:
        machine_schedules[machine_id].append({
            "job_id": job_id,
            "operation_index": op_idx,
            "start_time": est,
            "end_time": end,
            "duration": duration,
        })
    return machine_schedules


def _evaluate_order_fast(job_order, durations, machines, num_jobs, num_machines):
    """Fast makespan-only evaluation without building start_times list."""
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
        next_op[job_id] = op_idx + 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end
    return max(job_ready)


def _find_critical_ops(job_order, durations, machines, num_jobs, num_machines):
    """Find indices in job_order on or near the critical path using forward+backward pass."""
    n = len(job_order)
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    starts = [0] * n
    ends = [0] * n
    op_jobs = [0] * n
    op_machines = [0] * n
    op_durations = [0] * n

    for idx in range(n):
        job_id = job_order[idx]
        op_idx = next_op[job_id]
        machine_id = machines[job_id][op_idx]
        duration = durations[job_id][op_idx]
        est = job_ready[job_id]
        mr = machine_ready[machine_id]
        if mr > est:
            est = mr
        end = est + duration
        starts[idx] = est
        ends[idx] = end
        op_jobs[idx] = job_id
        op_machines[idx] = machine_id
        op_durations[idx] = duration
        next_op[job_id] = op_idx + 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end

    makespan = max(job_ready)

    # Backward pass: compute slack
    job_latest = [makespan] * num_jobs
    machine_latest = [makespan] * num_machines
    slack = [0] * n

    for idx in range(n - 1, -1, -1):
        job_id = op_jobs[idx]
        machine_id = op_machines[idx]
        duration = op_durations[idx]
        latest_end = job_latest[job_id]
        ml = machine_latest[machine_id]
        if ml < latest_end:
            latest_end = ml
        latest_start = latest_end - duration
        slack[idx] = latest_start - starts[idx]
        job_latest[job_id] = latest_start
        machine_latest[machine_id] = latest_start

    max_slack = max(1, makespan // 20)
    critical = [i for i in range(n) if slack[i] <= max_slack]
    if len(critical) < 2:
        threshold = makespan * 0.85
        critical = [i for i in range(n) if ends[i] >= threshold]
    return critical


def _local_search_v3(job_order, durations, machines, num_jobs, num_machines, time_limit):
    """Local search with critical path focus and simulated annealing."""
    best_order = list(job_order)
    best_makespan = _evaluate_order_fast(best_order, durations, machines, num_jobs, num_machines)

    start_time = time.perf_counter()
    n = len(best_order)
    rng = random.Random(42)

    current_order = list(best_order)
    current_makespan = best_makespan

    critical_indices = None
    last_critical_update = -1
    no_improve_count = 0

    iteration = 0
    check_interval = 200

    while True:
        if iteration % check_interval == 0:
            if time.perf_counter() - start_time >= time_limit:
                break

        # Update critical indices periodically
        if critical_indices is None or iteration - last_critical_update > 3000:
            critical_indices = _find_critical_ops(current_order, durations, machines, num_jobs, num_machines)
            if len(critical_indices) < 2:
                critical_indices = list(range(n))
            last_critical_update = iteration

        use_critical = rng.random() < 0.75 and len(critical_indices) > 1

        move = rng.randint(0, 2)
        if move == 0:
            # Adjacent swap
            if use_critical:
                i = rng.choice(critical_indices)
                i = min(i, n - 2)
            else:
                i = rng.randint(0, n - 2)
            if current_order[i] == current_order[i + 1]:
                iteration += 1
                continue
            current_order[i], current_order[i + 1] = current_order[i + 1], current_order[i]
            ms = _evaluate_order_fast(current_order, durations, machines, num_jobs, num_machines)
            if ms < best_makespan:
                best_makespan = ms
                best_order = list(current_order)
                current_makespan = ms
                no_improve_count = 0
            elif ms <= current_makespan:
                current_makespan = ms
                no_improve_count += 1
            else:
                current_order[i], current_order[i + 1] = current_order[i + 1], current_order[i]
                no_improve_count += 1
        elif move == 1:
            # Swap two positions
            if use_critical:
                i = rng.choice(critical_indices)
                j = rng.randint(max(0, i - 25), min(n - 1, i + 25))
            else:
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
            if i == j or current_order[i] == current_order[j]:
                iteration += 1
                continue
            current_order[i], current_order[j] = current_order[j], current_order[i]
            ms = _evaluate_order_fast(current_order, durations, machines, num_jobs, num_machines)
            if ms < best_makespan:
                best_makespan = ms
                best_order = list(current_order)
                current_makespan = ms
                no_improve_count = 0
            elif ms <= current_makespan:
                current_makespan = ms
                no_improve_count += 1
            else:
                current_order[i], current_order[j] = current_order[j], current_order[i]
                no_improve_count += 1
        else:
            # Insert move
            if use_critical:
                i = rng.choice(critical_indices)
                j = rng.randint(max(0, i - 20), min(n - 1, i + 20))
            else:
                i = rng.randint(0, n - 1)
                j = rng.randint(0, n - 1)
            if i == j:
                iteration += 1
                continue
            val = current_order[i]
            if i < j:
                for k in range(i, j):
                    current_order[k] = current_order[k + 1]
            else:
                for k in range(i, j, -1):
                    current_order[k] = current_order[k - 1]
            current_order[j] = val
            ms = _evaluate_order_fast(current_order, durations, machines, num_jobs, num_machines)
            if ms < best_makespan:
                best_makespan = ms
                best_order = list(current_order)
                current_makespan = ms
                no_improve_count = 0
            elif ms <= current_makespan:
                current_makespan = ms
                no_improve_count += 1
            else:
                if i < j:
                    for k in range(j, i, -1):
                        current_order[k] = current_order[k - 1]
                else:
                    for k in range(j, i):
                        current_order[k] = current_order[k + 1]
                current_order[i] = val
                no_improve_count += 1

        # Restart from best if stuck
        if no_improve_count > 8000:
            current_order = list(best_order)
            current_makespan = best_makespan
            for _ in range(n // 4):
                i = rng.randint(0, n - 2)
                if current_order[i] != current_order[i + 1]:
                    current_order[i], current_order[i + 1] = current_order[i + 1], current_order[i]
            current_makespan = _evaluate_order_fast(current_order, durations, machines, num_jobs, num_machines)
            no_improve_count = 0
            critical_indices = None

        iteration += 1

    return best_makespan, best_order


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Multi-rule greedy + local search scheduler."""
    durations: list[list[int]] = instance["duration_matrix"]
    machines_mat: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    total_operations = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines_mat) + 1

    rules = ["spt", "lpt", "mwr", "lor", "est"]
    
    best_makespan = float('inf')
    best_order = None
    
    # Try multiple dispatching rules
    for rule in rules:
        ms, order = _greedy_schedule(durations, machines_mat, num_jobs, num_machines, total_operations, rule)
        if ms < best_makespan:
            best_makespan = ms
            best_order = order

    # Apply local search with time budget
    time_budget = 40.0  # seconds per instance (5 instances * 40s = 200s, well under 300s)

    all_starts = []
    for rule in rules:
        ms, order = _greedy_schedule(durations, machines_mat, num_jobs, num_machines, total_operations, rule)
        all_starts.append((ms, order))
    all_starts.sort(key=lambda x: x[0])
    seen_ms = set()
    unique_starts = []
    for ms, order in all_starts:
        if ms not in seen_ms:
            seen_ms.add(ms)
            unique_starts.append((ms, order))

    num_starts = min(4, len(unique_starts))
    per_start_time = time_budget / max(num_starts, 1)

    for ms_init, order_init in unique_starts[:num_starts]:
        ls_makespan, ls_order = _local_search_v3(
            order_init, durations, machines_mat, num_jobs, num_machines, per_start_time
        )
        if ls_makespan < best_makespan:
            best_makespan = ls_makespan
            best_order = ls_order

    # Build final schedule
    _, start_times = _evaluate_order(best_order, durations, machines_mat, num_jobs, num_machines)
    machine_schedules = _build_machine_schedules(start_times, num_machines)

    return {
        "name": instance["name"],
        "makespan": best_makespan,
        "machine_schedules": machine_schedules,
        "solved_by": "MultiRuleGreedy+LocalSearch",
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
