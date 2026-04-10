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


import random


def _greedy_schedule(durations, machines, num_jobs, num_machines, total_operations, rule="spt"):
    """Build schedule using a dispatching rule. Returns (makespan, job_order)."""
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    job_order = []
    remaining_work = [sum(d) for d in durations]

    scheduled = 0
    while scheduled < total_operations:
        best = None
        best_key = None
        for jid in range(num_jobs):
            oi = next_op[jid]
            if oi >= len(durations[jid]):
                continue
            mid = machines[jid][oi]
            dur = durations[jid][oi]
            est = max(job_ready[jid], machine_ready[mid])
            if rule == "spt":
                key = (est, dur, jid)
            elif rule == "lpt":
                key = (est, -dur, jid)
            elif rule == "mwr":
                key = (est, -remaining_work[jid], jid)
            elif rule == "lwr":
                key = (est, remaining_work[jid], jid)
            elif rule == "mopnr":
                key = (est, -(len(durations[jid]) - oi), jid)
            elif rule == "fifo":
                key = (est, jid)
            else:
                key = (est, dur, jid)
            if best_key is None or key < best_key:
                best_key = key
                best = (jid, oi, mid, dur, est)

        if best is None:
            break
        jid, oi, mid, dur, est = best
        end = est + dur
        job_order.append(jid)
        next_op[jid] += 1
        job_ready[jid] = end
        machine_ready[mid] = end
        remaining_work[jid] -= dur
        scheduled += 1

    makespan = max(job_ready) if job_ready else 0
    return makespan, job_order


def _build_from_order(job_order, durations, machines, num_jobs, num_machines):
    """Build semi-active schedule from job dispatch order."""
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    ops = []
    for jid in job_order:
        oi = next_op[jid]
        if oi >= len(durations[jid]):
            continue
        mid = machines[jid][oi]
        dur = durations[jid][oi]
        est = max(job_ready[jid], machine_ready[mid])
        end = est + dur
        ops.append((jid, oi, mid, est, end, dur))
        next_op[jid] += 1
        job_ready[jid] = end
        machine_ready[mid] = end
    makespan = max(job_ready) if job_ready else 0
    return makespan, ops


def _build_from_order_fast(job_order, durations, machines, num_jobs, num_machines):
    """Build semi-active schedule from job dispatch order. Returns makespan only."""
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    for jid in job_order:
        oi = next_op[jid]
        mid = machines[jid][oi]
        dur = durations[jid][oi]
        est = job_ready[jid]
        mr = machine_ready[mid]
        if mr > est:
            est = mr
        end = est + dur
        next_op[jid] = oi + 1
        job_ready[jid] = end
        machine_ready[mid] = end
    return max(job_ready)


def _randomized_greedy(durations, machines, num_jobs, num_machines, total_operations, rng, alpha=0.3):
    """GRASP-style randomized greedy construction."""
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    job_order = []
    remaining_work = [sum(d) for d in durations]

    scheduled = 0
    while scheduled < total_operations:
        candidates = []
        for jid in range(num_jobs):
            oi = next_op[jid]
            if oi >= len(durations[jid]):
                continue
            mid = machines[jid][oi]
            dur = durations[jid][oi]
            est = max(job_ready[jid], machine_ready[mid])
            candidates.append((est, dur, remaining_work[jid], jid, oi, mid))

        if not candidates:
            break

        min_est = min(c[0] for c in candidates)
        max_est = max(c[0] for c in candidates)
        threshold = min_est + alpha * (max_est - min_est)
        rcl = [c for c in candidates if c[0] <= threshold]
        rcl.sort(key=lambda c: (-c[2], c[1], c[3]))
        pick_idx = rng.randint(0, min(2, len(rcl) - 1))
        chosen = rcl[pick_idx]

        est, dur, rw, jid, oi, mid = chosen
        end = est + dur
        job_order.append(jid)
        next_op[jid] = oi + 1
        job_ready[jid] = end
        machine_ready[mid] = end
        remaining_work[jid] -= dur
        scheduled += 1

    makespan = max(job_ready) if job_ready else 0
    return makespan, job_order


def _find_critical_positions(job_order, durations, machines, num_jobs, num_machines):
    """Find positions in job_order that correspond to critical path operations."""
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    op_starts = []
    op_ends = []
    op_machines = []

    for idx, jid in enumerate(job_order):
        oi = next_op[jid]
        mid = machines[jid][oi]
        dur = durations[jid][oi]
        est = job_ready[jid]
        mr = machine_ready[mid]
        if mr > est:
            est = mr
        end = est + dur
        op_starts.append(est)
        op_ends.append(end)
        op_machines.append(mid)
        next_op[jid] = oi + 1
        job_ready[jid] = end
        machine_ready[mid] = end

    makespan = max(job_ready)
    # Backward pass: find critical and near-critical positions
    critical = []
    threshold = makespan * 0.95  # near-critical
    for idx in range(len(job_order)):
        if op_ends[idx] >= threshold:
            critical.append(idx)
    return critical, makespan


def _simulated_annealing(job_order, durations, machines, num_jobs, num_machines, time_limit):
    """Simulated annealing on job order representation."""
    import math

    best_order = list(job_order)
    best_ms = _build_from_order_fast(best_order, durations, machines, num_jobs, num_machines)
    current = list(best_order)
    current_ms = best_ms
    rng = random.Random(42)
    n = len(current)
    start = time.perf_counter()

    temp = best_ms * 0.04
    temp_min = best_ms * 0.0003
    cooling = 0.9999
    iterations = 0
    check_interval = 500
    no_improve_count = 0
    last_best = best_ms
    while True:
        if iterations % check_interval == 0:
            elapsed = time.perf_counter() - start
            if elapsed >= time_limit:
                break
            if iterations == check_interval and elapsed > 0:
                rate = check_interval / elapsed
                check_interval = max(100, int(rate * 0.05))

        iterations += 1
        move_type = rng.random()

        if move_type < 0.4:
            i = rng.randint(0, n - 2)
            if current[i] == current[i + 1]:
                continue
            j = i + 1
        elif move_type < 0.7:
            i = rng.randint(0, n - 1)
            offset = rng.randint(1, min(30, n - 1))
            j = (i + offset) % n
            if current[i] == current[j]:
                continue
        elif move_type < 0.85:
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            saved_val = current[i]
            current.pop(i)
            current.insert(j, saved_val)
            ms = _build_from_order_fast(current, durations, machines, num_jobs, num_machines)
            delta = ms - current_ms
            if delta < 0 or (temp > 0 and rng.random() < math.exp(min(-delta / temp, 0))):
                current_ms = ms
                if ms < best_ms:
                    best_ms = ms
                    best_order = list(current)
            else:
                current.pop(j)
                current.insert(i, saved_val)
            temp *= cooling
            continue
        else:
            # Long-range swap
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j or current[i] == current[j]:
                continue

        current[i], current[j] = current[j], current[i]
        ms = _build_from_order_fast(current, durations, machines, num_jobs, num_machines)
        delta = ms - current_ms

        if delta < 0 or (temp > 0 and rng.random() < math.exp(min(-delta / temp, 0))):
            current_ms = ms
            if ms < best_ms:
                best_ms = ms
                best_order = list(current)
        else:
            current[i], current[j] = current[j], current[i]

        temp *= cooling
        if temp < temp_min:
            no_improve_count += 1
            if no_improve_count >= 3:
                # Diversify: random perturbation of best
                current = list(best_order)
                for _ in range(max(3, n // 50)):
                    a = rng.randint(0, n - 1)
                    b = rng.randint(0, n - 1)
                    current[a], current[b] = current[b], current[a]
                current_ms = _build_from_order_fast(current, durations, machines, num_jobs, num_machines)
                no_improve_count = 0
                temp = best_ms * 0.035
            else:
                temp = best_ms * 0.02
                current = list(best_order)
                current_ms = best_ms
        if best_ms < last_best:
            last_best = best_ms
            no_improve_count = 0

    return best_ms, best_order


def _local_search_swap(job_order, durations, machines, num_jobs, num_machines, time_limit):
    """Combined local search: first-improvement + simulated annealing."""
    best_order = list(job_order)
    best_ms = _build_from_order_fast(best_order, durations, machines, num_jobs, num_machines)
    n = len(best_order)
    start = time.perf_counter()

    # Quick first-improvement pass with adjacent swaps
    current = list(best_order)
    current_ms = best_ms
    improved = True
    while improved and (time.perf_counter() - start) < time_limit * 0.12:
        improved = False
        for i in range(n - 1):
            if (time.perf_counter() - start) >= time_limit * 0.12:
                break
            if current[i] == current[i + 1]:
                continue
            current[i], current[i + 1] = current[i + 1], current[i]
            ms = _build_from_order_fast(current, durations, machines, num_jobs, num_machines)
            if ms < current_ms:
                current_ms = ms
                if ms < best_ms:
                    best_ms = ms
                    best_order = list(current)
                improved = True
            else:
                current[i], current[i + 1] = current[i + 1], current[i]

    # Simulated annealing for the remaining time
    remaining = time_limit - (time.perf_counter() - start)
    if remaining > 0.2:
        ms, order = _simulated_annealing(best_order, durations, machines, num_jobs, num_machines, remaining)
        if ms < best_ms:
            best_ms = ms
            best_order = order

    return best_ms, best_order


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Multi-rule greedy + local search scheduler."""
    durations: list[list[int]] = instance["duration_matrix"]
    machines_mat: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    total_operations = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines_mat) + 1

    t0 = time.perf_counter()
    time_budget = 18.0  # seconds per instance

    # Try multiple dispatching rules
    rules = ["spt", "lpt", "mwr", "lwr", "mopnr", "fifo"]
    best_ms = float("inf")
    best_order = None
    all_starts = []

    for rule in rules:
        ms, order = _greedy_schedule(durations, machines_mat, num_jobs, num_machines, total_operations, rule)
        all_starts.append((ms, order))
        if ms < best_ms:
            best_ms = ms
            best_order = order

    # GRASP: randomized greedy constructions
    rng = random.Random(123)
    for _ in range(20):
        alpha = rng.uniform(0.05, 0.6)
        ms, order = _randomized_greedy(durations, machines_mat, num_jobs, num_machines, total_operations, rng, alpha)
        all_starts.append((ms, order))
        if ms < best_ms:
            best_ms = ms
            best_order = order

    # Sort starting points by quality and pick diverse top ones
    all_starts.sort(key=lambda x: x[0])
    # Pick top 3 distinct starting points for multi-start SA
    sa_starts = []
    seen_ms = set()
    for ms_val, order_val in all_starts:
        if ms_val not in seen_ms and len(sa_starts) < 3:
            sa_starts.append((ms_val, order_val))
            seen_ms.add(ms_val)

    # Multi-start SA: divide time among top starting points
    elapsed = time.perf_counter() - t0
    remaining = time_budget - elapsed
    if remaining > 0.5 and sa_starts:
        n_starts = len(sa_starts)
        # Give more time to the best start
        time_fracs = [0.55, 0.30, 0.15][:n_starts]
        total_frac = sum(time_fracs)
        time_fracs = [f / total_frac for f in time_fracs]

        for idx, (start_ms, start_order) in enumerate(sa_starts):
            elapsed_now = time.perf_counter() - t0
            rem = time_budget - elapsed_now
            if rem < 0.3:
                break
            alloc = rem * time_fracs[idx] / sum(time_fracs[idx:]) * 0.95
            alloc = min(alloc, rem * 0.95)
            ms, order = _local_search_swap(start_order, durations, machines_mat, num_jobs, num_machines, alloc)
            if ms < best_ms:
                best_ms = ms
                best_order = order

    # Build final schedule
    _, ops = _build_from_order(best_order, durations, machines_mat, num_jobs, num_machines)

    machine_schedules: list[list[dict[str, int]]] = [[] for _ in range(num_machines)]
    for jid, oi, mid, st, end, dur in ops:
        machine_schedules[mid].append({
            "job_id": jid,
            "operation_index": oi,
            "start_time": st,
            "end_time": end,
            "duration": dur,
        })

    return {
        "name": instance["name"],
        "makespan": best_ms,
        "machine_schedules": machine_schedules,
        "solved_by": "MultiRuleGreedyLS",
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
