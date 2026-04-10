# EVOLVE-BLOCK-START
"""Enhanced multi-rule greedy scheduler for SWV (Storer, Wu & Vaccari, 1992).

Improvements over baseline:
- Multiple dispatch rules (SPT, LPT, MWKR, LWKR, MOPNR, combinations)
- Machine workload awareness and bottleneck detection
- Look-ahead heuristics for next operation
- Forward shift improvement post-processing
- Selects best result from all rules
"""

from __future__ import annotations

import argparse
import os
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

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


def compute_machine_workloads(durations: list[list[int]], machines: list[list[int]]) -> list[int]:
    """Compute total workload for each machine."""
    num_machines = max(max(row) for row in machines) + 1
    workloads = [0] * num_machines
    for job_id, job_durations in enumerate(durations):
        for op_idx, duration in enumerate(job_durations):
            machine_id = machines[job_id][op_idx]
            workloads[machine_id] += duration
    return workloads


def greedy_solve_with_rule(
    durations: list[list[int]],
    machines: list[list[int]],
    priority_func: Callable,
    machine_workloads: list[int],
    max_workload: int,
    bottleneck_machine: int,
) -> tuple[int, list[list[dict[str, int]]]]:
    """Run greedy scheduling with given priority function."""
    num_jobs = len(durations)
    total_operations = sum(len(job) for job in durations)
    num_machines = len(machine_workloads)

    job_total_work = [sum(d) for d in durations]

    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines

    job_remaining_work = list(job_total_work)
    job_remaining_ops = [len(d) for d in durations]
    machine_remaining = list(machine_workloads)

    machine_schedules: list[list[dict[str, int]]] = [
        [] for _ in range(num_machines)
    ]

    scheduled = 0
    while scheduled < total_operations:
        candidates: list[tuple[tuple, int, int, int, int, int]] = []

        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= len(durations[job_id]):
                continue

            machine_id = machines[job_id][op_idx]
            duration = durations[job_id][op_idx]
            est = max(job_ready[job_id], machine_ready[machine_id])

            rem_work = job_remaining_work[job_id]
            rem_ops = job_remaining_ops[job_id]
            mach_rem = machine_remaining[machine_id]
            mach_total = machine_workloads[machine_id]

            next_mach = machines[job_id][op_idx + 1] if op_idx + 1 < len(machines[job_id]) else -1
            next_dur = durations[job_id][op_idx + 1] if op_idx + 1 < len(durations[job_id]) else 0

            priority = priority_func(
                est, duration, job_id, rem_work, rem_ops,
                machine_ready[machine_id], mach_rem, mach_total,
                next_mach, next_dur, max_workload, bottleneck_machine
            )
            candidates.append((priority, est, duration, job_id, op_idx, machine_id))

        if not candidates:
            raise RuntimeError("No schedulable operation found.")

        _, est, duration, job_id, op_idx, machine_id = min(candidates, key=lambda x: x[0])
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
        job_remaining_work[job_id] -= duration
        job_remaining_ops[job_id] -= 1
        machine_remaining[machine_id] -= duration
        scheduled += 1

    makespan = max(job_ready) if job_ready else 0
    return makespan, machine_schedules


def find_critical_path(
    op_info: dict[tuple[int, int], dict],
    durations: list[list[int]],
    machines: list[list[int]],
) -> set[tuple[int, int]]:
    """Find operations on the critical path using backward pass from makespan."""
    if not op_info:
        return set()

    # Find makespan and last operation(s)
    makespan = 0
    last_ops = []
    for key, op in op_info.items():
        if op["end_time"] > makespan:
            makespan = op["end_time"]
            last_ops = [key]
        elif op["end_time"] == makespan:
            last_ops.append(key)

    critical = set()

    def trace_back(job_id: int, op_idx: int):
        """Recursively trace back through critical path."""
        key = (job_id, op_idx)
        if key in critical:
            return
        critical.add(key)

        op = op_info[key]
        start = op["start_time"]

        # Check job predecessor
        if op_idx > 0:
            pred_key = (job_id, op_idx - 1)
            if pred_key in op_info and op_info[pred_key]["end_time"] == start:
                trace_back(job_id, op_idx - 1)

        # Check machine predecessor
        mach_id = machines[job_id][op_idx]
        mach_pred = None
        mach_pred_end = 0
        for k, other_op in op_info.items():
            if k == key:
                continue
            other_mach = machines[other_op["job_id"]][other_op["operation_index"]]
            if other_mach == mach_id and other_op["end_time"] <= start:
                if other_op["end_time"] > mach_pred_end:
                    mach_pred = k
                    mach_pred_end = other_op["end_time"]

        if mach_pred and mach_pred_end == start:
            trace_back(mach_pred[0], mach_pred[1])

    for job_id, op_idx in last_ops:
        trace_back(job_id, op_idx)

    return critical


def iterative_local_search(
    machine_schedules: list[list[dict[str, int]]],
    durations: list[list[int]],
    machines: list[list[int]],
    max_iterations: int = 20,
) -> int:
    """Iterative local search with critical path focus and swap moves.

    Key improvements:
    - Critical path identification for focused search
    - Swap adjacent and non-adjacent operations on same machine
    - Forward shift to fill gaps
    - Propagate changes through job chains
    """
    num_jobs = len(durations)
    num_machines = len(machine_schedules)

    # Build operation lookup for quick access
    op_info: dict[tuple[int, int], dict] = {}
    for mach_ops in machine_schedules:
        for op in mach_ops:
            op_info[(op["job_id"], op["operation_index"])] = op

    def get_job_pred_end(job_id: int, op_idx: int) -> int:
        if op_idx == 0:
            return 0
        prev_key = (job_id, op_idx - 1)
        return op_info[prev_key]["end_time"] if prev_key in op_info else 0

    def get_job_succ_start(job_id: int, op_idx: int) -> int:
        if op_idx >= len(durations[job_id]) - 1:
            return float('inf')
        next_key = (job_id, op_idx + 1)
        return op_info[next_key]["start_time"] if next_key in op_info else float('inf')

    def compute_makespan() -> int:
        ms = 0
        for ops in machine_schedules:
            for op in ops:
                ms = max(ms, op["end_time"])
        return ms

    def propagate_forward(job_id: int, from_op_idx: int):
        """Propagate start time changes forward through job chain."""
        for op_idx in range(from_op_idx, len(durations[job_id])):
            key = (job_id, op_idx)
            if key not in op_info:
                continue
            op = op_info[key]
            pred_end = get_job_pred_end(job_id, op_idx)
            if op["start_time"] < pred_end:
                shift = pred_end - op["start_time"]
                op["start_time"] = pred_end
                op["end_time"] = op["end_time"] + shift

    def try_shift_earlier(mach_id: int, ops: list, op_idx_in_list: int) -> bool:
        """Try to shift an operation earlier on its machine."""
        op = ops[op_idx_in_list]
        job_id = op["job_id"]
        op_idx = op["operation_index"]
        duration = op["duration"]

        job_pred_end = get_job_pred_end(job_id, op_idx)
        succ_start = get_job_succ_start(job_id, op_idx)

        # Find earliest available slot
        if op_idx_in_list == 0:
            earliest_start = job_pred_end
        else:
            earliest_start = max(job_pred_end, ops[op_idx_in_list - 1]["end_time"])

        if earliest_start < op["start_time"]:
            new_end = earliest_start + duration
            if new_end <= succ_start:
                op["start_time"] = earliest_start
                op["end_time"] = new_end
                propagate_forward(job_id, op_idx + 1)
                return True
        return False

    def try_swap_ops(mach_id: int, ops: list, idx1: int, idx2: int) -> bool:
        """Try swapping two operations (not necessarily adjacent) on the same machine."""
        if idx1 >= idx2 or idx2 >= len(ops):
            return False

        op1 = ops[idx1]
        op2 = ops[idx2]

        job1, op_idx1 = op1["job_id"], op1["operation_index"]
        job2, op_idx2 = op2["job_id"], op2["operation_index"]
        dur1, dur2 = op1["duration"], op2["duration"]

        # Calculate new positions if swapped
        if idx1 == 0:
            machine_start1 = 0
        else:
            machine_start1 = ops[idx1 - 1]["end_time"]

        # op2 would go in position idx1
        job_pred_end2 = get_job_pred_end(job2, op_idx2)
        new_start2 = max(machine_start1, job_pred_end2)
        new_end2 = new_start2 + dur2

        succ_start2 = get_job_succ_start(job2, op_idx2)
        if new_end2 > succ_start2:
            return False

        # Check if op1 can fit at position idx2 (accounting for operations between)
        # For simplicity, we only do adjacent swaps
        if idx2 != idx1 + 1:
            return False

        # op1 would go second
        job_pred_end1 = get_job_pred_end(job1, op_idx1)
        new_start1 = max(new_end2, job_pred_end1)
        new_end1 = new_start1 + dur1

        succ_start1 = get_job_succ_start(job1, op_idx1)
        if new_end1 > succ_start1:
            return False

        # Check if swap improves anything
        old_end = op2["end_time"]
        if new_end1 >= old_end:
            return False

        # Apply swap
        op2["start_time"] = new_start2
        op2["end_time"] = new_end2
        op1["start_time"] = new_start1
        op1["end_time"] = new_end1

        # Swap positions in list
        ops[idx1], ops[idx2] = ops[idx2], ops[idx1]

        # Propagate changes
        propagate_forward(job1, op_idx1 + 1)
        propagate_forward(job2, op_idx2 + 1)

        return True

    def fill_gaps():
        """Fill gaps by shifting all operations earlier where possible."""
        changed = True
        iterations = 0
        while changed and iterations < 10:
            changed = False
            iterations += 1
            for mach_id, ops in enumerate(machine_schedules):
                if not ops:
                    continue
                ops.sort(key=lambda x: x["start_time"])
                for i in range(len(ops)):
                    if try_shift_earlier(mach_id, ops, i):
                        changed = True

    def rebuild_schedule():
        """Rebuild machine schedules from op_info."""
        for mach_id in range(num_machines):
            machine_schedules[mach_id] = []
        for key, op in op_info.items():
            job_id, op_idx = key
            mach_id = machines[job_id][op_idx]
            machine_schedules[mach_id].append(op)
        for ops in machine_schedules:
            ops.sort(key=lambda x: x["start_time"])

    current_makespan = compute_makespan()

    for iteration in range(max_iterations):
        improved = False

        # Phase 1: Fill gaps
        fill_gaps()

        # Phase 2: Find critical path and focus swaps there
        critical_ops = find_critical_path(op_info, durations, machines)

        # Group critical ops by machine
        critical_by_machine: dict[int, list[tuple[int, int, int]]] = {}
        for job_id, op_idx in critical_ops:
            mach_id = machines[job_id][op_idx]
            if mach_id not in critical_by_machine:
                critical_by_machine[mach_id] = []
            key = (job_id, op_idx)
            if key in op_info:
                critical_by_machine[mach_id].append((op_info[key]["start_time"], job_id, op_idx))

        # Sort each machine's critical ops by start time
        for mach_id in critical_by_machine:
            critical_by_machine[mach_id].sort()

        # Try swaps on critical path operations first
        for mach_id, crit_ops in sorted(critical_by_machine.items(), key=lambda x: -len(x[1])):
            ops = machine_schedules[mach_id]
            if len(ops) < 2:
                continue
            ops.sort(key=lambda x: x["start_time"])

            # Find indices of critical ops in the sorted list
            crit_indices = set()
            for _, job_id, op_idx in crit_ops:
                for i, op in enumerate(ops):
                    if op["job_id"] == job_id and op["operation_index"] == op_idx:
                        crit_indices.add(i)
                        break

            # Try swaps involving critical operations
            for i in sorted(crit_indices):
                if i < len(ops) - 1:
                    if try_swap_ops(mach_id, ops, i, i + 1):
                        improved = True
                        fill_gaps()
                        break
                if i > 0:
                    if try_swap_ops(mach_id, ops, i - 1, i):
                        improved = True
                        fill_gaps()
                        break
            if improved:
                break

        # Phase 3: General swap moves if no critical path improvement
        if not improved:
            for mach_id, ops in enumerate(machine_schedules):
                if len(ops) < 2:
                    continue
                ops.sort(key=lambda x: x["start_time"])
                for i in range(len(ops) - 1):
                    if try_swap_ops(mach_id, ops, i, i + 1):
                        improved = True
                        fill_gaps()
                        break
                if improved:
                    break

        # Phase 4: Try machines with latest finishing operations
        if not improved:
            machine_end_times = []
            for mach_id, ops in enumerate(machine_schedules):
                if ops:
                    max_end = max(op["end_time"] for op in ops)
                    machine_end_times.append((max_end, mach_id))
            machine_end_times.sort(reverse=True)

            for _, mach_id in machine_end_times[:5]:
                ops = machine_schedules[mach_id]
                if len(ops) < 2:
                    continue
                ops.sort(key=lambda x: x["start_time"])
                for i in range(len(ops) - 1):
                    if try_swap_ops(mach_id, ops, i, i + 1):
                        improved = True
                        fill_gaps()
                        break
                if improved:
                    break

        new_makespan = compute_makespan()
        if new_makespan < current_makespan:
            current_makespan = new_makespan
        elif not improved:
            break

    return current_makespan


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Enhanced multi-rule greedy scheduler for JSSP."""
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]

    machine_workloads = compute_machine_workloads(durations, machines)
    max_workload = max(machine_workloads) if machine_workloads else 1
    bottleneck_machine = machine_workloads.index(max_workload) if machine_workloads else 0

    # Define dispatch rules
    # Parameters: est, duration, job_id, remaining_work, remaining_ops,
    #             machine_ready, machine_remaining, machine_total_workload,
    #             next_machine, next_duration, max_workload, bottleneck_machine
    rules = [
        # Core effective rules (SPT, LPT, MWKR variants)
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, rw, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -ro, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw, -d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -ro, -d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -ro, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw // max(ro, 1), j),

        # Weighted MWKR variants
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw * 2, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw * 3, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw * 4, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw * 5, d, j),

        # Combined heuristics
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw - ro * 10, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw - ro * 5, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw * 2 - ro * 10, j),

        # Bottleneck-aware rules
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -mt, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -mrem, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -mt - rw, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -ro - mrem // 100, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw * 3 - ro * 20 - mt, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -mrem - rw, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -mrem * 2, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -mrem * 3, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -mt * 2, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -mt * 3, d, j),

        # Look-ahead rules (prefer operations going to bottleneck next)
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, 0 if nm == bm else 1, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -nd if nm == bm else 0, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -nd if nm == bm else nd, d, j),

        # Machine workload normalized rules
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -(mt * 100 // mw), d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -((rw + mt) // 2), d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw * mt // mw, j),

        # Combined machine and job awareness
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw - mt, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -ro * 5 - mt // 10, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw // 10 - mt // 10, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw, -mt, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -ro * 50, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -ro * 50 - mt // 100, d, j),

        # WINQ-inspired rules (prefer operations going to busy machines next)
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -mt if nm >= 0 else 0, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -mrem if nm >= 0 else 0, d, j),

        # Critical ratio variants
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw // max(d, 1), j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw * d // 100, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -(rw + mt) // max(d, 1), j),

        # More bottleneck-focused
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -100 if nm == bm else 0, -rw, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -50 if nm == bm else 0, -ro, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -200 if mt == mw else 0, -rw, d, j),

        # Duration-weighted remaining work
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw // max(ro, 1), -d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw * ro // 10, d, j),

        # Machine congestion awareness
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, mr, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, mr - mrem, d, j),

        # More aggressive MWKR
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw * 10, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e, -rw * 20, d, j),

        # Combined EST + work + ops
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e + rw, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e + rw // 2, d, j),
        lambda e, d, j, rw, ro, mr, mrem, mt, nm, nd, mw, bm: (e + ro * 10, d, j),
    ]

    best_makespan = float('inf')
    best_schedules = None

    for rule in rules:
        try:
            makespan, schedules = greedy_solve_with_rule(
                durations, machines, rule, machine_workloads, max_workload, bottleneck_machine
            )
            # Apply iterative local search improvement
            improved_makespan = iterative_local_search(schedules, durations, machines)
            if improved_makespan < best_makespan:
                best_makespan = improved_makespan
                best_schedules = schedules
        except Exception:
            continue

    return {
        "name": instance["name"],
        "makespan": best_makespan,
        "machine_schedules": best_schedules,
        "solved_by": "MultiRuleGreedyWithForwardShift",
        "family": FAMILY_PREFIX,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=f"Run enhanced multi-rule greedy on {FAMILY_NAME}."
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