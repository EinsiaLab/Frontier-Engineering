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
import random
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


def _evaluate_perm_fast(perm, durations, machines, num_jobs, num_machines):
    """Fast makespan evaluation."""
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    for job_id in perm:
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


def _decode_permutation(perm, durations, machines, num_jobs, num_machines):
    """Decode a job permutation list into a schedule."""
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    machine_schedules = [[] for _ in range(num_machines)]
    for job_id in perm:
        op_idx = next_op[job_id]
        if op_idx >= len(durations[job_id]):
            continue
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
        next_op[job_id] += 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end
    makespan = max(job_ready) if job_ready else 0
    return makespan, machine_schedules


def _compute_tail_times(durations):
    """Compute tail times (remaining processing time from each operation onwards)."""
    num_jobs = len(durations)
    tail = [[0] * len(durations[j]) for j in range(num_jobs)]
    for j in range(num_jobs):
        n_ops = len(durations[j])
        if n_ops > 0:
            tail[j][n_ops - 1] = durations[j][n_ops - 1]
            for i in range(n_ops - 2, -1, -1):
                tail[j][i] = durations[j][i] + tail[j][i + 1]
    return tail


def _build_perm_from_greedy(durations, machines, num_jobs, num_machines, total_operations, priority_rule):
    """Build a permutation list from a greedy dispatching rule."""
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    tail = _compute_tail_times(durations)
    num_ops_per_job = [len(durations[j]) for j in range(num_jobs)]
    perm = []

    for _ in range(total_operations):
        best_key = None
        best_job = -1
        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= num_ops_per_job[job_id]:
                continue
            machine_id = machines[job_id][op_idx]
            duration = durations[job_id][op_idx]
            est = job_ready[job_id]
            mr = machine_ready[machine_id]
            if mr > est:
                est = mr

            if priority_rule == "est_lrt":
                key = (est, -tail[job_id][op_idx], job_id)
            elif priority_rule == "est_spt":
                key = (est, duration, job_id)
            elif priority_rule == "est_lpt":
                key = (est, -duration, job_id)
            elif priority_rule == "mwkr":
                key = (-tail[job_id][op_idx], est, job_id)
            elif priority_rule == "lwkr":
                key = (tail[job_id][op_idx], est, job_id)
            elif priority_rule == "mopnr":
                key = (-(num_ops_per_job[job_id] - op_idx), est, job_id)
            elif priority_rule == "fcfs":
                key = (job_ready[job_id], est, job_id)
            else:
                key = (est, duration, job_id)

            if best_key is None or key < best_key:
                best_key = key
                best_job = job_id

        job_id = best_job
        op_idx = next_op[job_id]
        machine_id = machines[job_id][op_idx]
        duration = durations[job_id][op_idx]
        est = job_ready[job_id]
        mr = machine_ready[machine_id]
        if mr > est:
            est = mr
        end = est + duration
        perm.append(job_id)
        next_op[job_id] += 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end

    return perm


def _get_critical_blocks(perm, durations, machines, num_jobs, num_machines):
    """Get critical path and return critical blocks (consecutive ops on same machine)."""
    total_ops = len(perm)
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines

    starts = [0] * total_ops
    ends = [0] * total_ops
    op_machine = [0] * total_ops
    pred_job = [-1] * total_ops
    pred_machine = [-1] * total_ops
    last_on_machine = [-1] * num_machines
    last_on_job = [-1] * num_jobs

    for i, job_id in enumerate(perm):
        op_idx = next_op[job_id]
        mid = machines[job_id][op_idx]
        dur = durations[job_id][op_idx]
        est = job_ready[job_id]
        mr = machine_ready[mid]
        if mr > est:
            est = mr
        end = est + dur
        starts[i] = est
        ends[i] = end
        op_machine[i] = mid
        pred_job[i] = last_on_job[job_id]
        pred_machine[i] = last_on_machine[mid]
        last_on_job[job_id] = i
        last_on_machine[mid] = i
        next_op[job_id] = op_idx + 1
        job_ready[job_id] = end
        machine_ready[mid] = end

    makespan = max(job_ready)

    # Trace critical path
    critical_set = set()
    # Find last op achieving makespan
    last_idx = -1
    for i in range(total_ops - 1, -1, -1):
        if ends[i] == makespan:
            last_idx = i
            break

    stack = [last_idx]
    critical_order = []
    while stack:
        idx = stack.pop()
        if idx < 0 or idx in critical_set:
            continue
        critical_set.add(idx)
        critical_order.append(idx)
        pj = pred_job[idx]
        pm = pred_machine[idx]
        if pj >= 0 and ends[pj] == starts[idx]:
            stack.append(pj)
        if pm >= 0 and ends[pm] == starts[idx]:
            stack.append(pm)

    # Build critical blocks: consecutive critical ops on same machine
    # Sort critical ops by their position in perm
    critical_sorted = sorted(critical_set)

    blocks = []
    if critical_sorted:
        current_block = [critical_sorted[0]]
        current_machine = op_machine[critical_sorted[0]]
        for k in range(1, len(critical_sorted)):
            idx = critical_sorted[k]
            if op_machine[idx] == current_machine:
                # Check if consecutive on machine
                if pred_machine[idx] == current_block[-1]:
                    current_block.append(idx)
                else:
                    if len(current_block) >= 2:
                        blocks.append((current_machine, current_block))
                    current_block = [idx]
                    current_machine = op_machine[idx]
            else:
                if len(current_block) >= 2:
                    blocks.append((current_machine, current_block))
                current_block = [idx]
                current_machine = op_machine[idx]
        if len(current_block) >= 2:
            blocks.append((current_machine, current_block))

    return makespan, blocks, critical_set


def _n5_neighborhood(perm, durations, machines, num_jobs, num_machines):
    """Generate N5 neighborhood moves based on critical blocks."""
    makespan, blocks, critical_set = _get_critical_blocks(perm, durations, machines, num_jobs, num_machines)

    moves = []
    for machine_id, block in blocks:
        blen = len(block)
        if blen >= 2:
            # Swap first two in block
            moves.append((block[0], block[1]))
            # Swap last two in block
            moves.append((block[blen - 2], block[blen - 1]))
            # Also try swapping first with second, and second-to-last with last
            if blen >= 3:
                moves.append((block[0], block[2]))
                moves.append((block[blen - 3], block[blen - 1]))

    return makespan, moves


def _tabu_search(perm, durations, machines, num_jobs, num_machines, time_limit=20.0):
    """Tabu search using critical path neighborhood."""
    best_perm = perm[:]
    best_ms = _evaluate_perm_fast(best_perm, durations, machines, num_jobs, num_machines)
    current_perm = perm[:]
    current_ms = best_ms
    total_ops = len(perm)

    tabu_tenure = max(10, num_jobs)
    tabu_list = {}  # (i, j) -> iteration when it becomes non-tabu
    iteration = 0

    start_time = time.perf_counter()

    while time.perf_counter() - start_time < time_limit:
        iteration += 1

        # Get moves from critical path
        ms_check, moves = _n5_neighborhood(current_perm, durations, machines, num_jobs, num_machines)

        if not moves:
            # Random restart
            random.shuffle(current_perm)
            current_ms = _evaluate_perm_fast(current_perm, durations, machines, num_jobs, num_machines)
            continue

        best_move = None
        best_move_ms = float('inf')
        best_move_pair = None

        for (i, j) in moves:
            # Apply swap
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            new_ms = _evaluate_perm_fast(current_perm, durations, machines, num_jobs, num_machines)
            # Undo swap
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]

            pair = (min(i, j), max(i, j))
            is_tabu = pair in tabu_list and tabu_list[pair] > iteration

            # Aspiration: accept if better than best known
            if new_ms < best_ms:
                if new_ms < best_move_ms:
                    best_move_ms = new_ms
                    best_move = (i, j)
                    best_move_pair = pair
            elif not is_tabu:
                if new_ms < best_move_ms:
                    best_move_ms = new_ms
                    best_move = (i, j)
                    best_move_pair = pair

        if best_move is None:
            # All moves are tabu - pick the least bad
            for (i, j) in moves:
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                new_ms = _evaluate_perm_fast(current_perm, durations, machines, num_jobs, num_machines)
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                if best_move is None or new_ms < best_move_ms:
                    best_move_ms = new_ms
                    best_move = (i, j)
                    best_move_pair = (min(i, j), max(i, j))

        if best_move is not None:
            i, j = best_move
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            current_ms = best_move_ms
            tabu_list[best_move_pair] = iteration + tabu_tenure + random.randint(0, 5)

            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]

        # Periodic diversification
        if iteration % 200 == 0:
            # Clean old tabu entries
            tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}

    return best_ms, best_perm


def _multi_start_tabu(durations, machines, num_jobs, num_machines, total_operations, time_limit=45.0):
    """Multi-start tabu search with different initial solutions."""
    rules = ["est_lrt", "est_spt", "est_lpt", "mwkr", "lwkr", "mopnr", "fcfs"]

    best_ms = float('inf')
    best_perm = None

    # Generate initial solutions from dispatching rules
    initial_perms = []
    for rule in rules:
        p = _build_perm_from_greedy(durations, machines, num_jobs, num_machines, total_operations, rule)
        ms = _evaluate_perm_fast(p, durations, machines, num_jobs, num_machines)
        initial_perms.append((ms, p))
        if ms < best_ms:
            best_ms = ms
            best_perm = p[:]

    # Sort by quality
    initial_perms.sort(key=lambda x: x[0])

    start_time = time.perf_counter()
    remaining = time_limit - (time.perf_counter() - start_time)

    # Run tabu search from best initial solutions
    num_starts = min(len(initial_perms), 5)
    for idx in range(num_starts):
        elapsed = time.perf_counter() - start_time
        remaining = time_limit - elapsed
        if remaining < 2.0:
            break

        per_start_time = remaining / (num_starts - idx)
        ms_init, perm_init = initial_perms[idx]

        ms, perm = _tabu_search(perm_init, durations, machines, num_jobs, num_machines, time_limit=per_start_time)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]

    # If time remains, do another tabu from best found
    elapsed = time.perf_counter() - start_time
    remaining = time_limit - elapsed
    if remaining > 3.0:
        ms, perm = _tabu_search(best_perm, durations, machines, num_jobs, num_machines, time_limit=remaining - 0.5)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm[:]

    return best_ms, best_perm


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Advanced solver using multiple dispatching rules + tabu search."""
    durations: list[list[int]] = instance["duration_matrix"]
    machines_mat: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    total_operations = sum(len(job) for job in durations)
    num_machines = 1 + max(max(row) for row in machines_mat)

    time_limit = 50.0

    best_ms, best_perm = _multi_start_tabu(
        durations, machines_mat, num_jobs, num_machines, total_operations, time_limit=time_limit
    )

    final_ms, machine_schedules = _decode_permutation(best_perm, durations, machines_mat, num_jobs, num_machines)

    return {
        "name": instance["name"],
        "makespan": final_ms,
        "machine_schedules": machine_schedules,
        "solved_by": "TabuSearch",
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
