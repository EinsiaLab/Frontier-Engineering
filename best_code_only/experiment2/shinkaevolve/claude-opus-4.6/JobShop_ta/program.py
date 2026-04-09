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


import random as _random
import math as _math


def _greedy_schedule(
    durations: list[list[int]],
    machines: list[list[int]],
    num_jobs: int,
    num_machines: int,
    total_operations: int,
    rule: str = "spt",
    remaining_work: list[list[int]] | None = None,
) -> tuple[int, list[list[tuple[int, int]]]]:
    """Build a schedule using a greedy dispatching rule.

    Returns (makespan, machine_orders) where machine_orders[m] is list of
    (job_id, op_idx) in scheduled order for machine m.
    """
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    machine_orders: list[list[tuple[int, int]]] = [[] for _ in range(num_machines)]

    scheduled = 0
    while scheduled < total_operations:
        best = None
        best_key = None

        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= len(durations[job_id]):
                continue

            machine_id = machines[job_id][op_idx]
            dur = durations[job_id][op_idx]
            est = job_ready[job_id]
            mr = machine_ready[machine_id]
            if mr > est:
                est = mr

            if rule == "spt":
                key = (est, dur, job_id)
            elif rule == "lpt":
                key = (est, -dur, job_id)
            elif rule == "mwr":
                rw = remaining_work[job_id][op_idx] if remaining_work else dur
                key = (est, -rw, job_id)
            elif rule == "lwr":
                rw = remaining_work[job_id][op_idx] if remaining_work else dur
                key = (est, rw, job_id)
            elif rule == "mopnr":
                ops_left = len(durations[job_id]) - op_idx
                key = (est, -ops_left, job_id)
            elif rule == "lopnr":
                ops_left = len(durations[job_id]) - op_idx
                key = (est, ops_left, job_id)
            elif rule == "est_random":
                key = (est, _random.random(), job_id)
            else:
                key = (est, dur, job_id)

            if best_key is None or key < best_key:
                best_key = key
                best = (job_id, op_idx, machine_id, dur, est)

        if best is None:
            break

        job_id, op_idx, machine_id, dur, est = best
        end = est + dur

        machine_orders[machine_id].append((job_id, op_idx))
        next_op[job_id] += 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end
        scheduled += 1

    makespan = max(job_ready) if job_ready else 0
    return makespan, machine_orders


def _compute_times(
    durations: list[list[int]],
    machines: list[list[int]],
    num_jobs: int,
    num_machines: int,
    machine_orders: list[list[tuple[int, int]]],
) -> tuple[int, list[list[int]], list[list[int]]]:
    """Compute start/end times from machine_orders using topological scheduling.
    Returns (makespan, op_start, op_end).
    """
    num_ops = [len(durations[j]) for j in range(num_jobs)]

    # Build flat arrays for speed
    # For each operation (j, o), we need:
    #   - job predecessor end time
    #   - machine predecessor end time
    # We process in topological order using a ready-set approach.

    # Flatten: assign global id to each (job, op)
    # op_gid[j][o] = global id
    total_ops = sum(num_ops)
    op_gid = []
    gid = 0
    for j in range(num_jobs):
        ids = list(range(gid, gid + num_ops[j]))
        op_gid.append(ids)
        gid += num_ops[j]

    # For each global op, store (job, op_idx, duration)
    g_job = [0] * total_ops
    g_opidx = [0] * total_ops
    g_dur = [0] * total_ops
    g_start = [0] * total_ops
    g_end = [0] * total_ops

    for j in range(num_jobs):
        for o in range(num_ops[j]):
            g = op_gid[j][o]
            g_job[g] = j
            g_opidx[g] = o
            g_dur[g] = durations[j][o]

    # Machine predecessor: for each op in machine_orders, the previous op on same machine
    # g_mpred[g] = global id of machine predecessor, or -1
    g_mpred = [-1] * total_ops

    for m in range(num_machines):
        prev_g = -1
        for job_id, op_idx in machine_orders[m]:
            g = op_gid[job_id][op_idx]
            g_mpred[g] = prev_g
            prev_g = g

    # Job predecessor: g_jpred[g] = global id of job predecessor, or -1
    g_jpred = [-1] * total_ops
    for j in range(num_jobs):
        for o in range(1, num_ops[j]):
            g_jpred[op_gid[j][o]] = op_gid[j][o - 1]

    # Topological order: use in-degree counting
    # in-degree = number of predecessors not yet processed
    in_deg = [0] * total_ops
    for g in range(total_ops):
        if g_jpred[g] >= 0:
            in_deg[g] += 1
        if g_mpred[g] >= 0:
            in_deg[g] += 1

    # Successors for topological sort
    g_jsucc = [-1] * total_ops
    g_msucc = [-1] * total_ops
    for j in range(num_jobs):
        for o in range(num_ops[j] - 1):
            g_jsucc[op_gid[j][o]] = op_gid[j][o + 1]
    for m in range(num_machines):
        for i in range(len(machine_orders[m]) - 1):
            j1, o1 = machine_orders[m][i]
            j2, o2 = machine_orders[m][i + 1]
            g_msucc[op_gid[j1][o1]] = op_gid[j2][o2]

    # BFS
    queue = []
    for g in range(total_ops):
        if in_deg[g] == 0:
            queue.append(g)

    head = 0
    while head < len(queue):
        g = queue[head]
        head += 1

        jp = g_jpred[g]
        mp = g_mpred[g]
        s = 0
        if jp >= 0 and g_end[jp] > s:
            s = g_end[jp]
        if mp >= 0 and g_end[mp] > s:
            s = g_end[mp]
        g_start[g] = s
        g_end[g] = s + g_dur[g]

        js = g_jsucc[g]
        if js >= 0:
            in_deg[js] -= 1
            if in_deg[js] == 0:
                queue.append(js)
        ms = g_msucc[g]
        if ms >= 0:
            in_deg[ms] -= 1
            if in_deg[ms] == 0:
                queue.append(ms)

    makespan = 0
    for g in range(total_ops):
        if g_end[g] > makespan:
            makespan = g_end[g]

    # Convert back to 2D arrays
    op_start = [[0] * num_ops[j] for j in range(num_jobs)]
    op_end = [[0] * num_ops[j] for j in range(num_jobs)]
    for j in range(num_jobs):
        for o in range(num_ops[j]):
            g = op_gid[j][o]
            op_start[j][o] = g_start[g]
            op_end[j][o] = g_end[g]

    return makespan, op_start, op_end


def _compute_makespan_only(
    durations: list[list[int]],
    num_jobs: int,
    num_machines: int,
    num_ops: list[int],
    op_gid: list[list[int]],
    total_ops: int,
    machine_orders: list[list[tuple[int, int]]],
) -> int:
    """Fast makespan-only computation using BFS topological sort."""
    g_end = [0] * total_ops
    g_dur = [0] * total_ops
    g_jpred = [-1] * total_ops
    g_mpred = [-1] * total_ops
    g_jsucc = [-1] * total_ops
    g_msucc = [-1] * total_ops
    in_deg = [0] * total_ops

    for j in range(num_jobs):
        dj = durations[j]
        gj = op_gid[j]
        nj = num_ops[j]
        for o in range(nj):
            g_dur[gj[o]] = dj[o]
        for o in range(1, nj):
            g_jpred[gj[o]] = gj[o - 1]
            in_deg[gj[o]] += 1
            g_jsucc[gj[o - 1]] = gj[o]

    for m in range(num_machines):
        mo = machine_orders[m]
        lm = len(mo)
        if lm == 0:
            continue
        prev_g = op_gid[mo[0][0]][mo[0][1]]
        for i in range(1, lm):
            j2, o2 = mo[i]
            cur_g = op_gid[j2][o2]
            g_mpred[cur_g] = prev_g
            g_msucc[prev_g] = cur_g
            in_deg[cur_g] += 1
            prev_g = cur_g

    queue = [0] * total_ops
    head = 0
    tail = 0
    for g in range(total_ops):
        if in_deg[g] == 0:
            queue[tail] = g
            tail += 1

    while head < tail:
        g = queue[head]
        head += 1

        jp = g_jpred[g]
        mp = g_mpred[g]
        s = 0
        if jp >= 0:
            e = g_end[jp]
            if e > s:
                s = e
        if mp >= 0:
            e = g_end[mp]
            if e > s:
                s = e
        g_end[g] = s + g_dur[g]

        js = g_jsucc[g]
        if js >= 0:
            in_deg[js] -= 1
            if in_deg[js] == 0:
                queue[tail] = js
                tail += 1
        ms = g_msucc[g]
        if ms >= 0:
            in_deg[ms] -= 1
            if in_deg[ms] == 0:
                queue[tail] = ms
                tail += 1

    makespan = 0
    for g in range(total_ops):
        if g_end[g] > makespan:
            makespan = g_end[g]
    return makespan


def _find_critical_blocks(
    durations: list[list[int]],
    machines: list[list[int]],
    num_jobs: int,
    num_machines: int,
    machine_orders: list[list[tuple[int, int]]],
    op_start: list[list[int]],
    op_end: list[list[int]],
    makespan: int,
) -> list[tuple[int, list[int]]]:
    """Find critical blocks (consecutive critical ops on same machine).
    Returns list of (machine_id, [positions in machine_orders]).
    """
    # Build reverse lookup: (job_id, op_idx) -> (machine_id, position)
    op_to_mpos: dict[tuple[int, int], tuple[int, int]] = {}
    for m in range(num_machines):
        for pos, (j, o) in enumerate(machine_orders[m]):
            op_to_mpos[(j, o)] = (m, pos)

    # Trace critical path backward from makespan
    # Find operation ending at makespan
    current = None
    for j in range(num_jobs):
        for o in range(len(durations[j])):
            if op_end[j][o] == makespan:
                current = (j, o)
                break
        if current:
            break

    if not current:
        return []

    # Collect critical path as list of (job, op, machine, pos)
    path = []
    while current is not None:
        j, o = current
        m, pos = op_to_mpos[(j, o)]
        path.append((j, o, m, pos))

        start = op_start[j][o]
        if start == 0:
            break

        next_current = None
        # Job predecessor
        if o > 0 and op_end[j][o - 1] == start:
            next_current = (j, o - 1)
        # Machine predecessor
        if pos > 0:
            pj, po = machine_orders[m][pos - 1]
            if op_end[pj][po] == start:
                next_current = (pj, po)
        current = next_current

    path.reverse()

    if len(path) < 2:
        return []

    # Group into blocks by machine
    blocks: list[tuple[int, list[int]]] = []
    cur_m = path[0][2]
    cur_positions = [path[0][3]]

    for i in range(1, len(path)):
        j, o, m, pos = path[i]
        if m == cur_m:
            cur_positions.append(pos)
        else:
            if len(cur_positions) >= 2:
                blocks.append((cur_m, cur_positions))
            cur_m = m
            cur_positions = [pos]

    if len(cur_positions) >= 2:
        blocks.append((cur_m, cur_positions))

    return blocks


def _generate_n5_moves(
    blocks: list[tuple[int, list[int]]],
) -> list[tuple[int, int, int]]:
    """Generate N5 neighborhood moves from critical blocks.
    Each move is (machine_id, pos_from, pos_to) meaning move the operation
    at pos_from to pos_to (insert before/after).
    For N5: swap first two or last two of each block.
    Returns list of (machine, pos1, pos2) for adjacent swaps.
    """
    moves = []
    for m, positions in blocks:
        if len(positions) >= 2:
            # Swap first two elements of block
            moves.append((m, positions[0], positions[1]))
            # Swap last two elements of block
            if len(positions) >= 2:
                moves.append((m, positions[-2], positions[-1]))
    return moves


def _tabu_search(
    durations: list[list[int]],
    machines_mat: list[list[int]],
    num_jobs: int,
    num_machines: int,
    num_ops: list[int],
    op_gid: list[list[int]],
    total_ops: int,
    machine_orders: list[list[tuple[int, int]]],
    makespan: int,
    time_limit: float = 2.0,
    seed: int = 42,
) -> tuple[int, list[list[tuple[int, int]]]]:
    """Tabu search with N5 critical block neighborhood."""
    best_ms = makespan
    best_orders = [lst[:] for lst in machine_orders]
    current_ms = makespan
    current_orders = [lst[:] for lst in machine_orders]

    start_time = time.perf_counter()
    rng = _random.Random(seed)

    # Tabu list: store (machine, job1, job2) of swapped pair
    tabu_list: dict[tuple[int, int, int], int] = {}
    tabu_tenure = max(7, num_jobs + num_machines // 2)

    iteration = 0

    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed > time_limit:
            break

        iteration += 1

        # Compute times for current solution
        ms_val, op_start, op_end = _compute_times(
            durations, machines_mat, num_jobs, num_machines, current_orders
        )
        current_ms = ms_val

        # Find critical blocks
        blocks = _find_critical_blocks(
            durations, machines_mat, num_jobs, num_machines,
            current_orders, op_start, op_end, current_ms
        )

        if not blocks:
            break

        # Generate N5 moves
        moves = _generate_n5_moves(blocks)

        if not moves:
            # Fallback: random swap on a critical machine
            critical_machines = [m for m, _ in blocks]
            if not critical_machines:
                break
            m = rng.choice(critical_machines)
            if len(current_orders[m]) < 2:
                continue
            pos = rng.randint(0, len(current_orders[m]) - 2)
            moves = [(m, pos, pos + 1)]

        # Evaluate all moves, pick best non-tabu (or aspiration)
        best_move_ms = None
        best_move_info = None

        for m, p1, p2 in moves:
            j1 = current_orders[m][p1][0]
            j2 = current_orders[m][p2][0]

            # Create new orders with swap
            new_orders = [lst[:] for lst in current_orders]
            new_orders[m][p1], new_orders[m][p2] = new_orders[m][p2], new_orders[m][p1]

            new_ms = _compute_makespan_only(
                durations, num_jobs, num_machines, num_ops, op_gid,
                total_ops, new_orders
            )

            # Check tabu status
            tabu_key = (m, min(j1, j2), max(j1, j2))
            is_tabu = tabu_key in tabu_list and tabu_list[tabu_key] > iteration

            # Aspiration: accept if better than best known
            if is_tabu and new_ms >= best_ms:
                continue

            if best_move_ms is None or new_ms < best_move_ms:
                best_move_ms = new_ms
                best_move_info = (m, p1, p2, j1, j2, new_orders)

        if best_move_info is None:
            # All moves are tabu, pick random
            if moves:
                m, p1, p2 = rng.choice(moves)
                new_orders = [lst[:] for lst in current_orders]
                new_orders[m][p1], new_orders[m][p2] = new_orders[m][p2], new_orders[m][p1]
                new_ms = _compute_makespan_only(
                    durations, num_jobs, num_machines, num_ops, op_gid,
                    total_ops, new_orders
                )
                j1 = current_orders[m][p1][0]
                j2 = current_orders[m][p2][0]
                best_move_info = (m, p1, p2, j1, j2, new_orders)
                best_move_ms = new_ms
            else:
                break

        m, p1, p2, j1, j2, new_orders = best_move_info

        # Add to tabu list
        tabu_key = (m, min(j1, j2), max(j1, j2))
        tabu_list[tabu_key] = iteration + tabu_tenure + rng.randint(0, 5)

        current_orders = new_orders
        current_ms = best_move_ms

        if current_ms < best_ms:
            best_ms = current_ms
            best_orders = [lst[:] for lst in current_orders]

        # Clean old tabu entries periodically
        if iteration % 100 == 0:
            tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}

    return best_ms, best_orders


def _orders_to_schedules(
    durations: list[list[int]],
    machines: list[list[int]],
    num_jobs: int,
    num_machines: int,
    machine_orders: list[list[tuple[int, int]]],
) -> tuple[int, list[list[dict[str, int]]]]:
    """Convert machine_orders to machine_schedules format."""
    ms, op_start, op_end = _compute_times(
        durations, machines, num_jobs, num_machines, machine_orders
    )

    machine_schedules: list[list[dict[str, int]]] = [[] for _ in range(num_machines)]
    for m in range(num_machines):
        for job_id, op_idx in machine_orders[m]:
            machine_schedules[m].append({
                "job_id": job_id,
                "operation_index": op_idx,
                "start_time": op_start[job_id][op_idx],
                "end_time": op_end[job_id][op_idx],
                "duration": durations[job_id][op_idx],
            })

    return ms, machine_schedules


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Advanced scheduler with multiple dispatching rules + tabu search.

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
    machines_mat: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    total_operations = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines_mat) + 1
    num_ops = [len(durations[j]) for j in range(num_jobs)]

    # Build op_gid mapping
    op_gid: list[list[int]] = []
    gid = 0
    for j in range(num_jobs):
        ids = list(range(gid, gid + num_ops[j]))
        op_gid.append(ids)
        gid += num_ops[j]

    # Precompute remaining work for MWR/LWR rules
    remaining_work: list[list[int]] = []
    for j in range(num_jobs):
        rw = [0] * len(durations[j])
        rw[-1] = durations[j][-1]
        for o in range(len(durations[j]) - 2, -1, -1):
            rw[o] = rw[o + 1] + durations[j][o]
        remaining_work.append(rw)

    # Determine time budget based on instance size
    size = num_jobs * num_machines
    if size <= 225:  # 15x15
        time_budget = 2.0
    elif size <= 400:  # 20x20
        time_budget = 4.0
    elif size <= 600:  # 30x20
        time_budget = 5.0
    else:  # 50x20, 100x20
        time_budget = 7.0

    start_time = time.perf_counter()

    # Phase 1: Generate initial solutions with multiple dispatching rules
    rules = ["spt", "lpt", "mwr", "lwr", "mopnr", "lopnr"]

    all_solutions: list[tuple[int, list[list[tuple[int, int]]]]] = []

    for rule in rules:
        ms, orders = _greedy_schedule(
            durations, machines_mat, num_jobs, num_machines,
            total_operations, rule, remaining_work
        )
        all_solutions.append((ms, orders))

    # Add random restarts
    rng = _random.Random(123)
    for _ in range(30):
        _random.seed(rng.randint(0, 10**9))
        ms, orders = _greedy_schedule(
            durations, machines_mat, num_jobs, num_machines,
            total_operations, "est_random", remaining_work
        )
        all_solutions.append((ms, orders))

    # Sort by makespan
    all_solutions.sort(key=lambda x: x[0])

    best_ms = all_solutions[0][0]
    best_orders = all_solutions[0][1]

    elapsed_phase1 = time.perf_counter() - start_time
    remaining_time = max(0.1, time_budget - elapsed_phase1)

    # Phase 2: Tabu search from multiple starting points
    # Use top solutions as starting points
    num_restarts = min(5, len(all_solutions))
    # Deduplicate by makespan (rough)
    seen_ms = set()
    unique_starts = []
    for ms, orders in all_solutions:
        if ms not in seen_ms or len(unique_starts) < 3:
            seen_ms.add(ms)
            unique_starts.append((ms, orders))
        if len(unique_starts) >= num_restarts:
            break

    time_per_restart = remaining_time / max(1, len(unique_starts))

    for idx, (ms, orders) in enumerate(unique_starts):
        if time.perf_counter() - start_time > time_budget - 0.05:
            break
        remaining_now = time_budget - (time.perf_counter() - start_time)
        t_limit = min(time_per_restart, remaining_now - 0.02)
        if t_limit < 0.05:
            break

        new_ms, new_orders = _tabu_search(
            durations, machines_mat, num_jobs, num_machines,
            num_ops, op_gid, total_operations,
            orders, ms, time_limit=t_limit, seed=42 + idx
        )
        if new_ms < best_ms:
            best_ms = new_ms
            best_orders = new_orders

    # Convert to output format
    makespan, machine_schedules = _orders_to_schedules(
        durations, machines_mat, num_jobs, num_machines, best_orders
    )

    return {
        "name": instance["name"],
        "makespan": makespan,
        "machine_schedules": machine_schedules,
        "solved_by": "TabuSearchWithCriticalBlocks",
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