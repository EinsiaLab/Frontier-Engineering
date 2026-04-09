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
from collections import deque


def _greedy_schedule(
    durations: list[list[int]],
    machines_matrix: list[list[int]],
    num_jobs: int,
    num_machines: int,
    num_ops_per_job: list[int],
    rule: str = "spt",
) -> list[list[tuple[int, int]]]:
    """Build a schedule using a dispatching rule. Returns machine_orders."""
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    machine_orders: list[list[tuple[int, int]]] = [[] for _ in range(num_machines)]
    remaining_work = [sum(d) for d in durations]
    remaining_ops = [len(d) for d in durations]
    total_ops = sum(num_ops_per_job)

    scheduled = 0
    while scheduled < total_ops:
        best = None
        best_key = None
        for j in range(num_jobs):
            o = next_op[j]
            if o >= num_ops_per_job[j]:
                continue
            m = machines_matrix[j][o]
            d = durations[j][o]
            est = max(job_ready[j], machine_ready[m])

            if rule == "spt":
                key = (est, d, j)
            elif rule == "lpt":
                key = (est, -d, j)
            elif rule == "mwr":
                key = (est, -remaining_work[j], j)
            elif rule == "lwr":
                key = (est, remaining_work[j], j)
            elif rule == "mopnr":
                key = (est, -remaining_ops[j], j)
            elif rule == "lopnr":
                key = (est, remaining_ops[j], j)
            elif rule == "est_lpt":
                key = (est, -d, -remaining_work[j], j)
            elif rule == "est_mwr":
                key = (est, -remaining_work[j], d, j)
            elif rule == "fifo":
                key = (est, j, o)
            elif rule == "est_mopnr":
                key = (est, -remaining_ops[j], -remaining_work[j], j)
            else:
                key = (est, d, j)

            if best_key is None or key < best_key:
                best_key = key
                best = (j, o, m, d, est)

        j, o, m, d, est = best
        end = est + d
        machine_orders[m].append((j, o))
        next_op[j] += 1
        job_ready[j] = end
        machine_ready[m] = end
        remaining_work[j] -= d
        remaining_ops[j] -= 1
        scheduled += 1

    return machine_orders


def _evaluate(
    durations: list[list[int]],
    machines_matrix: list[list[int]],
    num_jobs: int,
    num_machines: int,
    num_ops_per_job: list[int],
    machine_orders: list[list[tuple[int, int]]],
) -> tuple[int, list[list[int]], list[list[int]]]:
    """Evaluate makespan from machine_orders using topological BFS."""
    start_times = [[0] * num_ops_per_job[j] for j in range(num_jobs)]
    end_times = [[0] * num_ops_per_job[j] for j in range(num_jobs)]
    in_degree = [[0] * num_ops_per_job[j] for j in range(num_jobs)]

    # Machine predecessor/successor
    mp = [[None] * num_ops_per_job[j] for j in range(num_jobs)]
    ms = [[None] * num_ops_per_job[j] for j in range(num_jobs)]

    for m in range(num_machines):
        order = machine_orders[m]
        for i in range(len(order)):
            if i > 0:
                pj, po = order[i - 1]
                j, o = order[i]
                mp[j][o] = (pj, po)
                in_degree[j][o] += 1
            if i < len(order) - 1:
                nj, no = order[i + 1]
                j, o = order[i]
                ms[j][o] = (nj, no)

    for j in range(num_jobs):
        for o in range(1, num_ops_per_job[j]):
            in_degree[j][o] += 1

    queue = deque()
    for j in range(num_jobs):
        for o in range(num_ops_per_job[j]):
            if in_degree[j][o] == 0:
                queue.append((j, o))

    while queue:
        j, o = queue.popleft()
        s = 0
        if o > 0:
            s = end_times[j][o - 1]
        pred = mp[j][o]
        if pred is not None:
            e = end_times[pred[0]][pred[1]]
            if e > s:
                s = e
        start_times[j][o] = s
        end_times[j][o] = s + durations[j][o]

        if o + 1 < num_ops_per_job[j]:
            in_degree[j][o + 1] -= 1
            if in_degree[j][o + 1] == 0:
                queue.append((j, o + 1))
        succ = ms[j][o]
        if succ is not None:
            in_degree[succ[0]][succ[1]] -= 1
            if in_degree[succ[0]][succ[1]] == 0:
                queue.append((succ[0], succ[1]))

    makespan = 0
    for j in range(num_jobs):
        if num_ops_per_job[j] > 0:
            e = end_times[j][num_ops_per_job[j] - 1]
            if e > makespan:
                makespan = e

    return makespan, start_times, end_times


def _critical_path_and_blocks(
    durations: list[list[int]],
    machines_matrix: list[list[int]],
    num_jobs: int,
    num_machines: int,
    machine_orders: list[list[tuple[int, int]]],
    start_times: list[list[int]],
    end_times: list[list[int]],
    makespan: int,
    num_ops_per_job: list[int],
) -> list[tuple[int, list[tuple[int, int]]]]:
    """Find critical path and return critical blocks as (machine_id, [(j,o)...])."""
    # Build machine predecessor lookup
    machine_pred: dict[tuple[int, int], tuple[int, int]] = {}
    op_machine: dict[tuple[int, int], int] = {}
    for m in range(num_machines):
        order = machine_orders[m]
        for i, (j, o) in enumerate(order):
            op_machine[(j, o)] = m
            if i > 0:
                machine_pred[(j, o)] = order[i - 1]

    # Find operation ending at makespan
    current = None
    for j in range(num_jobs):
        if num_ops_per_job[j] > 0 and end_times[j][num_ops_per_job[j] - 1] == makespan:
            current = (j, num_ops_per_job[j] - 1)
            break
    if current is None:
        return []

    path = [current]
    while True:
        j, o = current
        s = start_times[j][o]
        if s == 0:
            break
        found = False
        if o > 0 and end_times[j][o - 1] == s:
            current = (j, o - 1)
            path.append(current)
            found = True
        if not found:
            pred = machine_pred.get((j, o))
            if pred is not None and end_times[pred[0]][pred[1]] == s:
                current = pred
                path.append(current)
                found = True
        if not found:
            break

    path.reverse()

    # Split into blocks
    if len(path) < 2:
        return []

    blocks = []
    cur_block = [path[0]]
    cur_m = op_machine[path[0]]
    for i in range(1, len(path)):
        m = op_machine[path[i]]
        if m == cur_m:
            cur_block.append(path[i])
        else:
            if len(cur_block) >= 2:
                blocks.append((cur_m, cur_block))
            cur_block = [path[i]]
            cur_m = m
    if len(cur_block) >= 2:
        blocks.append((cur_m, cur_block))

    return blocks


def _build_op_pos(machine_orders, num_machines):
    """Build (j,o) -> (machine, position) lookup."""
    op_pos = {}
    for m in range(num_machines):
        for pos, (j, o) in enumerate(machine_orders[m]):
            op_pos[(j, o)] = (m, pos)
    return op_pos


def _try_swap_evaluate(
    machine_orders: list[list[tuple[int, int]]],
    m: int,
    p1: int,
    p2: int,
    durations: list[list[int]],
    machines_matrix: list[list[int]],
    num_jobs: int,
    num_machines: int,
    num_ops_per_job: list[int],
) -> tuple[int, list[list[tuple[int, int]]]]:
    """Swap two positions on a machine and evaluate."""
    new_orders = [order[:] for order in machine_orders]
    new_orders[m][p1], new_orders[m][p2] = new_orders[m][p2], new_orders[m][p1]
    ms, _, _ = _evaluate(durations, machines_matrix, num_jobs, num_machines, num_ops_per_job, new_orders)
    return ms, new_orders


def _tabu_search(
    durations: list[list[int]],
    machines_matrix: list[list[int]],
    num_jobs: int,
    num_machines: int,
    num_ops_per_job: list[int],
    machine_orders: list[list[tuple[int, int]]],
    initial_makespan: int,
    time_limit: float,
    start_time: float,
) -> tuple[int, list[list[tuple[int, int]]]]:
    """Tabu search using critical block neighborhood."""
    best_ms = initial_makespan
    best_orders = [o[:] for o in machine_orders]
    current_ms = initial_makespan
    current_orders = [o[:] for o in machine_orders]

    tabu_list: dict[tuple, int] = {}
    tabu_tenure = max(7, num_jobs + num_machines // 2)
    iteration = 0
    no_improve_count = 0
    max_no_improve = max(200, num_jobs * num_machines)

    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed > time_limit:
            break

        iteration += 1
        no_improve_count += 1

        if no_improve_count > max_no_improve:
            break

        ms, st, et = _evaluate(durations, machines_matrix, num_jobs, num_machines, num_ops_per_job, current_orders)
        current_ms = ms

        blocks = _critical_path_and_blocks(
            durations, machines_matrix, num_jobs, num_machines,
            current_orders, st, et, ms, num_ops_per_job
        )

        if not blocks:
            break

        op_pos = _build_op_pos(current_orders, num_machines)

        # Generate neighborhood moves from critical blocks
        moves = []
        for machine_id, block_ops in blocks:
            if len(block_ops) < 2:
                continue
            # Swap first two operations in block
            j1, o1 = block_ops[0]
            j2, o2 = block_ops[1]
            _, p1 = op_pos[(j1, o1)]
            _, p2 = op_pos[(j2, o2)]
            moves.append((machine_id, p1, p2, j1, j2))

            # Swap last two operations in block
            if len(block_ops) >= 2:
                j1, o1 = block_ops[-2]
                j2, o2 = block_ops[-1]
                _, p1 = op_pos[(j1, o1)]
                _, p2 = op_pos[(j2, o2)]
                moves.append((machine_id, p1, p2, j1, j2))

        if not moves:
            break

        best_move_ms = float('inf')
        best_move = None
        best_move_orders = None

        for machine_id, p1, p2, j1, j2 in moves:
            if time.perf_counter() - start_time > time_limit:
                break

            new_ms, new_orders = _try_swap_evaluate(
                current_orders, machine_id, p1, p2,
                durations, machines_matrix, num_jobs, num_machines, num_ops_per_job
            )

            tabu_key = (machine_id, min(j1, j2), max(j1, j2))
            is_tabu = tabu_key in tabu_list and tabu_list[tabu_key] > iteration

            # Aspiration: accept if better than global best
            if new_ms < best_ms:
                is_tabu = False

            if not is_tabu and new_ms < best_move_ms:
                best_move_ms = new_ms
                best_move = tabu_key
                best_move_orders = new_orders

        # If all moves are tabu, pick the best anyway
        if best_move_orders is None:
            for machine_id, p1, p2, j1, j2 in moves:
                new_ms, new_orders = _try_swap_evaluate(
                    current_orders, machine_id, p1, p2,
                    durations, machines_matrix, num_jobs, num_machines, num_ops_per_job
                )
                if new_ms < best_move_ms:
                    best_move_ms = new_ms
                    best_move = (machine_id, min(j1, j2), max(j1, j2))
                    best_move_orders = new_orders

        if best_move_orders is None:
            break

        current_orders = best_move_orders
        current_ms = best_move_ms

        if best_move is not None:
            tabu_list[best_move] = iteration + tabu_tenure

        if current_ms < best_ms:
            best_ms = current_ms
            best_orders = [o[:] for o in current_orders]
            no_improve_count = 0

        # Clean old tabu entries periodically
        if iteration % 100 == 0:
            tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}

    return best_ms, best_orders


def _extended_neighborhood_search(
    durations: list[list[int]],
    machines_matrix: list[list[int]],
    num_jobs: int,
    num_machines: int,
    num_ops_per_job: list[int],
    machine_orders: list[list[tuple[int, int]]],
    initial_makespan: int,
    time_limit: float,
    start_time: float,
) -> tuple[int, list[list[tuple[int, int]]]]:
    """Extended neighborhood: try all adjacent swaps in critical blocks."""
    best_ms = initial_makespan
    best_orders = [o[:] for o in machine_orders]

    improved = True
    while improved:
        if time.perf_counter() - start_time > time_limit:
            break
        improved = False

        ms, st, et = _evaluate(durations, machines_matrix, num_jobs, num_machines, num_ops_per_job, best_orders)
        best_ms = ms

        blocks = _critical_path_and_blocks(
            durations, machines_matrix, num_jobs, num_machines,
            best_orders, st, et, ms, num_ops_per_job
        )

        if not blocks:
            break

        op_pos = _build_op_pos(best_orders, num_machines)

        found_improvement = False
        for machine_id, block_ops in blocks:
            if found_improvement:
                break
            if len(block_ops) < 2:
                continue

            # Try swapping first two
            j1, o1 = block_ops[0]
            j2, o2 = block_ops[1]
            _, p1 = op_pos[(j1, o1)]
            _, p2 = op_pos[(j2, o2)]
            new_ms, new_orders = _try_swap_evaluate(
                best_orders, machine_id, p1, p2,
                durations, machines_matrix, num_jobs, num_machines, num_ops_per_job
            )
            if new_ms < best_ms:
                best_ms = new_ms
                best_orders = new_orders
                improved = True
                found_improvement = True
                break

            # Try swapping last two
            j1, o1 = block_ops[-2]
            j2, o2 = block_ops[-1]
            _, p1 = op_pos[(j1, o1)]
            _, p2 = op_pos[(j2, o2)]
            new_ms, new_orders = _try_swap_evaluate(
                best_orders, machine_id, p1, p2,
                durations, machines_matrix, num_jobs, num_machines, num_ops_per_job
            )
            if new_ms < best_ms:
                best_ms = new_ms
                best_orders = new_orders
                improved = True
                found_improvement = True
                break

            # Try all adjacent swaps in block
            for bi in range(len(block_ops) - 1):
                if time.perf_counter() - start_time > time_limit:
                    break
                j1, o1 = block_ops[bi]
                j2, o2 = block_ops[bi + 1]
                _, p1 = op_pos[(j1, o1)]
                _, p2 = op_pos[(j2, o2)]
                new_ms, new_orders = _try_swap_evaluate(
                    best_orders, machine_id, p1, p2,
                    durations, machines_matrix, num_jobs, num_machines, num_ops_per_job
                )
                if new_ms < best_ms:
                    best_ms = new_ms
                    best_orders = new_orders
                    improved = True
                    found_improvement = True
                    break

    return best_ms, best_orders


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Advanced JSSP solver with multi-start greedy + tabu search.

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
    num_ops_per_job = [len(durations[j]) for j in range(num_jobs)]
    total_operations = sum(num_ops_per_job)
    num_machines = max(max(row) for row in machines) + 1

    solve_start = time.perf_counter()
    # Time budget: generous for larger instances
    time_limit = min(180.0, max(8.0, num_jobs * num_machines * 0.05))

    # Dispatching rules to try
    rules = ["spt", "lpt", "mwr", "lwr", "mopnr", "lopnr", "est_lpt", "est_mwr", "fifo", "est_mopnr"]

    # Phase 1: Generate initial solutions with different rules
    solutions = []
    for rule in rules:
        if time.perf_counter() - solve_start > time_limit * 0.15:
            break
        try:
            orders = _greedy_schedule(durations, machines, num_jobs, num_machines, num_ops_per_job, rule)
            ms, _, _ = _evaluate(durations, machines, num_jobs, num_machines, num_ops_per_job, orders)
            solutions.append((ms, orders, rule))
        except Exception:
            continue

    # Sort by makespan
    solutions.sort(key=lambda x: x[0])

    if not solutions:
        raise RuntimeError("Failed to produce any schedule.")

    best_makespan = solutions[0][0]
    best_orders = solutions[0][1]

    # Phase 2: Local search (hill climbing) on best solution
    remaining = time_limit - (time.perf_counter() - solve_start)
    if remaining > 0.5:
        ls_limit = min(remaining * 0.3, time_limit * 0.2)
        new_ms, new_orders = _extended_neighborhood_search(
            durations, machines, num_jobs, num_machines, num_ops_per_job,
            best_orders, best_makespan,
            solve_start + (time.perf_counter() - solve_start) + ls_limit,
            solve_start
        )
        if new_ms < best_makespan:
            best_makespan = new_ms
            best_orders = new_orders

    # Phase 3: Tabu search on best solutions
    num_starts = min(len(solutions), 5)
    time_per_start = (time_limit - (time.perf_counter() - solve_start)) / max(num_starts, 1)

    for i in range(num_starts):
        elapsed = time.perf_counter() - solve_start
        if elapsed > time_limit - 0.5:
            break

        ms_i, orders_i, rule_i = solutions[i]

        # First do local search
        ls_end = time.perf_counter() + min(time_per_start * 0.2, 2.0)
        new_ms, new_orders = _extended_neighborhood_search(
            durations, machines, num_jobs, num_machines, num_ops_per_job,
            orders_i, ms_i, ls_end, solve_start
        )

        # Then tabu search
        tabu_end = solve_start + min(elapsed + time_per_start, time_limit - 0.1)
        ts_ms, ts_orders = _tabu_search(
            durations, machines, num_jobs, num_machines, num_ops_per_job,
            new_orders, new_ms, tabu_end, solve_start
        )

        if ts_ms < best_makespan:
            best_makespan = ts_ms
            best_orders = [o[:] for o in ts_orders]

    # Final local search pass
    remaining = time_limit - (time.perf_counter() - solve_start)
    if remaining > 0.5:
        final_ms, final_orders = _extended_neighborhood_search(
            durations, machines, num_jobs, num_machines, num_ops_per_job,
            best_orders, best_makespan,
            solve_start + time_limit - 0.1, solve_start
        )
        if final_ms < best_makespan:
            best_makespan = final_ms
            best_orders = final_orders

    # Convert to output format
    final_ms, start_times, end_times = _evaluate(
        durations, machines, num_jobs, num_machines, num_ops_per_job, best_orders
    )

    machine_schedules: list[list[dict[str, int]]] = [[] for _ in range(num_machines)]
    for m in range(num_machines):
        for job_id, op_idx in best_orders[m]:
            machine_schedules[m].append({
                "job_id": job_id,
                "operation_index": op_idx,
                "start_time": start_times[job_id][op_idx],
                "end_time": end_times[job_id][op_idx],
                "duration": durations[job_id][op_idx],
            })

    return {
        "name": instance["name"],
        "makespan": final_ms,
        "machine_schedules": machine_schedules,
        "solved_by": "MultiStartTabuSearch",
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