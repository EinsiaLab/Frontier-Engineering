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


import random as _random


from collections import deque as _deque
from heapq import heappush as _heappush, heappop as _heappop


def _greedy_schedule(
    durations: list[list[int]],
    machines: list[list[int]],
    num_jobs: int,
    num_machines: int,
    rule: str = "est_spt",
    rng: _random.Random | None = None,
) -> list[list[tuple[int, int]]]:
    """Build a schedule using a dispatching rule."""
    num_ops = [len(durations[j]) for j in range(num_jobs)]
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    remaining_work = [sum(durations[j]) for j in range(num_jobs)]
    machine_orders: list[list[tuple[int, int]]] = [[] for _ in range(num_machines)]
    total = sum(num_ops)
    scheduled = 0

    while scheduled < total:
        best = None
        best_key: tuple = ()
        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= num_ops[job_id]:
                continue
            machine_id = machines[job_id][op_idx]
            dur = durations[job_id][op_idx]
            est = job_ready[job_id]
            mr = machine_ready[machine_id]
            if mr > est:
                est = mr

            if rule == "est_spt":
                key = (est, dur, job_id)
            elif rule == "est_lpt":
                key = (est, -dur, job_id)
            elif rule == "mwr":
                key = (-remaining_work[job_id], est, job_id)
            elif rule == "lwr":
                key = (remaining_work[job_id], est, job_id)
            elif rule == "est_random":
                key = (est, rng.random() if rng else 0.0, job_id)
            elif rule == "spt":
                key = (dur, est, job_id)
            elif rule == "lpt":
                key = (-dur, est, job_id)
            elif rule == "est_mwr":
                key = (est, -remaining_work[job_id], job_id)
            else:
                key = (est, dur, job_id)

            if best is None or key < best_key:
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
        remaining_work[job_id] -= dur
        scheduled += 1

    return machine_orders


# ---------------------------------------------------------------------------
# Flat-array based fast evaluation
# ---------------------------------------------------------------------------

def _build_flat(durations, num_jobs, num_ops):
    """Build flat duration array and job_offset."""
    total = sum(num_ops)
    job_offset = [0] * (num_jobs + 1)
    for j in range(num_jobs):
        job_offset[j + 1] = job_offset[j] + num_ops[j]
    dur_flat = [0] * total
    for j in range(num_jobs):
        off = job_offset[j]
        dj = durations[j]
        for o in range(num_ops[j]):
            dur_flat[off + o] = dj[o]
    return dur_flat, job_offset, total


def _evaluate_flat(
    dur_flat: list[int],
    num_jobs: int,
    num_machines: int,
    machine_orders: list[list[tuple[int, int]]],
    num_ops: list[int],
    job_offset: list[int],
    total_ops: int,
) -> tuple[int, list[int]]:
    """BFS evaluation using flat arrays. Returns (makespan, flat_start_times)."""
    start = [0] * total_ops
    job_succ = [-1] * total_ops
    machine_succ = [-1] * total_ops
    in_deg = [0] * total_ops

    for j in range(num_jobs):
        off = job_offset[j]
        nops = num_ops[j]
        for o in range(nops - 1):
            job_succ[off + o] = off + o + 1
            in_deg[off + o + 1] += 1

    for m in range(num_machines):
        order = machine_orders[m]
        prev_flat = -1
        for j_op, o_op in order:
            cur_flat = job_offset[j_op] + o_op
            if prev_flat >= 0:
                machine_succ[prev_flat] = cur_flat
                in_deg[cur_flat] += 1
            prev_flat = cur_flat

    queue = _deque()
    for idx in range(total_ops):
        if in_deg[idx] == 0:
            queue.append(idx)

    makespan = 0
    while queue:
        idx = queue.popleft()
        end_time = start[idx] + dur_flat[idx]
        if end_time > makespan:
            makespan = end_time

        js = job_succ[idx]
        if js >= 0:
            if end_time > start[js]:
                start[js] = end_time
            in_deg[js] -= 1
            if in_deg[js] == 0:
                queue.append(js)

        ms = machine_succ[idx]
        if ms >= 0:
            if end_time > start[ms]:
                start[ms] = end_time
            in_deg[ms] -= 1
            if in_deg[ms] == 0:
                queue.append(ms)

    return makespan, start


class _EvalBuffers:
    """Pre-allocated buffers for repeated evaluation calls."""
    __slots__ = ('start', 'machine_succ', 'in_deg', 'queue', 'job_succ',
                 'total_ops', 'dur_flat', 'job_offset', 'num_ops',
                 'num_jobs', 'num_machines', 'job_in_deg')

    def __init__(self, dur_flat, num_jobs, num_machines, num_ops, job_offset, total_ops):
        self.total_ops = total_ops
        self.dur_flat = dur_flat
        self.job_offset = job_offset
        self.num_ops = num_ops
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.start = [0] * total_ops
        self.machine_succ = [-1] * total_ops
        self.in_deg = [0] * total_ops
        self.queue = _deque()
        self.job_succ = [-1] * total_ops
        self.job_in_deg = [0] * total_ops
        for j in range(num_jobs):
            off = job_offset[j]
            nops = num_ops[j]
            for o in range(nops - 1):
                self.job_succ[off + o] = off + o + 1
                self.job_in_deg[off + o + 1] = 1

    def evaluate(self, machine_orders):
        """Evaluate and return (makespan, start_times_copy)."""
        total_ops = self.total_ops
        dur_flat = self.dur_flat
        job_succ = self.job_succ
        job_offset = self.job_offset
        num_machines = self.num_machines
        start = self.start
        machine_succ = self.machine_succ
        in_deg = self.in_deg
        job_in_deg = self.job_in_deg
        queue = self.queue

        for i in range(total_ops):
            start[i] = 0
            machine_succ[i] = -1
            in_deg[i] = job_in_deg[i]

        for m in range(num_machines):
            order = machine_orders[m]
            prev_flat = -1
            for j_op, o_op in order:
                cur_flat = job_offset[j_op] + o_op
                if prev_flat >= 0:
                    machine_succ[prev_flat] = cur_flat
                    in_deg[cur_flat] += 1
                prev_flat = cur_flat

        queue.clear()
        for idx in range(total_ops):
            if in_deg[idx] == 0:
                queue.append(idx)

        makespan = 0
        popleft = queue.popleft
        append = queue.append
        while queue:
            idx = popleft()
            end_time = start[idx] + dur_flat[idx]
            if end_time > makespan:
                makespan = end_time

            js = job_succ[idx]
            if js >= 0:
                if end_time > start[js]:
                    start[js] = end_time
                in_deg[js] -= 1
                if in_deg[js] == 0:
                    append(js)

            ms = machine_succ[idx]
            if ms >= 0:
                if end_time > start[ms]:
                    start[ms] = end_time
                in_deg[ms] -= 1
                if in_deg[ms] == 0:
                    append(ms)

        return makespan, list(start)

    def makespan_only(self, machine_orders):
        """Evaluate makespan only (no copy of start)."""
        total_ops = self.total_ops
        dur_flat = self.dur_flat
        job_succ = self.job_succ
        job_offset = self.job_offset
        num_machines = self.num_machines
        start = self.start
        machine_succ = self.machine_succ
        in_deg = self.in_deg
        job_in_deg = self.job_in_deg
        queue = self.queue

        for i in range(total_ops):
            start[i] = 0
            machine_succ[i] = -1
            in_deg[i] = job_in_deg[i]

        for m in range(num_machines):
            order = machine_orders[m]
            prev_flat = -1
            for j_op, o_op in order:
                cur_flat = job_offset[j_op] + o_op
                if prev_flat >= 0:
                    machine_succ[prev_flat] = cur_flat
                    in_deg[cur_flat] += 1
                prev_flat = cur_flat

        queue.clear()
        for idx in range(total_ops):
            if in_deg[idx] == 0:
                queue.append(idx)

        makespan = 0
        popleft = queue.popleft
        append = queue.append
        while queue:
            idx = popleft()
            end_time = start[idx] + dur_flat[idx]
            if end_time > makespan:
                makespan = end_time

            js = job_succ[idx]
            if js >= 0:
                if end_time > start[js]:
                    start[js] = end_time
                in_deg[js] -= 1
                if in_deg[js] == 0:
                    append(js)

            ms = machine_succ[idx]
            if ms >= 0:
                if end_time > start[ms]:
                    start[ms] = end_time
                in_deg[ms] -= 1
                if in_deg[ms] == 0:
                    append(ms)

        return makespan

    def eval_swap(self, machine_orders, swap_m, swap_i, swap_j):
        """Evaluate makespan after swapping, then unswap."""
        order = machine_orders[swap_m]
        order[swap_i], order[swap_j] = order[swap_j], order[swap_i]
        ms = self.makespan_only(machine_orders)
        order[swap_i], order[swap_j] = order[swap_j], order[swap_i]
        return ms


def _evaluate_swap_fast(
    dur_flat: list[int],
    num_jobs: int,
    num_machines: int,
    machine_orders: list[list[tuple[int, int]]],
    num_ops: list[int],
    job_offset: list[int],
    total_ops: int,
    swap_m: int,
    swap_i: int,
    swap_j: int,
) -> int:
    """Evaluate makespan after swapping positions swap_i and swap_j on machine swap_m.
    Swap-evaluate-unswap pattern."""
    order = machine_orders[swap_m]
    order[swap_i], order[swap_j] = order[swap_j], order[swap_i]
    ms, _ = _evaluate_flat(dur_flat, num_jobs, num_machines, machine_orders,
                           num_ops, job_offset, total_ops)
    order[swap_i], order[swap_j] = order[swap_j], order[swap_i]
    return ms


def _find_critical_blocks_flat(
    dur_flat: list[int],
    machines_mat: list[list[int]],
    num_jobs: int,
    num_machines: int,
    machine_orders: list[list[tuple[int, int]]],
    flat_start: list[int],
    num_ops: list[int],
    job_offset: list[int],
    makespan: int,
) -> list[tuple[int, list[int]]]:
    """Find critical path and return critical blocks as (machine_id, list_of_positions_in_machine_order).
    This is more efficient as we work with position indices directly."""
    # Build position lookup: (j,o) -> (m, pos)
    pos_lookup = {}
    for m in range(num_machines):
        for i, (j, o) in enumerate(machine_orders[m]):
            pos_lookup[(j, o)] = (m, i)

    # Find end op on critical path
    current = None
    for j in range(num_jobs):
        off = job_offset[j]
        nops = num_ops[j]
        if nops > 0:
            last = nops - 1
            flat_idx = off + last
            if flat_start[flat_idx] + dur_flat[flat_idx] == makespan:
                current = (j, last)
                break

    if current is None:
        return []

    # Trace back critical path
    path = []
    while current is not None:
        j, o = current
        path.append(current)
        flat_idx = job_offset[j] + o
        s = flat_start[flat_idx]
        pred = None
        # Check job predecessor
        if o > 0:
            prev_flat = flat_idx - 1
            if flat_start[prev_flat] + dur_flat[prev_flat] == s:
                pred = (j, o - 1)
        # Check machine predecessor
        if pred is None:
            m, idx = pos_lookup[(j, o)]
            if idx > 0:
                pj, po = machine_orders[m][idx - 1]
                pf = job_offset[pj] + po
                if flat_start[pf] + dur_flat[pf] == s:
                    pred = (pj, po)
        current = pred

    path.reverse()

    if not path:
        return []

    # Extract blocks: group consecutive ops on same machine
    blocks = []
    j0, o0 = path[0]
    cur_m, cur_pos = pos_lookup[path[0]]
    cur_positions = [cur_pos]

    for i in range(1, len(path)):
        j, o = path[i]
        m, pos = pos_lookup[(j, o)]
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


def _tabu_search(
    dur_flat: list[int],
    machines_mat: list[list[int]],
    num_jobs: int,
    num_machines: int,
    initial_orders: list[list[tuple[int, int]]],
    num_ops: list[int],
    job_offset: list[int],
    total_ops: int,
    max_iter: int = 1000000,
    tabu_tenure: int = 15,
    time_limit: float = 30.0,
    seed: int = 42,
    bufs: _EvalBuffers | None = None,
) -> tuple[int, list[list[tuple[int, int]]]]:
    """Tabu search using N5/N7 neighborhood with restarts and flat-array evaluation."""
    t0 = time.perf_counter()
    rng = _random.Random(seed)

    if bufs is None:
        bufs = _EvalBuffers(dur_flat, num_jobs, num_machines, num_ops, job_offset, total_ops)

    current_orders = [list(mo) for mo in initial_orders]
    current_ms, flat_st = bufs.evaluate(current_orders)
    best_makespan = current_ms
    best_orders = [list(mo) for mo in current_orders]

    tabu_dict: dict[tuple, int] = {}
    no_improve = 0
    max_no_improve = 800

    base_tenure = tabu_tenure
    tenure = base_tenure

    all_move_evals: list[tuple[tuple[int, int, int], int]] = []

    for iteration in range(max_iter):
        if iteration & 127 == 0:
            if time.perf_counter() - t0 > time_limit:
                break

        # Restart on stagnation
        if no_improve >= max_no_improve:
            current_orders = [list(mo) for mo in best_orders]
            current_ms, flat_st = bufs.evaluate(current_orders)

            # Random perturbation
            num_perturb = rng.randint(5, 15)
            for _ in range(num_perturb):
                blocks = _find_critical_blocks_flat(
                    dur_flat, machines_mat, num_jobs, num_machines,
                    current_orders, flat_st, num_ops, job_offset, current_ms,
                )
                if not blocks:
                    break
                m_id, block_pos = rng.choice(blocks)
                blen = len(block_pos)
                if blen >= 2:
                    k = rng.randint(0, blen - 2)
                    order = current_orders[m_id]
                    p1, p2 = block_pos[k], block_pos[k + 1]
                    order[p1], order[p2] = order[p2], order[p1]
                    current_ms, flat_st = bufs.evaluate(current_orders)

            tabu_dict.clear()
            no_improve = 0
            tenure = base_tenure + rng.randint(-3, 5)
            if tenure < 5:
                tenure = 5
            continue

        # Find critical blocks
        blocks = _find_critical_blocks_flat(
            dur_flat, machines_mat, num_jobs, num_machines,
            current_orders, flat_st, num_ops, job_offset, current_ms,
        )

        if not blocks:
            break

        # Generate N5-style moves
        moves = []
        for m_id, block_pos in blocks:
            blen = len(block_pos)
            if blen >= 2:
                moves.append((m_id, block_pos[0], block_pos[1]))
                moves.append((m_id, block_pos[-2], block_pos[-1]))
                for k in range(1, blen - 1):
                    moves.append((m_id, block_pos[k], block_pos[k + 1]))

        # Deduplicate
        seen = set()
        unique_moves = []
        for mv in moves:
            if mv not in seen:
                seen.add(mv)
                unique_moves.append(mv)

        best_move = None
        best_move_ms = None
        all_move_evals.clear()

        for mv in unique_moves:
            m_id, pi, pj = mv
            ms_new = bufs.eval_swap(current_orders, m_id, pi, pj)
            all_move_evals.append((mv, ms_new))

            # Get job/op ids for tabu
            order = current_orders[m_id]
            ji = order[pi][0]
            oi = order[pi][1]
            jj = order[pj][0]
            oj = order[pj][1]

            # Tabu check with operation-level keys
            tabu_key = (m_id, ji, oi, jj, oj)
            tabu_key_rev = (m_id, jj, oj, ji, oi)
            is_tabu = (
                (tabu_key in tabu_dict and tabu_dict[tabu_key] > iteration) or
                (tabu_key_rev in tabu_dict and tabu_dict[tabu_key_rev] > iteration)
            )

            if is_tabu and ms_new >= best_makespan:
                continue

            if best_move_ms is None or ms_new < best_move_ms:
                best_move_ms = ms_new
                best_move = mv

        if best_move is None:
            # All tabu, pick least bad from cached evaluations
            for mv, ms_new in all_move_evals:
                if best_move_ms is None or ms_new < best_move_ms:
                    best_move_ms = ms_new
                    best_move = mv
            if best_move is None:
                break

        # Apply move
        m_id, pi, pj = best_move
        order = current_orders[m_id]
        ji = order[pi][0]
        oi = order[pi][1]
        jj = order[pj][0]
        oj = order[pj][1]
        order[pi], order[pj] = order[pj], order[pi]
        current_ms = best_move_ms
        _, flat_st = bufs.evaluate(current_orders)

        # Update tabu with operation-level key
        tabu_dict[(m_id, ji, oi, jj, oj)] = iteration + tenure

        if current_ms < best_makespan:
            best_makespan = current_ms
            best_orders = [list(mo) for mo in current_orders]
            no_improve = 0
        else:
            no_improve += 1

    return best_makespan, best_orders


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Advanced solver using multiple dispatching rules + tabu search."""
    durations: list[list[int]] = instance["duration_matrix"]
    machines_mat: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    num_machines = max(max(row) for row in machines_mat) + 1
    num_ops = [len(durations[j]) for j in range(num_jobs)]
    total_ops = sum(num_ops)

    # Build flat arrays once
    dur_flat, job_offset, _ = _build_flat(durations, num_jobs, num_ops)

    # Create reusable evaluation buffers
    bufs = _EvalBuffers(dur_flat, num_jobs, num_machines, num_ops, job_offset, total_ops)

    # Try multiple dispatching rules for initial solution
    rules = ["est_spt", "est_lpt", "mwr", "lwr", "spt", "lpt", "est_mwr"]
    rng = _random.Random(42)

    all_initial: list[tuple[int, list[list[tuple[int, int]]]]] = []

    for rule in rules:
        orders = _greedy_schedule(durations, machines_mat, num_jobs, num_machines, rule, rng)
        ms, _ = bufs.evaluate(orders)
        all_initial.append((ms, orders))

    # Random restarts - more for diversity
    num_random = 300 if total_ops > 100 else 150
    for _ in range(num_random):
        orders = _greedy_schedule(durations, machines_mat, num_jobs, num_machines, "est_random", rng)
        ms, _ = bufs.evaluate(orders)
        all_initial.append((ms, orders))

    # Sort by makespan
    all_initial.sort(key=lambda x: x[0])

    # Quick local search descent on top solutions
    def _quick_descent(init_orders):
        cur_orders = [list(mo) for mo in init_orders]
        cur_ms, cur_st = bufs.evaluate(cur_orders)
        for _ in range(500):
            blocks = _find_critical_blocks_flat(
                dur_flat, machines_mat, num_jobs, num_machines,
                cur_orders, cur_st, num_ops, job_offset, cur_ms,
            )
            if not blocks:
                break
            improved = False
            for m_id, block_pos in blocks:
                blen = len(block_pos)
                if blen < 2:
                    continue
                swap_pairs = [(block_pos[0], block_pos[1])]
                if blen >= 2:
                    swap_pairs.append((block_pos[-2], block_pos[-1]))
                for k in range(1, blen - 1):
                    swap_pairs.append((block_pos[k], block_pos[k + 1]))
                for p1, p2 in swap_pairs:
                    ms_new = bufs.eval_swap(cur_orders, m_id, p1, p2)
                    if ms_new < cur_ms:
                        order = cur_orders[m_id]
                        order[p1], order[p2] = order[p2], order[p1]
                        cur_ms = ms_new
                        _, cur_st = bufs.evaluate(cur_orders)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return cur_ms, cur_orders

    # Improve top solutions with quick descent
    improved_initial: list[tuple[int, list[list[tuple[int, int]]]]] = []
    seen_ms: set[int] = set()
    for ms_i, orders_i in all_initial[:25]:
        if ms_i in seen_ms:
            continue
        seen_ms.add(ms_i)
        ms_imp, orders_imp = _quick_descent(orders_i)
        improved_initial.append((ms_imp, orders_imp))

    improved_initial.sort(key=lambda x: x[0])

    # Determine time budget
    solve_start = time.perf_counter()
    if total_ops <= 100:
        total_time = 14.0
    else:
        total_time = 55.0

    best_makespan = improved_initial[0][0]
    best_orders = improved_initial[0][1]

    # Run tabu search from multiple starting points
    num_restarts = min(12, len(improved_initial))
    # First restart gets 40%, second 20%, rest share 40%
    time_first = total_time * 0.40
    time_second = total_time * 0.20 if num_restarts > 1 else 0
    time_rest_total = total_time * 0.40
    time_rest = time_rest_total / max(1, num_restarts - 2) if num_restarts > 2 else 0

    base_tenure = max(8, num_jobs // 2)

    seed_counter = 42
    for i in range(num_restarts):
        if time.perf_counter() - solve_start > total_time - 0.5:
            break
        init_ms, init_orders = improved_initial[i]
        if i == 0:
            t_limit = time_first
        elif i == 1:
            t_limit = time_second
        else:
            t_limit = time_rest
        tenure = base_tenure + (i % 9) * 2
        seed_counter += 137

        ms_try, orders_try = _tabu_search(
            dur_flat, machines_mat, num_jobs, num_machines,
            init_orders, num_ops, job_offset, total_ops,
            max_iter=1000000,
            tabu_tenure=tenure,
            time_limit=t_limit,
            seed=seed_counter,
            bufs=bufs,
        )
        if ms_try < best_makespan:
            best_makespan = ms_try
            best_orders = orders_try

    # Final evaluation for start times
    ms_final, flat_st_final = _evaluate_flat(
        dur_flat, num_jobs, num_machines, best_orders, num_ops, job_offset, total_ops
    )

    machine_schedules: list[list[dict[str, int]]] = [[] for _ in range(num_machines)]
    for m in range(num_machines):
        for j, o in best_orders[m]:
            flat_idx = job_offset[j] + o
            s = flat_st_final[flat_idx]
            d = dur_flat[flat_idx]
            machine_schedules[m].append({
                "job_id": j,
                "operation_index": o,
                "start_time": s,
                "end_time": s + d,
                "duration": d,
            })

    return {
        "name": instance["name"],
        "makespan": ms_final,
        "machine_schedules": machine_schedules,
        "solved_by": "TabuSearchBaseline",
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