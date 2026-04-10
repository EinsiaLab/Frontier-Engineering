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


import random
import heapq


def _greedy_schedule(durations, machines, num_jobs, num_machines, total_ops, key_func):
    """Build a schedule using a greedy dispatching rule defined by key_func.
    Returns (makespan, operation_order) where operation_order is list of (job_id, op_idx)."""
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    order = []
    remaining_work = [sum(durations[j]) for j in range(num_jobs)]

    scheduled = 0
    while scheduled < total_ops:
        best = None
        best_key = None
        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= len(durations[job_id]):
                continue
            machine_id = machines[job_id][op_idx]
            duration = durations[job_id][op_idx]
            est = max(job_ready[job_id], machine_ready[machine_id])
            k = key_func(est, duration, job_id, op_idx, machine_id, remaining_work[job_id], job_ready[job_id])
            if best_key is None or k < best_key:
                best_key = k
                best = (est, duration, job_id, op_idx, machine_id)

        est, duration, job_id, op_idx, machine_id = best
        end = est + duration
        next_op[job_id] += 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end
        remaining_work[job_id] -= duration
        order.append((job_id, op_idx))
        scheduled += 1

    makespan = max(job_ready)
    return makespan, order


def _build_schedule_from_order(durations, machines, num_jobs, num_machines, order):
    """Given an operation order, compute the actual schedule respecting precedence and machine constraints."""
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    machine_schedules = [[] for _ in range(num_machines)]

    for job_id, op_idx in order:
        machine_id = machines[job_id][op_idx]
        duration = durations[job_id][op_idx]
        est = max(job_ready[job_id], machine_ready[machine_id])
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

    return max(job_ready), machine_schedules


def _compute_makespan_from_order(durations, machines, num_jobs, num_machines, order):
    """Quickly compute makespan from operation order."""
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    for job_id, op_idx in order:
        machine_id = machines[job_id][op_idx]
        duration = durations[job_id][op_idx]
        est = max(job_ready[job_id], machine_ready[machine_id])
        end = est + duration
        job_ready[job_id] = end
        machine_ready[machine_id] = end
    return max(job_ready)


def _check_swap_feasible(order, i, j, job_positions):
    """Check if swapping positions i and j maintains precedence."""
    ji, oi = order[i]
    jj, oj = order[j]
    
    # After swap: position i has (jj, oj), position j has (ji, oi)
    # For job ji, op oi moves from position i to position j (j > i means later)
    # For job jj, op oj moves from position j to position i (i < j means earlier)
    
    # Check job jj: op oj moves to position i (earlier)
    # Its predecessor (oj-1) must be before position i
    if oj > 0:
        pred_pos = job_positions.get((jj, oj - 1), -1)
        if pred_pos >= i:  # predecessor at or after new position
            return False
    # Its successor (oj+1) must be after position i
    if (jj, oj + 1) in job_positions:
        succ_pos = job_positions[(jj, oj + 1)]
        if succ_pos <= i:
            return False
    
    # Check job ji: op oi moves to position j (later)
    if oi > 0:
        pred_pos = job_positions.get((ji, oi - 1), -1)
        if pred_pos >= j:
            return False
    if (ji, oi + 1) in job_positions:
        succ_pos = job_positions[(ji, oi + 1)]
        if succ_pos <= j:
            return False
    
    return True


def _get_critical_blocks(durations, machines, num_jobs, num_machines, order):
    """Find critical blocks: consecutive critical operations on the same machine."""
    n = len(order)
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    start_times = [0] * n
    end_times = [0] * n
    job_pred = [-1] * n
    machine_pred = [-1] * n
    
    last_on_machine = [-1] * num_machines
    last_op_of_job = [-1] * num_jobs
    
    for i, (job_id, op_idx) in enumerate(order):
        machine_id = machines[job_id][op_idx]
        duration = durations[job_id][op_idx]
        jp = last_op_of_job[job_id]
        mp = last_on_machine[machine_id]
        job_pred[i] = jp
        machine_pred[i] = mp
        est = max(job_ready[job_id], machine_ready[machine_id])
        end = est + duration
        start_times[i] = est
        end_times[i] = end
        job_ready[job_id] = end
        machine_ready[machine_id] = end
        last_op_of_job[job_id] = i
        last_on_machine[machine_id] = i
    
    makespan = max(end_times)
    
    # Trace critical path
    critical_idx = end_times.index(makespan)
    critical_set = set()
    stack = [critical_idx]
    while stack:
        idx = stack.pop()
        if idx < 0 or idx in critical_set:
            continue
        critical_set.add(idx)
        st = start_times[idx]
        jp = job_pred[idx]
        mp = machine_pred[idx]
        if jp >= 0 and end_times[jp] == st:
            stack.append(jp)
        if mp >= 0 and end_times[mp] == st:
            stack.append(mp)
    
    # Build machine ops
    machine_ops = [[] for _ in range(num_machines)]
    for idx, (jid, oidx) in enumerate(order):
        mid = machines[jid][oidx]
        machine_ops[mid].append(idx)
    
    # Find critical blocks on each machine
    blocks = []
    for m in range(num_machines):
        ops = machine_ops[m]
        block = []
        for idx in ops:
            if idx in critical_set:
                block.append(idx)
            else:
                if len(block) >= 2:
                    blocks.append((m, block))
                block = []
        if len(block) >= 2:
            blocks.append((m, block))
    
    return makespan, critical_set, blocks, machine_ops


def _generate_moves(order, blocks):
    """Generate N5-style moves from critical blocks."""
    moves = []
    for m, block in blocks:
        if len(block) < 2:
            continue
        # Swap first two in block
        i, j = block[0], block[1]
        if order[i][0] != order[j][0]:
            moves.append((i, j))
        # Swap last two in block
        i, j = block[-2], block[-1]
        if order[i][0] != order[j][0]:
            moves.append((i, j))
        # Interior swaps
        for k in range(1, len(block) - 1):
            i, j = block[k], block[k + 1]
            if order[i][0] != order[j][0]:
                moves.append((i, j))
    return moves


def _local_search(durations, machines, num_jobs, num_machines, order, makespan, time_limit):
    """Critical-block based local search (N5 neighborhood) with tabu."""
    start_time = time.perf_counter()
    best_makespan = makespan
    best_order = list(order)
    current_makespan = makespan
    current_order = list(order)
    
    tabu_list = {}
    iteration = 0
    rng = random.Random(42)
    base_tenure = max(7, num_jobs)
    no_improve_count = 0
    max_no_improve = max(150, num_jobs * num_machines)
    restarts = 0
    max_restarts = 12
    
    while (time.perf_counter() - start_time) < time_limit:
        if no_improve_count >= max_no_improve:
            restarts += 1
            if restarts > max_restarts:
                break
            current_order = list(best_order)
            current_makespan = best_makespan
            temp_order = list(current_order)
            temp_positions = {}
            for idx2, (jid2, oidx2) in enumerate(temp_order):
                temp_positions[(jid2, oidx2)] = idx2
            m_ops = [[] for _ in range(num_machines)]
            for idx2, (jid2, oidx2) in enumerate(temp_order):
                m_ops[machines[jid2][oidx2]].append(idx2)
            n_perturb = max(3, num_jobs // 3) + restarts * 2
            for _ in range(n_perturb):
                m = rng.randint(0, num_machines - 1)
                ops = m_ops[m]
                if len(ops) < 2:
                    continue
                k = rng.randint(0, len(ops) - 2)
                ii, jj = ops[k], ops[k + 1]
                if temp_order[ii][0] == temp_order[jj][0]:
                    continue
                if _check_swap_feasible(temp_order, ii, jj, temp_positions):
                    oa, ob = temp_order[ii], temp_order[jj]
                    temp_order[ii], temp_order[jj] = temp_order[jj], temp_order[ii]
                    temp_positions[oa] = jj
                    temp_positions[ob] = ii
            current_order = temp_order
            current_makespan = _compute_makespan_from_order(durations, machines, num_jobs, num_machines, current_order)
            no_improve_count = 0
            tabu_list.clear()
        iteration += 1
        tabu_tenure = base_tenure + rng.randint(0, base_tenure // 2)
        
        cur_ms, critical_set, blocks, machine_ops = _get_critical_blocks(
            durations, machines, num_jobs, num_machines, current_order
        )
        
        job_positions = {}
        for idx, (jid, oidx) in enumerate(current_order):
            job_positions[(jid, oidx)] = idx
        
        moves = _generate_moves(current_order, blocks)
        
        best_move = None
        best_move_ms = float('inf')
        best_non_tabu_move = None
        best_non_tabu_ms = float('inf')
        
        for i, j in moves:
            if (time.perf_counter() - start_time) >= time_limit:
                break
            
            if not _check_swap_feasible(current_order, i, j, job_positions):
                continue
            
            new_order = list(current_order)
            new_order[i], new_order[j] = new_order[j], new_order[i]
            new_ms = _compute_makespan_from_order(durations, machines, num_jobs, num_machines, new_order)
            
            move_key = (current_order[i], current_order[j])
            is_tabu = move_key in tabu_list and tabu_list[move_key] > iteration
            
            # Aspiration: accept tabu move if it improves best known
            if new_ms < best_makespan:
                if new_ms < best_move_ms:
                    best_move_ms = new_ms
                    best_move = (i, j, new_order, new_ms)
            
            if not is_tabu:
                if new_ms < best_non_tabu_ms:
                    best_non_tabu_ms = new_ms
                    best_non_tabu_move = (i, j, new_order, new_ms)
        
        # Choose move
        chosen = None
        if best_move is not None and best_move_ms < best_makespan:
            chosen = best_move
        elif best_non_tabu_move is not None:
            chosen = best_non_tabu_move
        elif best_move is not None:
            chosen = best_move
        
        if chosen is None:
            no_improve_count = max_no_improve  # trigger restart
            continue
        
        i, j, new_order, new_ms = chosen
        # Add reverse move to tabu
        rev_key = (current_order[j], current_order[i])
        tabu_list[rev_key] = iteration + tabu_tenure
        
        current_order = new_order
        current_makespan = new_ms
        
        if new_ms < best_makespan:
            best_makespan = new_ms
            best_order = list(new_order)
            no_improve_count = 0
        else:
            no_improve_count += 1

    return best_makespan, best_order


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Multi-rule greedy + local search scheduler."""
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    total_operations = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines) + 1

    # Define multiple dispatching rules
    rules = [
        # EST + SPT
        lambda est, dur, jid, oidx, mid, rw, jr: (est, dur, jid),
        # EST + LPT (longest processing time)
        lambda est, dur, jid, oidx, mid, rw, jr: (est, -dur, jid),
        # EST + MWR (most work remaining)
        lambda est, dur, jid, oidx, mid, rw, jr: (est, -rw, jid),
        # EST + LWR (least work remaining)  
        lambda est, dur, jid, oidx, mid, rw, jr: (est, rw, jid),
        # Pure SPT
        lambda est, dur, jid, oidx, mid, rw, jr: (dur, est, jid),
        # Pure MWR
        lambda est, dur, jid, oidx, mid, rw, jr: (-rw, est, jid),
        # FIFO (earliest job ready)
        lambda est, dur, jid, oidx, mid, rw, jr: (jr, est, jid),
        # EST + SPT with MWR tiebreak
        lambda est, dur, jid, oidx, mid, rw, jr: (est, dur, -rw, jid),
    ]

    best_makespan = float('inf')
    best_order = None

    solve_start = time.perf_counter()

    for rule in rules:
        ms, order = _greedy_schedule(durations, machines, num_jobs, num_machines, total_operations, rule)
        if ms < best_makespan:
            best_makespan = ms
            best_order = order

    # Random perturbation: generate additional schedules with randomized tie-breaking
    rng = random.Random(42)
    time_for_random = min(2.0, 10.0 - (time.perf_counter() - solve_start))
    while (time.perf_counter() - solve_start) < max(0.5, time_for_random):
        eps = rng.random() * 0.3
        def _rand_rule(est, dur, jid, oidx, mid, rw, jr, _eps=eps, _rng=rng):
            return (est + int(_eps * dur * _rng.random()), -rw + int(_eps * dur * _rng.random()), jid)
        ms, order = _greedy_schedule(durations, machines, num_jobs, num_machines, total_operations, _rand_rule)
        if ms < best_makespan:
            best_makespan = ms
            best_order = order

    # Local search on best solution
    time_remaining = min(8.0, 12.0 - (time.perf_counter() - solve_start))
    if time_remaining > 0.1:
        best_makespan, best_order = _local_search(
            durations, machines, num_jobs, num_machines,
            best_order, best_makespan, time_remaining
        )

    makespan, machine_schedules = _build_schedule_from_order(
        durations, machines, num_jobs, num_machines, best_order
    )

    return {
        "name": instance["name"],
        "makespan": makespan,
        "machine_schedules": machine_schedules,
        "solved_by": "MultiRuleGreedyLocalSearch",
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
