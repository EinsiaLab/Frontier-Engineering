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
import random
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


def _compute_tail_times(durations, machines, num_machines):
    """Compute tail times (time remaining from each operation to end of job)."""
    num_jobs = len(durations)
    tail = [[0] * len(durations[j]) for j in range(num_jobs)]
    for j in range(num_jobs):
        n_ops = len(durations[j])
        if n_ops > 0:
            tail[j][n_ops - 1] = durations[j][n_ops - 1]
            for k in range(n_ops - 2, -1, -1):
                tail[j][k] = durations[j][k] + tail[j][k + 1]
    return tail


def _greedy_schedule(durations, machines, num_jobs, num_machines, priority_rule="EST_LRT"):
    """Build a schedule using a greedy priority rule."""
    total_operations = sum(len(job) for job in durations)
    
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines

    # Precompute tail times for LRT rule
    tail = None
    if "LRT" in priority_rule or "MWKR" in priority_rule:
        tail = _compute_tail_times(durations, machines, num_machines)

    op_start = [[0] * len(durations[j]) for j in range(num_jobs)]
    machine_job_order = [[] for _ in range(num_machines)]

    scheduled = 0
    while scheduled < total_operations:
        best = None
        best_key = None
        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= len(durations[job_id]):
                continue
            machine_id = machines[job_id][op_idx]
            duration = durations[job_id][op_idx]
            est = max(job_ready[job_id], machine_ready[machine_id])

            if priority_rule == "EST_SPT":
                key = (est, duration, job_id)
            elif priority_rule == "EST_LRT":
                key = (est, -tail[job_id][op_idx], job_id)
            elif priority_rule == "EST_MWKR":
                key = (est, -tail[job_id][op_idx], job_id)
            elif priority_rule == "EST_LPT":
                key = (est, -duration, job_id)
            else:
                key = (est, duration, job_id)

            if best_key is None or key < best_key:
                best_key = key
                best = (est, duration, job_id, op_idx, machine_id)

        est, duration, job_id, op_idx, machine_id = best
        end = est + duration

        op_start[job_id][op_idx] = est
        machine_job_order[machine_id].append((job_id, op_idx))

        next_op[job_id] += 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end
        scheduled += 1

    makespan = max(job_ready) if job_ready else 0
    return makespan, op_start, machine_job_order


def _build_machine_schedules(durations, op_start, machine_job_order, num_machines):
    """Build machine_schedules from op_start and machine_job_order."""
    machine_schedules = [[] for _ in range(num_machines)]
    for m in range(num_machines):
        for job_id, op_idx in machine_job_order[m]:
            st = op_start[job_id][op_idx]
            dur = durations[job_id][op_idx]
            machine_schedules[m].append({
                "job_id": job_id,
                "operation_index": op_idx,
                "start_time": st,
                "end_time": st + dur,
                "duration": dur,
            })
    return machine_schedules


def _left_shift_schedule(durations, machines, num_jobs, num_machines, job_sequences_per_machine):
    """Given machine orderings, compute earliest start times (semi-active schedule)."""
    num_ops = [len(durations[j]) for j in range(num_jobs)]
    op_start = [[0] * num_ops[j] for j in range(num_jobs)]
    
    # Build: for each (job, op_idx), which position on which machine
    op_machine_pos = [[0] * num_ops[j] for j in range(num_jobs)]
    for m in range(num_machines):
        for pos, (job_id, op_idx) in enumerate(job_sequences_per_machine[m]):
            op_machine_pos[job_id][op_idx] = pos

    # Topological order: process operations in order respecting both job and machine precedence
    # Use a simple iterative approach
    job_next_scheduled = [0] * num_jobs
    machine_next_scheduled = [0] * num_machines
    machine_ready = [0] * num_machines
    job_ready = [0] * num_jobs

    total_ops = sum(num_ops)
    scheduled = 0
    
    while scheduled < total_ops:
        for j in range(num_jobs):
            k = job_next_scheduled[j]
            if k >= num_ops[j]:
                continue
            m = machines[j][k]
            pos = op_machine_pos[j][k]
            if machine_next_scheduled[m] != pos:
                continue
            # Both job and machine predecessors are done
            est = max(job_ready[j], machine_ready[m])
            dur = durations[j][k]
            op_start[j][k] = est
            end = est + dur
            job_ready[j] = end
            machine_ready[m] = end
            job_next_scheduled[j] = k + 1
            machine_next_scheduled[m] = pos + 1
            scheduled += 1

    makespan = max(job_ready) if job_ready else 0
    return makespan, op_start


def _extract_critical_path(durations, machines, num_jobs, num_machines, op_start, machine_job_order):
    """Extract critical path operations."""
    num_ops = [len(durations[j]) for j in range(num_jobs)]
    # Find makespan
    makespan = 0
    for j in range(num_jobs):
        for k in range(num_ops[j]):
            end = op_start[j][k] + durations[j][k]
            if end > makespan:
                makespan = end

    # Build machine position lookup
    machine_pos_lookup = {}
    for m in range(num_machines):
        for pos, (j, k) in enumerate(machine_job_order[m]):
            machine_pos_lookup[(j, k)] = (m, pos)

    # Backward pass: find critical path
    critical = []
    # Start from the operation(s) that end at makespan
    current = None
    for j in range(num_jobs):
        for k in range(num_ops[j]):
            if op_start[j][k] + durations[j][k] == makespan:
                current = (j, k)
                break
        if current:
            break

    while current is not None:
        j, k = current
        critical.append(current)
        st = op_start[j][k]
        if st == 0:
            break
        # Check job predecessor
        found = False
        if k > 0:
            prev_end = op_start[j][k-1] + durations[j][k-1]
            if prev_end == st:
                current = (j, k-1)
                found = True
        if not found:
            # Check machine predecessor
            m, pos = machine_pos_lookup[(j, k)]
            if pos > 0:
                pj, pk = machine_job_order[m][pos - 1]
                prev_end = op_start[pj][pk] + durations[pj][pk]
                if prev_end == st:
                    current = (pj, pk)
                    found = True
        if not found:
            break

    critical.reverse()
    return critical


def _get_critical_blocks(critical, machines, machine_job_order):
    """Get blocks of consecutive critical path operations on the same machine."""
    if not critical:
        return []
    
    machine_pos_lookup = {}
    for m in range(len(machine_job_order)):
        for pos, (j, k) in enumerate(machine_job_order[m]):
            machine_pos_lookup[(j, k)] = (m, pos)
    
    blocks = []
    current_block = [critical[0]]
    current_machine = machines[critical[0][0]][critical[0][1]]
    
    for i in range(1, len(critical)):
        j, k = critical[i]
        m = machines[j][k]
        if m == current_machine:
            # Check if consecutive on machine
            _, pos_prev = machine_pos_lookup[critical[i-1]]
            _, pos_curr = machine_pos_lookup[critical[i]]
            if pos_curr == pos_prev + 1:
                current_block.append(critical[i])
                continue
        if len(current_block) >= 2:
            blocks.append((current_machine, current_block))
        current_block = [critical[i]]
        current_machine = m
    
    if len(current_block) >= 2:
        blocks.append((current_machine, current_block))
    
    return blocks


def _neighborhood_search(durations, machines, num_jobs, num_machines, 
                          makespan, op_start, machine_job_order, time_limit):
    """Tabu search with N5/N7-style neighborhood based on critical path."""
    best_makespan = makespan
    best_machine_job_order = [list(seq) for seq in machine_job_order]
    best_op_start = [list(row) for row in op_start]
    
    current_makespan = makespan
    current_machine_job_order = [list(seq) for seq in machine_job_order]
    current_op_start = [list(row) for row in op_start]
    
    tabu_list = {}
    tabu_tenure = max(10, num_jobs)
    
    start_time = time.perf_counter()
    iteration = 0
    no_improve = 0
    
    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed > time_limit:
            break
        
        iteration += 1
        
        # Extract critical path
        critical = _extract_critical_path(
            durations, machines, num_jobs, num_machines,
            current_op_start, current_machine_job_order
        )
        
        # Get critical blocks
        blocks = _get_critical_blocks(critical, machines, current_machine_job_order)
        
        if not blocks:
            break
        
        # Generate neighborhood moves: swap adjacent pairs in critical blocks
        moves = []
        for m, block in blocks:
            machine_pos_lookup = {}
            for pos, (j, k) in enumerate(current_machine_job_order[m]):
                machine_pos_lookup[(j, k)] = pos
            
            # For each block, try swapping first two and last two operations
            if len(block) >= 2:
                # Swap first two
                pos1 = machine_pos_lookup[block[0]]
                pos2 = machine_pos_lookup[block[1]]
                moves.append((m, pos1, pos2))
                
                # Swap last two
                if len(block) > 2:
                    pos1 = machine_pos_lookup[block[-2]]
                    pos2 = machine_pos_lookup[block[-1]]
                    moves.append((m, pos1, pos2))
                
                # Also try moving first op to end of block and last op to start
                if len(block) >= 3:
                    for i in range(len(block) - 1):
                        pos1 = machine_pos_lookup[block[i]]
                        pos2 = machine_pos_lookup[block[i + 1]]
                        if (m, pos1, pos2) not in moves:
                            moves.append((m, pos1, pos2))
        
        if not moves:
            break
        
        # Evaluate moves
        best_move = None
        best_move_makespan = float('inf')
        best_move_data = None
        
        for m, pos1, pos2 in moves:
            # Check tabu
            op1 = current_machine_job_order[m][pos1]
            op2 = current_machine_job_order[m][pos2]
            move_key = (m, op1, op2)
            is_tabu = move_key in tabu_list and tabu_list[move_key] > iteration
            
            # Apply swap
            new_order = [list(seq) for seq in current_machine_job_order]
            new_order[m][pos1], new_order[m][pos2] = new_order[m][pos2], new_order[m][pos1]
            
            new_ms, new_starts = _left_shift_schedule(
                durations, machines, num_jobs, num_machines, new_order
            )
            
            # Aspiration: accept if better than best known even if tabu
            if is_tabu and new_ms >= best_makespan:
                continue
            
            if new_ms < best_move_makespan:
                best_move_makespan = new_ms
                best_move = (m, pos1, pos2, op1, op2)
                best_move_data = (new_order, new_starts)
        
        if best_move is None:
            # All moves are tabu, reduce tenure or break
            no_improve += 1
            if no_improve > 50:
                break
            continue
        
        m, pos1, pos2, op1, op2 = best_move
        new_order, new_starts = best_move_data
        
        # Update tabu: reverse move is tabu
        tabu_list[(m, op2, op1)] = iteration + tabu_tenure
        
        current_makespan = best_move_makespan
        current_machine_job_order = new_order
        current_op_start = new_starts
        
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_machine_job_order = [list(seq) for seq in current_machine_job_order]
            best_op_start = [list(row) for row in current_op_start]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 200:
                # Restart from best
                current_makespan = best_makespan
                current_machine_job_order = [list(seq) for seq in best_machine_job_order]
                current_op_start = [list(row) for row in best_op_start]
                no_improve = 0
                tabu_tenure = max(10, num_jobs + random.randint(-5, 5))
    
    return best_makespan, best_op_start, best_machine_job_order


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Greedy + local search scheduler on raw benchmark matrices."""
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    num_machines = max(max(row) for row in machines) + 1

    # Try multiple priority rules and pick the best initial solution
    rules = ["EST_LRT", "EST_SPT", "EST_LPT"]
    best_makespan = float('inf')
    best_op_start = None
    best_machine_job_order = None
    
    for rule in rules:
        ms, op_st, mjo = _greedy_schedule(durations, machines, num_jobs, num_machines, rule)
        if ms < best_makespan:
            best_makespan = ms
            best_op_start = op_st
            best_machine_job_order = mjo

    # Determine time budget based on instance size
    total_ops = sum(len(durations[j]) for j in range(num_jobs))
    if total_ops <= 225:  # 15x15
        time_limit = 1.0
    elif total_ops <= 400:  # 20x20
        time_limit = 2.0
    elif total_ops <= 600:  # 30x20
        time_limit = 3.0
    else:
        time_limit = 4.0

    # Run tabu search
    random.seed(42)
    best_makespan, best_op_start, best_machine_job_order = _neighborhood_search(
        durations, machines, num_jobs, num_machines,
        best_makespan, best_op_start, best_machine_job_order,
        time_limit
    )

    machine_schedules = _build_machine_schedules(
        durations, best_op_start, best_machine_job_order, num_machines
    )

    return {
        "name": instance["name"],
        "makespan": best_makespan,
        "machine_schedules": machine_schedules,
        "solved_by": "GreedyTabuSearch",
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
