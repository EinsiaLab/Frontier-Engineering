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


def _compute_makespan_fast(machine_schedules: list[list[dict[str, int]]]) -> int:
    """Fast makespan computation."""
    makespan = 0
    for ops in machine_schedules:
        if ops:
            makespan = max(makespan, max(op["end_time"] for op in ops))
    return makespan


def _recompute_schedule(
    durations: list[list[int]],
    machines: list[list[int]],
    machine_schedules: list[list[dict[str, int]]],
    changed_machine: int,
    new_order: list[tuple[int, int]],
) -> tuple[list[list[dict[str, int]]], int]:
    """Recompute schedule after changing one machine's operation order.
    Returns new schedules and new makespan.
    """
    # Build job end times from unchanged machines
    job_op_end = {}  # (job_id, op_idx) -> end_time
    for m_id, ops in enumerate(machine_schedules):
        if m_id != changed_machine:
            for op in ops:
                job_op_end[(op["job_id"], op["operation_index"])] = op["end_time"]

    # Create new schedules
    new_schedules = [list(ops) for ops in machine_schedules]

    # Rebuild changed machine
    new_ops = []
    machine_time = 0
    makespan = 0

    for job_id, op_idx in new_order:
        duration = durations[job_id][op_idx]

        # Job predecessor constraint
        job_ready = 0
        if op_idx > 0:
            pred_end = job_op_end.get((job_id, op_idx - 1), 0)
            job_ready = pred_end

        start = max(machine_time, job_ready)
        end = start + duration
        new_ops.append({
            "job_id": job_id,
            "operation_index": op_idx,
            "start_time": start,
            "end_time": end,
            "duration": duration,
        })
        machine_time = end
        job_op_end[(job_id, op_idx)] = end

    new_schedules[changed_machine] = new_ops

    # Compute makespan
    for m_id, ops in enumerate(new_schedules):
        if ops:
            makespan = max(makespan, max(op["end_time"] for op in ops))

    return new_schedules, makespan


def _fast_critical_block_improve(
    durations: list[list[int]],
    machines: list[list[int]],
    machine_schedules: list[list[dict[str, int]]],
    max_iterations: int = 15,
) -> tuple[int, list[list[dict[str, int]]]]:
    """Fast improvement focusing on critical blocks."""
    num_machines = len(machine_schedules)

    # Compute current makespan
    makespan = _compute_makespan_fast(machine_schedules)

    best_makespan = makespan
    best_schedules = [list(ops) for ops in machine_schedules]

    def build_job_op_info(schedules):
        info = {}
        for m_id, ops in enumerate(schedules):
            for op in ops:
                info[(op['job_id'], op['operation_index'])] = {
                    'machine': m_id,
                    'start': op['start_time'],
                    'end': op['end_time'],
                    'duration': op['duration']
                }
        return info

    def recompute_machine_schedule(m_id, op_order, job_op_info, durations):
        """Recompute schedule for a single machine given operation order."""
        new_ops = []
        machine_time = 0

        for job_id, op_idx in op_order:
            duration = durations[job_id][op_idx]

            # Job predecessor constraint
            job_ready = 0
            if op_idx > 0:
                pred_key = (job_id, op_idx - 1)
                if pred_key in job_op_info:
                    job_ready = job_op_info[pred_key]['end']

            start = max(machine_time, job_ready)
            end = start + duration
            new_ops.append({
                'job_id': job_id,
                'operation_index': op_idx,
                'start_time': start,
                'end_time': end,
                'duration': duration,
            })
            machine_time = end

        return new_ops

    for iteration in range(max_iterations):
        improved = False

        # Find operations on critical path (ending at makespan)
        job_op_info = build_job_op_info(best_schedules)

        # Trace critical path backwards from makespan
        critical_ops = set()
        for m_id, ops in enumerate(best_schedules):
            for op in ops:
                if op['end_time'] == best_makespan:
                    # Trace back through this operation's critical path
                    current = (op['job_id'], op['operation_index'])
                    while current:
                        critical_ops.add(current)
                        j, o = current

                        # Find predecessor on critical path
                        pred_job = (j, o - 1) if o > 0 else None
                        pred_machine = None

                        # Check job predecessor
                        job_pred_end = job_op_info[pred_job]['end'] if pred_job and pred_job in job_op_info else 0

                        # Check machine predecessor
                        m = job_op_info[current]['machine']
                        m_ops = best_schedules[m]
                        for i, mop in enumerate(m_ops):
                            if (mop['job_id'], mop['operation_index']) == current and i > 0:
                                pred_machine = (m_ops[i-1]['job_id'], m_ops[i-1]['operation_index'])
                                break

                        machine_pred_end = job_op_info[pred_machine]['end'] if pred_machine and pred_machine in job_op_info else 0

                        if job_pred_end >= machine_pred_end and pred_job:
                            current = pred_job
                        elif pred_machine:
                            current = pred_machine
                        else:
                            current = None

        # Find critical machines (machines with critical operations)
        critical_machines = set()
        for m_id, ops in enumerate(best_schedules):
            for op in ops:
                if (op['job_id'], op['operation_index']) in critical_ops:
                    critical_machines.add(m_id)
                    break

        # Try swapping adjacent operations on critical machines
        for m_id in critical_machines:
            ops = best_schedules[m_id]
            if len(ops) < 2:
                continue

            # Find critical blocks (consecutive critical ops)
            for i in range(len(ops) - 1):
                op1_key = (ops[i]['job_id'], ops[i]['operation_index'])
                op2_key = (ops[i+1]['job_id'], ops[i+1]['operation_index'])

                # Only swap if at least one is critical
                if op1_key not in critical_ops and op2_key not in critical_ops:
                    continue

                # Try swap
                test_schedules = [list(m_ops) for m_ops in best_schedules]
                test_schedules[m_id][i], test_schedules[m_id][i+1] = \
                    test_schedules[m_id][i+1], test_schedules[m_id][i]

                # Rebuild just this machine's schedule
                op_order = [(op['job_id'], op['operation_index']) for op in test_schedules[m_id]]
                test_schedules[m_id] = recompute_machine_schedule(
                    m_id, op_order, job_op_info, durations
                )

                # Compute new makespan
                test_makespan = max(op['end_time'] for ops in test_schedules for op in ops)

                if test_makespan < best_makespan:
                    best_makespan = test_makespan
                    best_schedules = test_schedules
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return best_makespan, best_schedules


def _enhanced_local_search(
    durations: list[list[int]],
    machines: list[list[int]],
    machine_schedules: list[list[dict[str, int]]],
    max_iterations: int = 8,
) -> tuple[int, list[list[dict[str, int]]]]:
    """Enhanced local search with swap and insertion moves on all machines."""
    num_machines = len(machine_schedules)

    best_makespan = _compute_makespan_fast(machine_schedules)
    best_schedules = [list(ops) for ops in machine_schedules]

    for iteration in range(max_iterations):
        improved = False

        # Try all machines, prioritizing those with operations near makespan
        machine_priority = []
        for m_id, ops in enumerate(machine_schedules):
            max_end = max((op["end_time"] for op in ops), default=0)
            machine_priority.append((max_end, m_id))
        machine_priority.sort(reverse=True)

        for _, m_id in machine_priority:
            ops = best_schedules[m_id]
            n = len(ops)
            if n < 2:
                continue

            # Get current operation order
            op_order = [(op["job_id"], op["operation_index"]) for op in ops]

            # Try adjacent swaps
            for i in range(n - 1):
                new_order = op_order.copy()
                new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]

                test_schedules, test_makespan = _recompute_schedule(
                    durations, machines, best_schedules, m_id, new_order
                )

                if test_makespan < best_makespan:
                    best_makespan = test_makespan
                    best_schedules = test_schedules
                    improved = True
                    break

            if improved:
                break

            # Try forward insertions (move operation i to position j > i)
            for i in range(n):
                for j in range(i + 2, min(i + 7, n + 1)):  # Extended search depth
                    new_order = op_order.copy()
                    op = new_order.pop(i)
                    new_order.insert(j - 1, op)

                    test_schedules, test_makespan = _recompute_schedule(
                        durations, machines, best_schedules, m_id, new_order
                    )

                    if test_makespan < best_makespan:
                        best_makespan = test_makespan
                        best_schedules = test_schedules
                        improved = True
                        break

                if improved:
                    break

            if improved:
                break

            # Try backward insertions (move operation i to position j < i)
            for i in range(1, n):
                for j in range(max(0, i - 6), i):  # Extended search depth
                    new_order = op_order.copy()
                    op = new_order.pop(i)
                    new_order.insert(j, op)

                    test_schedules, test_makespan = _recompute_schedule(
                        durations, machines, best_schedules, m_id, new_order
                    )

                    if test_makespan < best_makespan:
                        best_makespan = test_makespan
                        best_schedules = test_schedules
                        improved = True
                        break

                if improved:
                    break

            if improved:
                break

        if not improved:
            break

    return best_makespan, best_schedules


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Multi-pass greedy scheduler with hybrid local search improvement.

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
    total_operations = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines) + 1

    # Precompute total work for each job
    job_total_work = [sum(job_durations) for job_durations in durations]

    # Precompute operations count per job
    job_num_ops = [len(job) for job in durations]

    # Precompute machine workload for bottleneck identification
    machine_workload = [0] * num_machines
    for job_id in range(num_jobs):
        for op_idx, machine_id in enumerate(machines[job_id]):
            machine_workload[machine_id] += durations[job_id][op_idx]

    # Precompute look-ahead: remaining work AFTER each operation
    job_lookahead = []
    for job_id in range(num_jobs):
        lookahead = [0] * len(durations[job_id])
        total = 0
        for op_idx in range(len(durations[job_id]) - 1, -1, -1):
            lookahead[op_idx] = total
            total += durations[job_id][op_idx]
        job_lookahead.append(lookahead)

    # Precompute operation position ratio (op_idx / num_ops) for tie-breaking
    job_op_ratio = []
    for job_id in range(num_jobs):
        n_ops = len(durations[job_id])
        job_op_ratio.append([op_idx / n_ops if n_ops > 0 else 0 for op_idx in range(n_ops)])

    # Priority rules: each takes (candidate, remaining_work, remaining_ops, lookahead, op_ratio)
    # candidate = (est, duration, job_id, op_idx, machine_id)
    priority_rules = [
        lambda c, rw, ro, la, opr: (c[0], c[1], c[2]),                                    # SPT
        lambda c, rw, ro, la, opr: (c[0], -c[1], c[2]),                                   # LPT
        lambda c, rw, ro, la, opr: (c[0], -rw[c[2]], c[2]),                               # MWR
        lambda c, rw, ro, la, opr: (c[0], rw[c[2]], c[2]),                                # LWR
        lambda c, rw, ro, la, opr: (c[0], -ro[c[2]], c[2]),                               # MOR
        lambda c, rw, ro, la, opr: (c[0], c[2]),                                          # FIFO
        lambda c, rw, ro, la, opr: (c[0], -rw[c[2]], c[1], c[2]),                        # MWKR
        lambda c, rw, ro, la, opr: (c[0], rw[c[2]], c[1], c[2]),                         # LWKR
        lambda c, rw, ro, la, opr: (c[0], -ro[c[2]], c[1], c[2]),                        # MOR+SPT
        lambda c, rw, ro, la, opr: (c[0], -rw[c[2]], -machine_workload[c[4]], c[2]),     # MWR+MBN
        lambda c, rw, ro, la, opr: (c[0], c[1], -machine_workload[c[4]], c[2]),          # SPT+MBN
        lambda c, rw, ro, la, opr: (c[0], rw[c[2]] - c[1], c[2]),                        # SLK
        lambda c, rw, ro, la, opr: (c[0], -rw[c[2]] * 10 + c[1], c[2]),                  # WMWR+SPT
        lambda c, rw, ro, la, opr: (c[0], -ro[c[2]] * 10 + c[1], c[2]),                  # WMOR+SPT
        lambda c, rw, ro, la, opr: (c[0], -rw[c[2]], -c[1], c[2]),                       # MWR+LPT
        lambda c, rw, ro, la, opr: (c[0], -rw[c[2]] - c[1], c[2]),                       # MWR-SPT
        # Look-ahead based rules
        lambda c, rw, ro, la, opr: (c[0], -la[c[2]][c[3]], c[2]),                        # Max look-ahead
        lambda c, rw, ro, la, opr: (c[0], la[c[2]][c[3]], c[2]),                         # Min look-ahead
        lambda c, rw, ro, la, opr: (c[0], -la[c[2]][c[3]], c[1], c[2]),                  # LA+SPT
        lambda c, rw, ro, la, opr: (c[0], -la[c[2]][c[3]], -c[1], c[2]),                 # LA+LPT
        lambda c, rw, ro, la, opr: (c[0], -la[c[2]][c[3]], -rw[c[2]], c[2]),             # LA+MWR
        lambda c, rw, ro, la, opr: (c[0], -opr[c[2]][c[3]], c[2]),                       # Early operation first
        lambda c, rw, ro, la, opr: (c[0], opr[c[2]][c[3]], c[2]),                        # Late operation first
        lambda c, rw, ro, la, opr: (c[0], -la[c[2]][c[3]] - c[1], c[2]),                 # LA-SPT
        lambda c, rw, ro, la, opr: (c[0], -la[c[2]][c[3]] * 10 + c[1], c[2]),            # WLA+SPT
        lambda c, rw, ro, la, opr: (c[0], -la[c[2]][c[3]], -machine_workload[c[4]], c[2]),  # LA+MBN
        # Additional combined rules for better exploration
        lambda c, rw, ro, la, opr: (c[0], -la[c[2]][c[3]] - rw[c[2]], c[2]),             # LA+MWR combined
        lambda c, rw, ro, la, opr: (c[0], -la[c[2]][c[3]] * 5 - rw[c[2]] * 5, c[2]),     # Weighted LA+MWR
        lambda c, rw, ro, la, opr: (c[0], -ro[c[2]] - la[c[2]][c[3]], c[2]),             # MOR+LA
        lambda c, rw, ro, la, opr: (c[0], -(rw[c[2]] + la[c[2]][c[3]]) / max(c[1], 1), c[2]),  # Work per duration
        lambda c, rw, ro, la, opr: (c[0], -machine_workload[c[4]] * 2 - rw[c[2]], c[2]), # Double bottleneck
        lambda c, rw, ro, la, opr: (c[0], c[1] - la[c[2]][c[3]], c[2]),                  # SPT-LA
        lambda c, rw, ro, la, opr: (c[0], -la[c[2]][c[3]], -machine_workload[c[4]] * 2, c[2]),  # LA+Double MBN
        lambda c, rw, ro, la, opr: (c[0], -(rw[c[2]] * ro[c[2]]), c[2]),                 # Work*Ops remaining
        lambda c, rw, ro, la, opr: (c[0], -la[c[2]][c[3]], -ro[c[2]], c[2]),             # LA+MOR
        lambda c, rw, ro, la, opr: (c[0], -la[c[2]][c[3]] / max(c[1], 1), c[2]),         # LA per duration
        lambda c, rw, ro, la, opr: (c[0], -(la[c[2]][c[3]] + c[1]), c[2]),               # LA+duration
    ]

    best_makespan = float('inf')
    best_schedules: list[list[dict[str, int]]] = []

    # Store top solutions for local search diversification
    top_solutions: list[tuple[int, list[list[dict[str, int]]]]] = []

    for priority_fn in priority_rules:
        next_op = [0] * num_jobs
        job_ready = [0] * num_jobs
        machine_ready = [0] * num_machines
        remaining_work = job_total_work.copy()
        remaining_ops = job_num_ops.copy()

        machine_schedules: list[list[dict[str, int]]] = [
            [] for _ in range(num_machines)
        ]

        scheduled = 0
        while scheduled < total_operations:
            candidates: list[tuple[int, int, int, int, int]] = []
            for job_id in range(num_jobs):
                op_idx = next_op[job_id]
                if op_idx >= len(durations[job_id]):
                    continue

                machine_id = machines[job_id][op_idx]
                duration = durations[job_id][op_idx]
                est = max(job_ready[job_id], machine_ready[machine_id])
                candidates.append((est, duration, job_id, op_idx, machine_id))

            if not candidates:
                raise RuntimeError("No schedulable operation found.")

            est, duration, job_id, op_idx, machine_id = min(
                candidates,
                key=lambda x: priority_fn(x, remaining_work, remaining_ops, job_lookahead, job_op_ratio),
            )
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
            remaining_work[job_id] -= duration
            remaining_ops[job_id] -= 1
            scheduled += 1

        makespan = max(job_ready) if job_ready else 0

        # Apply fast critical block improvement first
        improved_makespan, improved_schedules = _fast_critical_block_improve(
            durations, machines, machine_schedules, max_iterations=15
        )

        if improved_makespan < best_makespan:
            best_makespan = improved_makespan
            best_schedules = improved_schedules
        elif makespan < best_makespan:
            best_makespan = makespan
            best_schedules = machine_schedules

        # Keep solutions within 15% of best for enhanced local search
        if improved_makespan <= best_makespan * 1.15:
            top_solutions.append((improved_makespan, [list(ops) for ops in improved_schedules]))

    # Apply enhanced local search to top solutions for further improvement
    for sol_makespan, sol_schedules in top_solutions[:10]:  # Limit to top 10
        final_makespan, final_schedules = _enhanced_local_search(
            durations, machines, sol_schedules, max_iterations=10
        )

        if final_makespan < best_makespan:
            best_makespan = final_makespan
            best_schedules = final_schedules

    return {
        "name": instance["name"],
        "makespan": best_makespan,
        "machine_schedules": best_schedules,
        "solved_by": "GreedyMultiPassHybridLocalSearchBaseline",
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