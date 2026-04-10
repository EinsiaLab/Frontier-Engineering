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


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Multi-pass greedy with extended rules and local search improvement."""
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]
    
    num_jobs = len(durations)
    total_ops = sum(len(j) for j in durations)
    num_machines = max(max(row) for row in machines) + 1
    job_total_work = [sum(j) for j in durations]
    job_num_ops = [len(j) for j in durations]
    
    def run_greedy(priority_key):
        next_op = [0] * num_jobs
        job_ready = [0] * num_jobs
        machine_ready = [0] * num_machines
        job_remaining = job_total_work[:]
        schedules = [[] for _ in range(num_machines)]
        
        for _ in range(total_ops):
            candidates = []
            for j in range(num_jobs):
                op = next_op[j]
                if op >= len(durations[j]):
                    continue
                m, d = machines[j][op], durations[j][op]
                est = max(job_ready[j], machine_ready[m])
                rem = job_remaining[j]
                ops_left = job_num_ops[j] - op
                candidates.append((est, d, rem, j, op, m, ops_left))
            
            if not candidates:
                break
            
            c = min(candidates, key=priority_key)
            est, d, _, j, op, m, _ = c
            end = est + d
            schedules[m].append({
                "job_id": j, "operation_index": op,
                "start_time": est, "end_time": end, "duration": d
            })
            next_op[j] += 1
            job_ready[j] = end
            machine_ready[m] = end
            job_remaining[j] -= d
        
        return max(job_ready) if job_ready else 0, schedules
    
    def rebuild_schedule(sched):
        """Rebuild entire schedule respecting all precedence constraints."""
        job_end = [0] * num_jobs
        mach_end = [0] * num_machines
        new_sched = [[] for _ in range(num_machines)]
        for m, ops in enumerate(sched):
            for op in sorted(ops, key=lambda x: x["start_time"]):
                j, op_idx = op["job_id"], op["operation_index"]
                d = durations[j][op_idx]
                st = max(job_end[j], mach_end[m])
                en = st + d
                new_sched[m].append({
                    "job_id": j, "operation_index": op_idx,
                    "start_time": st, "end_time": en, "duration": d
                })
                job_end[j] = en
                mach_end[m] = en
        return max(job_end), new_sched
    
    def local_search(makespan, schedules):
        """Simple local search with swaps and insertions, critical machine focus."""
        improved = True
        while improved:
            improved = False
            crit_m = max(range(num_machines),
                        key=lambda m: max((op["end_time"] for op in schedules[m]), default=0))
            for m in [crit_m] + [x for x in range(num_machines) if x != crit_m]:
                ops = schedules[m]
                n = len(ops)
                if n < 2:
                    continue
                # Adjacent swaps
                for i in range(n - 1):
                    if ops[i]["job_id"] == ops[i+1]["job_id"]:
                        continue
                    new_ops = ops[:]
                    new_ops[i], new_ops[i+1] = new_ops[i+1], new_ops[i]
                    test_sched = schedules[:]
                    test_sched[m] = new_ops
                    new_ms, new_sched = rebuild_schedule(test_sched)
                    if new_ms < makespan:
                        makespan, schedules = new_ms, new_sched
                        improved = True
                        break
                if improved:
                    break
                # Insertion moves
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        new_ops = ops[:]
                        op = new_ops.pop(i)
                        new_ops.insert(j if j < i else j - 1, op)
                        test_sched = schedules[:]
                        test_sched[m] = new_ops
                        new_ms, new_sched = rebuild_schedule(test_sched)
                        if new_ms < makespan:
                            makespan, schedules = new_ms, new_sched
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return makespan, schedules
    
    # Extended dispatching rules proven effective
    rules = [
        lambda x: (x[0], x[1], x[3]),           # SPT
        lambda x: (x[0], -x[1], x[3]),          # LPT
        lambda x: (x[0], -x[2], x[1], x[3]),    # MWR
        lambda x: (x[0], x[6], x[1], x[3]),     # FOPNR
        lambda x: (x[0], -x[6], x[1], x[3]),    # MOR
        lambda x: (x[0], x[2], x[1], x[3]),     # LWR
        lambda x: (x[0], -x[2], x[6], x[1], x[3]),  # MWR+MOR
        lambda x: (x[0], x[1], x[6], x[3]),     # SPT+FOPNR
        lambda x: (x[0], x[1] - x[2]//10, x[3]),  # Weighted SPT-MWR
        lambda x: (x[0], -x[2] + x[1], x[3]),   # MWR adjusted
        lambda x: (x[0], x[6] - x[2]//20, x[3]), # FOPNR-MWR
        lambda x: (x[0], x[1] + x[6], x[3]),    # SPT+FOPNR weighted
    ]
    
    best_makespan = float('inf')
    best_schedules = None
    
    for rule in rules:
        makespan, schedules = run_greedy(rule)
        if makespan < best_makespan:
            best_makespan = makespan
            best_schedules = schedules
    
    # Apply local search improvement
    best_makespan, best_schedules = local_search(best_makespan, best_schedules)
    
    return {
        "name": instance["name"],
        "makespan": best_makespan,
        "machine_schedules": best_schedules,
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
