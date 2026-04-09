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


def _solve_with_priority(durations, machines, priority_func):
    """Solve using a given priority function for dispatching."""
    num_jobs = len(durations)
    total_ops = sum(len(j) for j in durations)
    num_machines = max(max(r) for r in machines) + 1
    
    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    schedules = [[] for _ in range(num_machines)]
    scheduled = 0
    total_work = [sum(durations[j]) for j in range(num_jobs)]
    
    while scheduled < total_ops:
        candidates = []
        for jid in range(num_jobs):
            op = next_op[jid]
            if op >= len(durations[jid]):
                continue
            mid = machines[jid][op]
            dur = durations[jid][op]
            est = max(job_ready[jid], machine_ready[mid])
            work_rem = sum(durations[jid][op:])
            ops_rem = len(durations[jid]) - op
            slack = work_rem - est if work_rem > 0 else 0
            crit_ratio = work_rem / max(1, est + work_rem)
            candidates.append((est, dur, jid, op, mid, work_rem, ops_rem, slack, crit_ratio, total_work[jid]))
        
        if not candidates:
            break
        
        best = min(candidates, key=priority_func)
        est, dur, jid, op, mid = best[:5]
        end = est + dur
        
        schedules[mid].append({
            "job_id": jid, "operation_index": op,
            "start_time": est, "end_time": end, "duration": dur
        })
        
        next_op[jid] += 1
        job_ready[jid] = end
        machine_ready[mid] = end
        scheduled += 1
    
    return max(job_ready) if job_ready else 0, schedules

def _rebuild_schedule(schedules, durations, machines):
    """Rebuild schedule respecting precedence constraints."""
    num_jobs = len(durations)
    sequences = [[(op["job_id"], op["operation_index"]) for op in ops] for ops in schedules]
    job_end = [0] * num_jobs
    machine_end = [0] * len(schedules)
    new_schedules = [[] for _ in range(len(schedules))]
    
    for mid, seq in enumerate(sequences):
        for jid, op_idx in seq:
            dur = durations[jid][op_idx]
            est = max(job_end[jid], machine_end[mid])
            end = est + dur
            job_end[jid] = end
            machine_end[mid] = end
            new_schedules[mid].append({"job_id": jid, "operation_index": op_idx, "start_time": est, "end_time": end, "duration": dur})
    
    return max(job_end) if job_end else 0, new_schedules

def _identify_critical(schedules, durations, makespan):
    """Find operations on the critical path."""
    num_jobs = len(durations)
    job_end = [0] * num_jobs
    machine_end = [0] * len(schedules)
    op_ec = {}
    
    for mid, ops in enumerate(schedules):
        for op in ops:
            jid, op_idx, dur = op["job_id"], op["operation_index"], op["duration"]
            est = max(job_end[jid], machine_end[mid])
            op_ec[(jid, op_idx)] = est + dur
            job_end[jid] = est + dur
            machine_end[mid] = est + dur
    
    job_lst = [makespan] * num_jobs
    machine_lst = [makespan] * len(schedules)
    critical = set()
    
    for mid in range(len(schedules) - 1, -1, -1):
        for op in reversed(schedules[mid]):
            jid, op_idx, dur = op["job_id"], op["operation_index"], op["duration"]
            lst = min(job_lst[jid], machine_lst[mid]) - dur
            if op_ec.get((jid, op_idx), 0) >= lst:
                critical.add((jid, op_idx))
            job_lst[jid] = lst
            machine_lst[mid] = lst
    return critical

def _get_critical_blocks(ops, critical):
    """Extract critical blocks as lists of consecutive indices."""
    blocks = []
    block = []
    for i, op in enumerate(ops):
        jid, op_idx = op["job_id"], op["operation_index"]
        if (jid, op_idx) in critical:
            block.append(i)
        elif block:
            if len(block) >= 2: blocks.append(block)
            block = []
    if len(block) >= 2: blocks.append(block)
    return blocks

def _local_search(makespan, schedules, durations, machines):
    """Enhanced local search with swaps, insertions, and tabu-based diversification."""
    best_makespan, best_schedules = makespan, [list(ops) for ops in schedules]
    tabu_list = []  # Store recent (makespan, machine_seq_hash) to avoid cycling
    max_tabu = 10
    
    for iteration in range(100):
        improved = False
        critical = _identify_critical(best_schedules, durations, best_makespan)
        
        # Phase 1: Critical block boundary swaps
        for mid in range(len(best_schedules)):
            if improved: break
            ops = best_schedules[mid]
            blocks = _get_critical_blocks(ops, critical)
            for block in blocks:
                if improved: break
                i, j = block[0], block[1]
                ops[i], ops[j] = ops[j], ops[i]
                new_ms, new_sched = _rebuild_schedule(best_schedules, durations, machines)
                seq_hash = sum(hash((op["job_id"], op["operation_index"])) for op in new_sched[mid])
                if new_ms < best_makespan and (new_ms, seq_hash) not in tabu_list:
                    best_makespan, best_schedules = new_ms, new_sched
                    tabu_list.append((new_ms, seq_hash))
                    if len(tabu_list) > max_tabu: tabu_list.pop(0)
                    improved = True
                    break
                ops[i], ops[j] = ops[j], ops[i]
                if len(block) >= 3:
                    i, j = block[-2], block[-1]
                    ops[i], ops[j] = ops[j], ops[i]
                    new_ms, new_sched = _rebuild_schedule(best_schedules, durations, machines)
                    seq_hash = sum(hash((op["job_id"], op["operation_index"])) for op in new_sched[mid])
                    if new_ms < best_makespan and (new_ms, seq_hash) not in tabu_list:
                        best_makespan, best_schedules = new_ms, new_sched
                        tabu_list.append((new_ms, seq_hash))
                        if len(tabu_list) > max_tabu: tabu_list.pop(0)
                        improved = True
                        break
                    ops[i], ops[j] = ops[j], ops[i]
        
        # Phase 2: Adjacent swaps on critical ops
        if not improved:
            for mid in range(len(best_schedules)):
                if improved: break
                ops = best_schedules[mid]
                for i in range(len(ops) - 1):
                    jid1, op1 = ops[i]["job_id"], ops[i]["operation_index"]
                    jid2, op2 = ops[i+1]["job_id"], ops[i+1]["operation_index"]
                    if (jid1, op1) not in critical and (jid2, op2) not in critical:
                        continue
                    ops[i], ops[i+1] = ops[i+1], ops[i]
                    new_ms, new_sched = _rebuild_schedule(best_schedules, durations, machines)
                    if new_ms < best_makespan:
                        best_makespan, best_schedules = new_ms, new_sched
                        improved = True
                        break
                    ops[i], ops[i+1] = ops[i+1], ops[i]
        
        # Phase 3: Insertion moves for critical ops
        if not improved:
            for mid in range(len(best_schedules)):
                if improved: break
                ops = best_schedules[mid]
                n = len(ops)
                for i in range(n):
                    jid, op_idx = ops[i]["job_id"], ops[i]["operation_index"]
                    if (jid, op_idx) not in critical: continue
                    for j in range(n):
                        if j == i or j == i + 1: continue
                        item = ops.pop(i)
                        ops.insert(j if j < i else j - 1, item)
                        new_ms, new_sched = _rebuild_schedule(best_schedules, durations, machines)
                        if new_ms < best_makespan:
                            best_makespan, best_schedules = new_ms, new_sched
                            improved = True
                            break
                        ops.pop(j if j < i else j - 1)
                        ops.insert(i, item)
        
        if not improved:
            break
    
    return best_makespan, best_schedules


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Multi-start dispatching with various priority rules.

    Tries multiple dispatching rules and returns the best solution found.
    """
    durations = instance["duration_matrix"]
    machines = instance["machines_matrix"]
    
    # c = (est, dur, jid, op, mid, work_rem, ops_rem, slack, crit_ratio, total_work)
    rules = [
        ("EST_SPT", lambda c: (c[0], c[1], c[2])),
        ("EST_LPT", lambda c: (c[0], -c[1], c[2])),
        ("EST_MWR", lambda c: (c[0], -c[5], c[2])),
        ("EST_LWR", lambda c: (c[0], c[5], c[2])),
        ("EST_MOR", lambda c: (c[0], -c[6], c[2])),
        ("EST_Slack", lambda c: (c[0], c[7], c[2])),
        ("EST_CR", lambda c: (c[0], -c[8], c[2])),
        ("SPT", lambda c: (c[1], c[0], c[2])),
        ("LPT", lambda c: (-c[1], c[0], c[2])),
        ("MWR", lambda c: (-c[5], c[0], c[2])),
        ("LWR", lambda c: (c[5], c[0], c[2])),
        ("MOR", lambda c: (-c[6], c[0], c[2])),
        ("LOR", lambda c: (c[6], c[0], c[2])),
        ("Slack", lambda c: (c[7], c[0], c[2])),
        ("CR", lambda c: (-c[8], c[0], c[2])),
        ("TotalWork", lambda c: (-c[9], c[0], c[2])),
        ("EST_FIFO", lambda c: (c[0], c[2], c[3])),
        # Weighted combination rules
        ("EST_WMWR", lambda c: (c[0], -c[5] - 0.3 * c[1], c[2])),
        ("EST_WMOR", lambda c: (c[0], -c[6] - 0.01 * c[5], c[2])),
        ("SPT_WMWR", lambda c: (c[1] - 0.05 * c[5], c[0], c[2])),
        ("SlackMWR", lambda c: (c[7] - 0.3 * c[5], c[0], c[2])),
        ("EST_SPTMWR", lambda c: (c[0], c[1] - 0.2 * c[5], c[2])),
        ("MWR_SPT", lambda c: (-c[5] + 0.1 * c[1], c[0], c[2])),
        ("LPT_MWR", lambda c: (-c[1] - 0.1 * c[5], c[0], c[2])),
        ("EST_TW", lambda c: (c[0], -c[9], c[2])),
        ("MWR_TW", lambda c: (-c[5] - 0.1 * c[9], c[0], c[2])),
        ("SPT_TW", lambda c: (c[1] - 0.01 * c[9], c[0], c[2])),
        # Additional sophisticated rules
        ("EST_MWR2", lambda c: (c[0], -c[5] - 0.5 * c[1], c[2])),
        ("EST_MOR2", lambda c: (c[0], -c[6] - 0.1 * c[5], c[2])),
        ("MWR_LPT", lambda c: (-c[5] - 0.2 * c[1], c[0], c[2])),
        ("Slack_SPT", lambda c: (c[7] + 0.5 * c[1], c[0], c[2])),
        ("CR_MWR", lambda c: (-c[8] - 0.01 * c[5], c[0], c[2])),
        ("EST_LPTMWR", lambda c: (c[0], -c[1] - 0.2 * c[5], c[2])),
        ("MWR_CR", lambda c: (-c[5] + c[8] * 100, c[0], c[2])),
        ("EST_SlackMWR", lambda c: (c[0], c[7] - 0.5 * c[5], c[2])),
    ]
    
    best_makespan = float('inf')
    best_schedules = None
    best_rule = None
    
    for rule_name, priority_func in rules:
        makespan, schedules = _solve_with_priority(durations, machines, priority_func)
        makespan, schedules = _local_search(makespan, schedules, durations, machines)
        if makespan < best_makespan:
            best_makespan = makespan
            best_schedules = schedules
            best_rule = rule_name
    
    return {
        "name": instance["name"],
        "makespan": best_makespan,
        "machine_schedules": best_schedules,
        "solved_by": f"MultiDispatch[{best_rule}]",
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
