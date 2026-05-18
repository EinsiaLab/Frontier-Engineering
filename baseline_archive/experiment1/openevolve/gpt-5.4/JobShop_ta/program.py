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
from functools import lru_cache
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


@lru_cache(maxsize=1)
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
    """Best-of-rules active scheduler with light critical-block improvement."""
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    job_lengths = [len(job) for job in durations]
    total_operations = sum(job_lengths)
    num_machines = max((max(row) for row in machines if row), default=-1) + 1
    if total_operations == 0:
        return {
            "name": instance["name"],
            "makespan": 0,
            "machine_schedules": [[] for _ in range(num_machines)],
            "solved_by": "ActiveScheduleGT/empty",
            "family": FAMILY_PREFIX,
        }

    job_totals = [sum(job) for job in durations]
    op_ids: list[list[int]] = []
    job_of: list[int] = []
    op_of: list[int] = []
    machine_of: list[int] = []
    duration_of: list[int] = []
    machine_totals = [0] * num_machines
    for job_id, job in enumerate(durations):
        row: list[int] = []
        for op_idx, duration in enumerate(job):
            op_id = len(job_of)
            row.append(op_id)
            machine_id = machines[job_id][op_idx]
            job_of.append(job_id)
            op_of.append(op_idx)
            machine_of.append(machine_id)
            duration_of.append(duration)
            machine_totals[machine_id] += duration
        op_ids.append(row)

    tail_work = [0] * total_operations
    tail_ops = [0] * total_operations
    job_successors: list[list[int]] = [[] for _ in range(total_operations)]
    job_indegree = [0] * total_operations
    for row in op_ids:
        work = ops = 0
        for idx in range(len(row) - 1, -1, -1):
            op_id = row[idx]
            work += duration_of[op_id]
            ops += 1
            tail_work[op_id] = work
            tail_ops[op_id] = ops
            if idx:
                prev_op = row[idx - 1]
                job_successors[prev_op].append(op_id)
                job_indegree[op_id] += 1

    def earliest_slot(
        schedule: list[tuple[int, int, int, int]],
        ready: int,
        duration: int,
    ) -> tuple[int, int]:
        prev_end = 0
        for idx, (start, end, _, _) in enumerate(schedule):
            slot = ready if ready > prev_end else prev_end
            if slot + duration <= start:
                return slot, idx
            prev_end = end
        return (ready if ready > prev_end else prev_end), len(schedule)

    def build(
        selector: Any,
        rule_name: str,
        pivot_mode: str = "ect",
    ) -> tuple[int, list[list[tuple[int, int, int, int]]], str]:
        next_op = [0] * num_jobs
        job_ready = [0] * num_jobs
        remain_job = job_totals[:]
        remain_machine = machine_totals[:]
        machine_schedules: list[list[tuple[int, int, int, int]]] = [
            [] for _ in range(num_machines)
        ]

        for _ in range(total_operations):
            candidates: list[tuple[int, int, int, int, int, int, int, int]] = []
            for job_id in range(num_jobs):
                op_idx = next_op[job_id]
                if op_idx >= job_lengths[job_id]:
                    continue
                op_id = op_ids[job_id][op_idx]
                machine_id = machine_of[op_id]
                duration = duration_of[op_id]
                start, insert_at = earliest_slot(
                    machine_schedules[machine_id],
                    job_ready[job_id],
                    duration,
                )
                end = start + duration
                candidates.append(
                    (
                        op_id,
                        machine_id,
                        duration,
                        start,
                        end,
                        insert_at,
                        remain_job[job_id],
                        remain_machine[machine_id],
                    )
                )

            if pivot_mode == "global":
                op_id, machine_id, duration, start, end, insert_at, _, _ = min(
                    candidates,
                    key=selector,
                )
            else:
                if pivot_mode == "est":
                    pivot = min(
                        candidates,
                        key=lambda c: (c[3], c[4], c[2], job_of[c[0]]),
                    )
                else:
                    pivot = min(
                        candidates,
                        key=lambda c: (c[4], c[3], c[2], job_of[c[0]]),
                    )
                conflict = [
                    c for c in candidates if c[1] == pivot[1] and c[3] < pivot[4]
                ]
                op_id, machine_id, duration, start, end, insert_at, _, _ = min(
                    conflict or [pivot],
                    key=selector,
                )

            machine_schedules[machine_id].insert(
                insert_at, (start, end, op_id, duration)
            )
            job_id = job_of[op_id]
            next_op[job_id] += 1
            job_ready[job_id] = end
            remain_job[job_id] -= duration
            remain_machine[machine_id] -= duration

        return (max(job_ready) if job_ready else 0), machine_schedules, rule_name

    rules = [
        ("ect/mwkr", "ect", lambda c: (-tail_work[c[0]], -tail_ops[c[0]], c[2], c[3], job_of[c[0]])),
        ("ect/mopnr", "ect", lambda c: (-tail_ops[c[0]], -tail_work[c[0]], c[2], c[3], job_of[c[0]])),
        ("ect/bottleneck", "ect", lambda c: (-c[7], -tail_work[c[0]], c[2], c[3], job_of[c[0]])),
        ("ect/jobload", "ect", lambda c: (-c[6], -tail_work[c[0]], c[2], c[3], job_of[c[0]])),
        ("ect/spt", "ect", lambda c: (c[2], c[3], -tail_work[c[0]], job_of[c[0]])),
        ("ect/ect", "ect", lambda c: (c[4], c[3], c[2], job_of[c[0]])),
        ("ect/critical", "ect", lambda c: (-(c[3] + tail_work[c[0]]), -c[7], c[2], job_of[c[0]])),
        ("est/critical", "est", lambda c: (-(c[3] + tail_work[c[0]]), -tail_ops[c[0]], c[2], job_of[c[0]])),
        ("est/mwkr", "est", lambda c: (-tail_work[c[0]], -tail_ops[c[0]], c[2], c[3], job_of[c[0]])),
        ("est/mopnr", "est", lambda c: (-tail_ops[c[0]], -c[6], c[2], c[3], job_of[c[0]])),
        ("est/jobload", "est", lambda c: (-c[6], -tail_work[c[0]], c[2], c[3], job_of[c[0]])),
        ("global/critical", "global", lambda c: (-(c[3] + tail_work[c[0]]), c[4], c[2], job_of[c[0]])),
        ("global/mwkr", "global", lambda c: (c[3], -tail_work[c[0]], -tail_ops[c[0]], c[2], job_of[c[0]])),
        ("global/jobload", "global", lambda c: (c[3], -c[6], -tail_work[c[0]], c[2], job_of[c[0]])),
        ("global/bottleneck", "global", lambda c: (c[3], -c[7], -tail_work[c[0]], c[2], job_of[c[0]])),
    ]
    if total_operations > 2200:
        rules = rules[:12]
    elif total_operations > 1800:
        rules = rules[:14]

    def schedule_from_orders(
        orders: list[list[int]],
    ) -> tuple[int, int, list[int], list[int], int] | None:
        indegree = job_indegree[:]
        successors = [succ[:] for succ in job_successors]

        for order in orders:
            for idx in range(1, len(order)):
                prev_op = order[idx - 1]
                op_id = order[idx]
                successors[prev_op].append(op_id)
                indegree[op_id] += 1

        ready = [i for i, degree in enumerate(indegree) if degree == 0]
        head = 0
        starts = [0] * total_operations
        critical_pred = [-1] * total_operations
        seen = 0

        while head < len(ready):
            op_id = ready[head]
            head += 1
            seen += 1
            finish = starts[op_id] + duration_of[op_id]
            for succ in successors[op_id]:
                if finish > starts[succ]:
                    starts[succ] = finish
                    critical_pred[succ] = op_id
                elif finish == starts[succ] and (
                    critical_pred[succ] == -1
                    or machine_of[op_id] == machine_of[succ]
                ):
                    critical_pred[succ] = op_id
                indegree[succ] -= 1
                if indegree[succ] == 0:
                    ready.append(succ)

        if seen != total_operations:
            return None

        end_op = 0
        best_end = total_completion = 0
        for op_id, start in enumerate(starts):
            end = start + duration_of[op_id]
            total_completion += end
            if end > best_end:
                best_end = end
                end_op = op_id
        return best_end, total_completion, starts, critical_pred, end_op

    def critical_blocks(critical_pred: list[int], end_op: int) -> list[list[int]]:
        path: list[int] = []
        while end_op != -1:
            path.append(end_op)
            end_op = critical_pred[end_op]
        path.reverse()

        blocks: list[list[int]] = []
        i = 0
        while i < len(path):
            j = i + 1
            machine_id = machine_of[path[i]]
            while j < len(path) and machine_of[path[j]] == machine_id:
                j += 1
            if j - i > 1:
                blocks.append(path[i:j])
            i = j
        return blocks

    def improve(
        machine_orders: list[list[int]],
    ) -> tuple[tuple[int, list[int], list[int], int] | None, list[list[int]], bool]:
        current = schedule_from_orders(machine_orders)
        improved = False
        max_rounds = 14 if total_operations < 400 else 8 if total_operations < 1200 else 5
        insert_span = 4 if total_operations < 700 else 3 if total_operations < 1400 else 2

        for _ in range(max_rounds):
            if current is None:
                break

            current_makespan, current_flow, _, critical_pred, end_op = current
            positions = [0] * total_operations
            for order in machine_orders:
                for idx, op_id in enumerate(order):
                    positions[op_id] = idx

            best_neighbor = None
            seen_moves: set[tuple[object, ...]] = set()
            for block in critical_blocks(critical_pred, end_op):
                if len(block) < 2:
                    continue

                machine_id = machine_of[block[0]]
                block_pos = [positions[op_id] for op_id in block]
                pair_ids = (
                    range(len(block) - 1)
                    if total_operations < 1500 or len(block) <= 6
                    else (0, 1, len(block) - 3, len(block) - 2)
                )

                for idx in pair_ids:
                    pos_a = block_pos[idx]
                    pos_b = block_pos[idx + 1]
                    if pos_b != pos_a + 1:
                        continue
                    move = ("swap", machine_id, pos_a)
                    if move in seen_moves:
                        continue
                    seen_moves.add(move)

                    new_order = machine_orders[machine_id][:]
                    new_order[pos_a], new_order[pos_b] = new_order[pos_b], new_order[pos_a]
                    neighbor_orders = machine_orders[:]
                    neighbor_orders[machine_id] = new_order
                    candidate = schedule_from_orders(neighbor_orders)
                    if candidate is None or candidate[:2] >= (current_makespan, current_flow):
                        continue
                    if best_neighbor is None or candidate[:2] < best_neighbor[0][:2]:
                        best_neighbor = (candidate, neighbor_orders)

                if len(block) > 2 and insert_span:
                    first_pos = block_pos[0]
                    last_pos = block_pos[-1]

                    for idx in range(2, min(len(block), 2 + insert_span)):
                        target_pos = block_pos[idx]
                        if target_pos <= first_pos + 1:
                            continue
                        move = ("right", machine_id, first_pos, target_pos)
                        if move in seen_moves:
                            continue
                        seen_moves.add(move)

                        new_order = machine_orders[machine_id][:]
                        op_id = new_order.pop(first_pos)
                        new_order.insert(target_pos, op_id)
                        neighbor_orders = machine_orders[:]
                        neighbor_orders[machine_id] = new_order
                        candidate = schedule_from_orders(neighbor_orders)
                        if candidate is None or candidate[:2] >= (current_makespan, current_flow):
                            continue
                        if best_neighbor is None or candidate[:2] < best_neighbor[0][:2]:
                            best_neighbor = (candidate, neighbor_orders)

                    for idx in range(max(0, len(block) - 2 - insert_span), len(block) - 2):
                        target_pos = block_pos[idx]
                        if last_pos <= target_pos + 1:
                            continue
                        move = ("left", machine_id, last_pos, target_pos)
                        if move in seen_moves:
                            continue
                        seen_moves.add(move)

                        new_order = machine_orders[machine_id][:]
                        op_id = new_order.pop(last_pos)
                        new_order.insert(target_pos, op_id)
                        neighbor_orders = machine_orders[:]
                        neighbor_orders[machine_id] = new_order
                        candidate = schedule_from_orders(neighbor_orders)
                        if candidate is None or candidate[:2] >= (current_makespan, current_flow):
                            continue
                        if best_neighbor is None or candidate[:2] < best_neighbor[0][:2]:
                            best_neighbor = (candidate, neighbor_orders)

            if best_neighbor is None:
                break
            current, machine_orders = best_neighbor
            improved = True

        return current, machine_orders, improved

    built = [build(selector, rule_name, pivot_mode) for rule_name, pivot_mode, selector in rules]
    built.sort(key=lambda x: x[0])

    chosen = built[: (7 if total_operations < 500 else 5 if total_operations < 1400 else 4)]
    seen_rules = {rule_name for _, _, rule_name in chosen}
    seen_modes = {rule_name.split("/", 1)[0] for _, _, rule_name in chosen}
    for mode in ("ect", "est", "global"):
        if mode in seen_modes:
            continue
        for item in built:
            if item[2].startswith(mode + "/") and item[2] not in seen_rules:
                chosen.append(item)
                seen_rules.add(item[2])
                break

    best = None
    for initial_makespan, raw_schedules, rule_name in chosen:
        machine_orders = [[op_id for _, _, op_id, _ in machine] for machine in raw_schedules]
        current, machine_orders, changed = improve(machine_orders)
        final_makespan = initial_makespan if current is None else current[0]
        candidate = (
            final_makespan,
            initial_makespan,
            current,
            machine_orders,
            raw_schedules,
            rule_name,
            changed,
        )
        if best is None or candidate[:2] < best[:2]:
            best = candidate

    makespan, _, current, machine_orders, raw_schedules, rule_name, improved = best

    if current is not None:
        makespan, _, starts, _, _ = current
        machine_schedules = [
            [
                {
                    "job_id": job_of[op_id],
                    "operation_index": op_of[op_id],
                    "start_time": starts[op_id],
                    "end_time": starts[op_id] + duration_of[op_id],
                    "duration": duration_of[op_id],
                }
                for op_id in order
            ]
            for order in machine_orders
        ]
    else:
        machine_schedules = [
            [
                {
                    "job_id": job_of[op_id],
                    "operation_index": op_of[op_id],
                    "start_time": start,
                    "end_time": end,
                    "duration": duration,
                }
                for start, end, op_id, duration in machine
            ]
            for machine in raw_schedules
        ]

    return {
        "name": instance["name"],
        "makespan": makespan,
        "machine_schedules": machine_schedules,
        "solved_by": f"ActiveScheduleGT/{rule_name}{'+cb' if improved else ''}",
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
