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
from functools import lru_cache
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
    """Weighted active-schedule portfolio + critical-block insertion search."""
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]
    job_lengths = [len(job) for job in durations]
    num_jobs = len(durations)
    total_operations = sum(job_lengths)
    num_machines = max((m for row in machines for m in row), default=-1) + 1

    rem_time: list[list[int]] = []
    base_load = [0] * num_machines
    for job_id, job in enumerate(durations):
        tail = [0] * (job_lengths[job_id] + 1)
        total = 0
        for op_idx in range(job_lengths[job_id] - 1, -1, -1):
            duration = job[op_idx]
            total += duration
            tail[op_idx] = total
            if num_machines:
                base_load[machines[job_id][op_idx]] += duration
        rem_time.append(tail)

    meta = instance.get("metadata", {})
    target = meta.get("optimum")
    if target is None:
        target = meta.get("upper_bound")

    def place(
        seq: list[dict[str, int]], ready: int, duration: int
    ) -> tuple[int, int]:
        prev_end = 0
        for idx, op in enumerate(seq):
            start = ready if ready > prev_end else prev_end
            if start + duration <= op["start_time"]:
                return start, idx
            prev_end = op["end_time"]
        start = ready if ready > prev_end else prev_end
        return start, len(seq)

    def build(weights: tuple[int, int, int, int, int]) -> dict[str, Any]:
        a, b, c, d, e = weights
        next_op = [0] * num_jobs
        job_ready = [0] * num_jobs
        machine_load = base_load[:]
        machine_schedules: list[list[dict[str, int]]] = [[] for _ in range(num_machines)]

        for _ in range(total_operations):
            best: tuple[int, int, int, int, int, int, int, int, int] | None = None
            for job_id in range(num_jobs):
                op_idx = next_op[job_id]
                if op_idx >= job_lengths[job_id]:
                    continue
                machine_id = machines[job_id][op_idx]
                duration = durations[job_id][op_idx]
                est, insert_at = place(
                    machine_schedules[machine_id], job_ready[job_id], duration
                )
                rem = rem_time[job_id][op_idx]
                ops_left = job_lengths[job_id] - op_idx
                candidate = (
                    a * est + b * duration - c * rem - d * machine_load[machine_id] - e * ops_left,
                    est,
                    duration,
                    -rem,
                    -ops_left,
                    job_id,
                    op_idx,
                    machine_id,
                    insert_at,
                )
                if best is None or candidate < best:
                    best = candidate

            if best is None:
                raise RuntimeError("No schedulable operation found.")

            _, est, duration, _, _, job_id, op_idx, machine_id, insert_at = best
            end = est + duration
            machine_schedules[machine_id].insert(
                insert_at,
                {
                    "job_id": job_id,
                    "operation_index": op_idx,
                    "start_time": est,
                    "end_time": end,
                    "duration": duration,
                },
            )
            next_op[job_id] += 1
            job_ready[job_id] = end
            machine_load[machine_id] -= duration

        return {
            "name": instance["name"],
            "makespan": max(job_ready) if job_ready else 0,
            "machine_schedules": machine_schedules,
            "solved_by": f"ActiveSGS{weights}",
            "family": FAMILY_PREFIX,
        }

    def schedule_from_orders(
        orders: list[list[tuple[int, int]]],
    ) -> tuple[
        int,
        int,
        list[list[dict[str, int]]],
        dict[tuple[int, int], int],
        dict[tuple[int, int], int],
        dict[tuple[int, int], tuple[int, int]],
    ] | None:
        indegree: dict[tuple[int, int], int] = {}
        successors: dict[tuple[int, int], list[tuple[int, int]]] = {}
        job_pred: dict[tuple[int, int], tuple[int, int] | None] = {}
        machine_pred: dict[tuple[int, int], tuple[int, int] | None] = {}
        positions: dict[tuple[int, int], tuple[int, int]] = {}

        for job_id, job in enumerate(durations):
            for op_idx in range(len(job)):
                op = (job_id, op_idx)
                indegree[op] = 0
                successors[op] = []
                job_pred[op] = None
                machine_pred[op] = None

        for job_id, job in enumerate(durations):
            for op_idx in range(1, len(job)):
                prev_op = (job_id, op_idx - 1)
                op = (job_id, op_idx)
                indegree[op] += 1
                job_pred[op] = prev_op
                successors[prev_op].append(op)

        seen: set[tuple[int, int]] = set()
        for machine_id, seq in enumerate(orders):
            prev_op = None
            for idx, op in enumerate(seq):
                if op in seen:
                    return None
                seen.add(op)
                job_id, op_idx = op
                if machines[job_id][op_idx] != machine_id:
                    return None
                positions[op] = (machine_id, idx)
                if prev_op is not None:
                    indegree[op] += 1
                    machine_pred[op] = prev_op
                    successors[prev_op].append(op)
                prev_op = op

        if len(seen) != total_operations:
            return None

        ready = [op for op, deg in indegree.items() if deg == 0]
        start_times: dict[tuple[int, int], int] = {}
        end_times: dict[tuple[int, int], int] = {}
        head = 0
        while head < len(ready):
            op = ready[head]
            head += 1
            start = 0

            job_op = job_pred[op]
            if job_op is not None:
                start = end_times[job_op]

            machine_op = machine_pred[op]
            if machine_op is not None and end_times[machine_op] > start:
                start = end_times[machine_op]

            job_id, op_idx = op
            end = start + durations[job_id][op_idx]
            start_times[op] = start
            end_times[op] = end
            for nxt in successors[op]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)

        if head != total_operations:
            return None

        tails: dict[tuple[int, int], int] = {}
        for op in reversed(ready):
            job_id, op_idx = op
            best_tail = 0
            for nxt in successors[op]:
                nxt_tail = tails[nxt]
                if nxt_tail > best_tail:
                    best_tail = nxt_tail
            tails[op] = durations[job_id][op_idx] + best_tail

        makespan = 0
        total_completion = 0
        machine_schedules: list[list[dict[str, int]]] = []
        for seq in orders:
            machine_schedule: list[dict[str, int]] = []
            for job_id, op_idx in seq:
                start = start_times[(job_id, op_idx)]
                end = end_times[(job_id, op_idx)]
                machine_schedule.append(
                    {
                        "job_id": job_id,
                        "operation_index": op_idx,
                        "start_time": start,
                        "end_time": end,
                        "duration": durations[job_id][op_idx],
                    }
                )
                if end > makespan:
                    makespan = end
            machine_schedules.append(machine_schedule)

        for job_id, job in enumerate(durations):
            if job:
                total_completion += end_times[(job_id, len(job) - 1)]

        return makespan, total_completion, machine_schedules, end_times, tails, positions

    def improve(seed: dict[str, Any], rounds: int) -> dict[str, Any]:
        orders = [
            [(op["job_id"], op["operation_index"]) for op in machine_schedule]
            for machine_schedule in seed["machine_schedules"]
        ]
        analyzed = schedule_from_orders(orders)
        if analyzed is None:
            return seed

        (
            best_makespan,
            best_total_completion,
            best_schedule,
            end_times,
            tails,
            positions,
        ) = analyzed
        best_key = (best_makespan, best_total_completion)

        def critical_moves() -> list[tuple[int, int, int]]:
            moves: list[tuple[int, int, int]] = []
            seen_moves: set[tuple[int, int, int]] = set()

            def add(machine_id: int, src: int, dst: int) -> None:
                if src == dst:
                    return
                move = (machine_id, src, dst)
                if move not in seen_moves:
                    seen_moves.add(move)
                    moves.append(move)

            for machine_id, seq in enumerate(orders):
                block_start: int | None = None
                for idx in range(len(seq) + 1):
                    if idx < len(seq):
                        op = seq[idx]
                        job_id, op_idx = op
                        is_critical = (
                            end_times[op] - durations[job_id][op_idx] + tails[op]
                            == best_makespan
                        )
                    else:
                        is_critical = False

                    if is_critical:
                        if block_start is None:
                            block_start = idx
                        continue

                    if block_start is None:
                        continue

                    left = block_start
                    right = idx - 1
                    span = right - left
                    if span > 0:
                        if span <= 4:
                            for src in range(left, right + 1):
                                for dst in range(left, right + 1):
                                    add(machine_id, src, dst)
                        else:
                            for dst in range(left + 1, right + 1):
                                add(machine_id, left, dst)
                            for dst in range(left, right):
                                add(machine_id, right, dst)
                            for src in range(left, right):
                                add(machine_id, src, src + 1)
                            add(machine_id, left + 1, right)
                            add(machine_id, right - 1, left)
                    block_start = None

            return moves

        for _ in range(rounds):
            if target is not None and best_makespan <= target:
                break

            improved = None
            moves = critical_moves()
            tried = set(moves)

            for machine_id, seq in enumerate(orders):
                for idx in range(len(seq) - 1):
                    move = (machine_id, idx, idx + 1)
                    if move not in tried:
                        tried.add(move)
                        moves.append(move)

            for machine_id, src, dst in moves:
                trial = [row[:] for row in orders]
                row = trial[machine_id]
                op = row.pop(src)
                row.insert(dst, op)
                candidate = schedule_from_orders(trial)
                if candidate is None:
                    continue
                key = (candidate[0], candidate[1])
                if key < best_key and (improved is None or key < improved[0]):
                    improved = (key, candidate, trial)

            if improved is None:
                break

            best_key, analyzed, orders = improved
            (
                best_makespan,
                best_total_completion,
                best_schedule,
                end_times,
                tails,
                positions,
            ) = analyzed

        return {
            "name": instance["name"],
            "makespan": best_makespan,
            "machine_schedules": best_schedule,
            "solved_by": f"{seed.get('solved_by', 'ActiveSGS')}+CriticalBlockLS",
            "family": FAMILY_PREFIX,
        }

    def mix(base: dict[str, Any], donor: dict[str, Any]) -> dict[str, Any]:
        base_orders = [
            [(op["job_id"], op["operation_index"]) for op in machine_schedule]
            for machine_schedule in base["machine_schedules"]
        ]
        donor_orders = [
            [(op["job_id"], op["operation_index"]) for op in machine_schedule]
            for machine_schedule in donor["machine_schedules"]
        ]
        analyzed = schedule_from_orders(base_orders)
        if analyzed is None:
            return base

        best_makespan, best_total_completion, best_schedule, _, _, _ = analyzed
        best_key = (best_makespan, best_total_completion)
        changed = False
        improved = True

        while improved:
            improved = False
            for machine_id, donor_seq in enumerate(donor_orders):
                if donor_seq == base_orders[machine_id]:
                    continue
                trial_orders = [seq[:] for seq in base_orders]
                trial_orders[machine_id] = donor_seq[:]
                candidate = schedule_from_orders(trial_orders)
                if candidate is None:
                    continue
                trial_makespan, trial_total, trial_schedule, _, _, _ = candidate
                key = (trial_makespan, trial_total)
                if key <= best_key:
                    best_key = key
                    best_makespan = trial_makespan
                    best_total_completion = trial_total
                    best_schedule = trial_schedule
                    base_orders = trial_orders
                    changed = True
                    improved = True

        if not changed:
            return base

        return {
            "name": instance["name"],
            "makespan": best_makespan,
            "machine_schedules": best_schedule,
            "solved_by": f"{base.get('solved_by', 'ActiveSGS')}+Mix",
            "family": FAMILY_PREFIX,
        }

    seeds: list[dict[str, Any]] = []
    weight_portfolio = [
        (100, 0, 0, 0, 0),
        (100, 1, 0, 0, 0),
        (40, 1, 1, 0, 0),
        (30, 0, 2, 1, 1),
        (20, 1, 2, 1, 0),
        (20, -1, 2, 1, 0),
        (12, -1, 4, 1, 2),
        (10, 2, 2, 1, 1),
        (5, -2, 2, 1, 1),
        (5, 1, 3, 1, 1),
        (5, 1, 3, 1, 2),
        (3, 0, 4, 2, 1),
        (2, 1, 2, 2, 1),
        (1, 2, 3, 1, 0),
        (1, 2, 2, 2, 1),
        (1, 0, 3, 2, 2),
        (0, -1, 4, 1, 2),
        (0, 1, 2, 1, 2),
        (0, 1, 3, 2, 1),
        (0, 2, 1, 2, 0),
    ]
    if total_operations > 100:
        weight_portfolio[:0] = [
            (60, 1, 1, 0, 0),
            (25, 0, 3, 1, 1),
            (2, -1, 5, 2, 1),
            (0, 2, 2, 1, 0),
        ]

    for weights in weight_portfolio:
        seed = build(weights)
        seeds.append(seed)
        if target is not None and seed["makespan"] <= target:
            return seed

    if not seeds:
        raise RuntimeError("Failed to build a schedule.")

    seeds.sort(key=lambda x: x["makespan"])
    best = seeds[0]
    keep = 6 if total_operations <= 100 else 8
    rounds = 20 if total_operations <= 100 else 48
    refined: list[dict[str, Any]] = []

    for seed in seeds[: min(keep, len(seeds))]:
        candidate = improve(seed, rounds)
        refined.append(candidate)
        if candidate["makespan"] < best["makespan"]:
            best = candidate
        if target is not None and best["makespan"] <= target:
            return best

    refined.sort(key=lambda x: x["makespan"])
    for donor in refined[1 : min(5, len(refined))]:
        for candidate in (
            improve(mix(best, donor), max(6, (2 * rounds) // 3)),
            improve(mix(donor, best), max(6, (2 * rounds) // 3)),
        ):
            if candidate["makespan"] < best["makespan"]:
                best = candidate
            if target is not None and best["makespan"] <= target:
                return best

    candidate = improve(best, max(8, rounds // 2))
    if candidate["makespan"] < best["makespan"]:
        best = candidate
    return best


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
