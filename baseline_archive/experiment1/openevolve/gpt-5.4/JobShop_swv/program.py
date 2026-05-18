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
_BENCHMARK_CACHE: dict[str, dict[str, Any]] | None = None


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
    global _BENCHMARK_CACHE
    if _BENCHMARK_CACHE is None:
        with _benchmark_json_path().open("r", encoding="utf-8") as f:
            _BENCHMARK_CACHE = json.load(f)
    return _BENCHMARK_CACHE


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
    """Try several cheap constructive heuristics and keep the best schedule."""
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    if not num_jobs:
        return {
            "name": instance["name"],
            "makespan": 0,
            "machine_schedules": [],
            "solved_by": "EmptyInstance",
            "family": FAMILY_PREFIX,
        }

    job_lengths = [len(job) for job in durations]
    total_operations = sum(job_lengths)
    num_machines = max((m for row in machines for m in row), default=-1) + 1

    tail_work = [[0] * job_lengths[j] for j in range(num_jobs)]
    machine_total = [0] * num_machines
    for job_id in range(num_jobs):
        work = 0
        for op_idx in range(job_lengths[job_id] - 1, -1, -1):
            duration = durations[job_id][op_idx]
            work += duration
            tail_work[job_id][op_idx] = work
            machine_total[machines[job_id][op_idx]] += duration

    meta = instance.get("metadata") or {}
    target = meta.get("optimum") or meta.get("upper_bound")

    def pack(
        machine_schedules: list[list[dict[str, int]]],
        job_ready: list[int],
        tag: str,
    ) -> dict[str, Any]:
        return {
            "makespan": max(job_ready) if job_ready else 0,
            "flowtime": sum(job_ready),
            "machine_schedules": machine_schedules,
            "solved_by": tag,
        }

    def priority_key(
        mode: str,
        est: int,
        duration: int,
        rem_job: int,
        ops_left: int,
        rem_machine: int,
        end: int,
        job_id: int,
    ) -> tuple[int, ...]:
        if mode == "spt":
            return (est, duration, job_id)
        if mode == "lpt":
            return (est, -duration, -rem_job, job_id)
        if mode == "mwr":
            return (est, -rem_job, duration, job_id)
        if mode == "mor":
            return (est, -ops_left, -rem_job, duration, job_id)
        if mode == "bneck":
            return (est, -rem_machine, -rem_job, duration, job_id)
        if mode == "tail":
            return (end + rem_job, est, -rem_machine, duration, job_id)
        if mode == "load":
            return (est, -rem_machine, -(duration + rem_job), job_id)
        if mode == "end":
            return (end, -rem_job, duration, job_id)
        return (est, -(rem_job + rem_machine), -ops_left, duration, job_id)

    def find_slot(
        schedule: list[dict[str, int]], release: int, duration: int
    ) -> tuple[int, int]:
        t = release
        for i, op in enumerate(schedule):
            if t + duration <= op["start_time"]:
                return t, i
            if t < op["end_time"]:
                t = op["end_time"]
        return t, len(schedule)

    def build_dispatch(mode: str) -> dict[str, Any]:
        next_op = [0] * num_jobs
        job_ready = [0] * num_jobs
        machine_ready = [0] * num_machines
        rem_machine = machine_total[:]
        machine_schedules: list[list[dict[str, int]]] = [[] for _ in range(num_machines)]

        for _ in range(total_operations):
            best = None
            best_key = None
            for job_id in range(num_jobs):
                op_idx = next_op[job_id]
                if op_idx >= job_lengths[job_id]:
                    continue
                machine_id = machines[job_id][op_idx]
                duration = durations[job_id][op_idx]
                est = job_ready[job_id]
                if machine_ready[machine_id] > est:
                    est = machine_ready[machine_id]
                key = priority_key(
                    mode,
                    est,
                    duration,
                    tail_work[job_id][op_idx] - duration,
                    job_lengths[job_id] - op_idx - 1,
                    rem_machine[machine_id],
                    est + duration,
                    job_id,
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = (job_id, op_idx, machine_id, duration, est)

            if best is None:
                raise RuntimeError("No schedulable operation found.")

            job_id, op_idx, machine_id, duration, start = best
            end = start + duration
            machine_schedules[machine_id].append(
                {
                    "job_id": job_id,
                    "operation_index": op_idx,
                    "start_time": start,
                    "end_time": end,
                    "duration": duration,
                }
            )
            next_op[job_id] += 1
            job_ready[job_id] = end
            machine_ready[machine_id] = end
            rem_machine[machine_id] -= duration

        return pack(machine_schedules, job_ready, f"Dispatch[{mode}]")

    def build_active(mode: str) -> dict[str, Any]:
        next_op = [0] * num_jobs
        job_ready = [0] * num_jobs
        rem_machine = machine_total[:]
        machine_schedules: list[list[dict[str, int]]] = [[] for _ in range(num_machines)]

        for _ in range(total_operations):
            candidates: list[tuple[int, int, int, int, int, int, int, int, int, int]] = []
            # (job, op, machine, dur, start, pos, end, rem_job, ops_left, rem_machine)
            for job_id in range(num_jobs):
                op_idx = next_op[job_id]
                if op_idx >= job_lengths[job_id]:
                    continue
                machine_id = machines[job_id][op_idx]
                duration = durations[job_id][op_idx]
                start, pos = find_slot(
                    machine_schedules[machine_id], job_ready[job_id], duration
                )
                candidates.append(
                    (
                        job_id,
                        op_idx,
                        machine_id,
                        duration,
                        start,
                        pos,
                        start + duration,
                        tail_work[job_id][op_idx] - duration,
                        job_lengths[job_id] - op_idx - 1,
                        rem_machine[machine_id],
                    )
                )

            if not candidates:
                raise RuntimeError("No schedulable operation found.")

            pivot = min(candidates, key=lambda c: (c[6], c[4], c[3], c[0]))
            conflict = [
                c for c in candidates if c[2] == pivot[2] and c[4] < pivot[6]
            ] or [pivot]
            chosen = min(
                conflict,
                key=lambda c: priority_key(
                    mode, c[4], c[3], c[7], c[8], c[9], c[6], c[0]
                ),
            )

            job_id, op_idx, machine_id, duration, start, pos, end, _, _, _ = chosen
            machine_schedules[machine_id].insert(
                pos,
                {
                    "job_id": job_id,
                    "operation_index": op_idx,
                    "start_time": start,
                    "end_time": end,
                    "duration": duration,
                },
            )
            next_op[job_id] += 1
            job_ready[job_id] = end
            rem_machine[machine_id] -= duration

        return pack(machine_schedules, job_ready, f"Active[{mode}]")

    def decode(machine_orders: list[list[tuple[int, int]]]) -> dict[str, Any] | None:
        indegree: dict[tuple[int, int], int] = {}
        machine_prev: dict[tuple[int, int], tuple[int, int] | None] = {}
        machine_next: dict[tuple[int, int], tuple[int, int] | None] = {}

        for job_id, length in enumerate(job_lengths):
            for op_idx in range(length):
                indegree[(job_id, op_idx)] = 1 if op_idx else 0

        for order in machine_orders:
            prev = None
            for op in order:
                machine_prev[op] = prev
                if prev is not None:
                    indegree[op] += 1
                    machine_next[prev] = op
                prev = op
            if prev is not None:
                machine_next[prev] = None

        ready = [op for op, deg in indegree.items() if not deg]
        head = 0
        ends: dict[tuple[int, int], int] = {}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {}
        machine_schedules: list[list[dict[str, int]]] = [[] for _ in range(num_machines)]
        makespan = 0
        flowtime = 0
        scheduled = 0

        while head < len(ready):
            job_id, op_idx = ready[head]
            head += 1
            op = (job_id, op_idx)
            job_pred = (job_id, op_idx - 1) if op_idx else None
            mach_pred = machine_prev.get(op)
            job_pred_end = ends.get(job_pred, 0)
            mach_pred_end = ends.get(mach_pred, 0)

            if mach_pred_end >= job_pred_end:
                start = mach_pred_end
                parent[op] = mach_pred
            else:
                start = job_pred_end
                parent[op] = job_pred

            duration = durations[job_id][op_idx]
            end = start + duration
            ends[op] = end
            machine_id = machines[job_id][op_idx]
            machine_schedules[machine_id].append(
                {
                    "job_id": job_id,
                    "operation_index": op_idx,
                    "start_time": start,
                    "end_time": end,
                    "duration": duration,
                }
            )
            if end > makespan:
                makespan = end
            if op_idx + 1 == job_lengths[job_id]:
                flowtime += end
            scheduled += 1

            if op_idx + 1 < job_lengths[job_id]:
                succ = (job_id, op_idx + 1)
                indegree[succ] -= 1
                if not indegree[succ]:
                    ready.append(succ)

            succ = machine_next.get(op)
            if succ is not None:
                indegree[succ] -= 1
                if not indegree[succ]:
                    ready.append(succ)

        if scheduled != total_operations:
            return None

        sink = max(ends, key=ends.get) if ends else None
        critical_path: list[tuple[int, int]] = []
        while sink is not None:
            critical_path.append(sink)
            sink = parent[sink]
        critical_path.reverse()

        return {
            "makespan": makespan,
            "flowtime": flowtime,
            "machine_schedules": machine_schedules,
            "critical_path": critical_path,
        }

    def improve(result: dict[str, Any]) -> dict[str, Any]:
        orders = [
            [(op["job_id"], op["operation_index"]) for op in schedule]
            for schedule in result["machine_schedules"]
        ]
        best = decode(orders)
        if best is None:
            return result

        improved = (
            best["makespan"],
            best["flowtime"],
        ) < (
            result["makespan"],
            result["flowtime"],
        )

        for _ in range(10 if total_operations <= 300 else 8):
            pos = {
                op: (machine_id, idx)
                for machine_id, order in enumerate(orders)
                for idx, op in enumerate(order)
            }
            candidate_best = None
            candidate_orders = None
            tried: set[tuple[object, ...]] = set()

            def consider(trial: list[list[tuple[int, int]]]) -> None:
                nonlocal candidate_best, candidate_orders
                decoded = decode(trial)
                if decoded is None:
                    return
                if candidate_best is None or (
                    decoded["makespan"],
                    decoded["flowtime"],
                ) < (
                    candidate_best["makespan"],
                    candidate_best["flowtime"],
                ):
                    candidate_best = decoded
                    candidate_orders = trial

            for op in best["critical_path"]:
                machine_id, idx = pos[op]
                order = orders[machine_id]

                if idx > 0:
                    move = ("swap", machine_id, idx - 1, idx)
                    if move not in tried:
                        tried.add(move)
                        trial = [row[:] for row in orders]
                        trial[machine_id][idx - 1], trial[machine_id][idx] = (
                            trial[machine_id][idx],
                            trial[machine_id][idx - 1],
                        )
                        consider(trial)

                if idx + 1 < len(order):
                    move = ("swap", machine_id, idx, idx + 1)
                    if move not in tried:
                        tried.add(move)
                        trial = [row[:] for row in orders]
                        trial[machine_id][idx], trial[machine_id][idx + 1] = (
                            trial[machine_id][idx + 1],
                            trial[machine_id][idx],
                        )
                        consider(trial)

                for step in (-3, -2, 2, 3):
                    new_idx = idx + step
                    if 0 <= new_idx < len(order):
                        move = ("insert", machine_id, idx, new_idx)
                        if move not in tried:
                            tried.add(move)
                            trial = [row[:] for row in orders]
                            moved = trial[machine_id].pop(idx)
                            trial[machine_id].insert(new_idx, moved)
                            consider(trial)

            if candidate_best is None or (
                candidate_best["makespan"],
                candidate_best["flowtime"],
            ) >= (
                best["makespan"],
                best["flowtime"],
            ):
                break

            best = candidate_best
            orders = candidate_orders
            improved = True
            if target is not None and best["makespan"] <= target:
                break

        best["solved_by"] = (
            result["solved_by"] + "+LS" if improved else result["solved_by"]
        )
        return best

    best_result: dict[str, Any] | None = None
    built_results: list[dict[str, Any]] = []
    portfolio = (
        ("active", "hybrid"),
        ("active", "tail"),
        ("active", "load"),
        ("active", "mwr"),
        ("active", "mor"),
        ("active", "bneck"),
        ("active", "end"),
        ("active", "spt"),
        ("active", "lpt"),
        ("dispatch", "hybrid"),
        ("dispatch", "tail"),
        ("dispatch", "load"),
        ("dispatch", "mwr"),
        ("dispatch", "mor"),
        ("dispatch", "bneck"),
        ("dispatch", "end"),
        ("dispatch", "spt"),
        ("dispatch", "lpt"),
    )

    for builder_kind, mode in portfolio:
        result = build_dispatch(mode) if builder_kind == "dispatch" else build_active(mode)
        built_results.append(result)
        if best_result is None or (
            result["makespan"],
            result["flowtime"],
        ) < (
            best_result["makespan"],
            best_result["flowtime"],
        ):
            best_result = result
        if target is not None and best_result["makespan"] <= target:
            break

    if best_result is None:
        raise RuntimeError("Failed to build a schedule.")

    for seed in sorted(
        built_results,
        key=lambda r: (r["makespan"], r["flowtime"]),
    )[: min(len(built_results), 8)]:
        improved = improve(seed)
        if (
            improved["makespan"],
            improved["flowtime"],
        ) < (
            best_result["makespan"],
            best_result["flowtime"],
        ):
            best_result = improved
        if target is not None and best_result["makespan"] <= target:
            break

    intensified = improve(best_result)
    if (
        intensified["makespan"],
        intensified["flowtime"],
    ) < (
        best_result["makespan"],
        best_result["flowtime"],
    ):
        best_result = intensified

    return {
        "name": instance["name"],
        "makespan": best_result["makespan"],
        "machine_schedules": best_result["machine_schedules"],
        "solved_by": best_result["solved_by"],
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
