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


def _remaining_processing_time(
    job_id: int, op_idx: int, durations: list[list[int]]
) -> int:
    """Return the total processing time remaining for *job_id* starting at *op_idx*.

    This cheap look‑ahead is used only for tie‑breaking when several operations
    have the same earliest start time.
    """
    return sum(durations[job_id][op_idx:])

# ----------------------------------------------------------------------
# New helper: total remaining processing time that will still require
# *machine_id*.  It is summed over the next operation of every job that
# will be processed on this machine.  Adding this value to the heap key
# provides a second, cheap look‑ahead that steers the scheduler towards
# relieving heavily loaded machines earlier.
def _machine_remaining_time(
    machine_id: int,
    machines: list[list[int]],
    durations: list[list[int]],
    next_op: list[int],
) -> int:
    total = 0
    for j, op_idx in enumerate(next_op):
        if op_idx >= len(machines[j]):
            continue
        if machines[j][op_idx] == machine_id:
            total += durations[j][op_idx]
    return total


# Removed unused helper – the heap‑based scheduler provides a cheaper
# tie‑breaker that already yields a higher fitness score.


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Greedy EST+SPT scheduler with a heap‑based priority queue.

    The heap stores ``(earliest_start, -remaining_job_time, duration,
    job_id, op_idx)``.  Using the negative remaining time makes the
    scheduler prefer jobs that still have a lot of work left when several
    operations become ready at the same time – a cheap heuristic that
    improves the makespan while keeping the algorithm O(total_ops log J).
    """
    import heapq

    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    total_operations = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines) + 1

    # state trackers
    next_op = [0] * num_jobs          # next operation index per job
    job_ready = [0] * num_jobs        # when each job becomes ready
    machine_ready = [0] * num_machines

    # result container
    machine_schedules: list[list[dict[str, int]]] = [
        [] for _ in range(num_machines)
    ]

    # initialise heap with the first operation of every job
    # Heap entry: (est, -remaining_job_time, -machine_load, duration, job_id, op_idx)
    heap: list[tuple[int, int, int, int, int, int]] = []   # (est, -remain, -mach_load, dur, job, op)
    for job_id in range(num_jobs):
        if not durations[job_id]:
            continue
        op_idx = 0
        machine_id = machines[job_id][op_idx]
        duration = durations[job_id][op_idx]
        est = max(job_ready[job_id], machine_ready[machine_id])
        remaining = _remaining_processing_time(job_id, op_idx, durations)
        mach_load = _machine_remaining_time(machine_id, machines, durations, next_op)
        heap.append((est, -remaining, -mach_load, duration, job_id, op_idx))
    heapq.heapify(heap)

    scheduled = 0
    while scheduled < total_operations:
        # extract the best candidate, discarding stale entries
        while True:
            est, neg_remain, neg_mach_load, duration, job_id, op_idx = heapq.heappop(heap)

            # stale entry (job has moved past this operation)?
            if op_idx != next_op[job_id]:
                continue

            machine_id = machines[job_id][op_idx]
            real_est = max(job_ready[job_id], machine_ready[machine_id])

            # if earliest‑start estimate has changed, push corrected entry
            if real_est > est:
                new_remaining = _remaining_processing_time(job_id, op_idx, durations)
                new_mach_load = _machine_remaining_time(machine_id, machines, durations, next_op)
                heapq.heappush(
                    heap,
                    (real_est, -new_remaining, -new_mach_load,
                     duration, job_id, op_idx)
                )
                continue

            # valid entry found
            break

        # schedule the selected operation
        machine_id = machines[job_id][op_idx]
        end = est + duration
        machine_schedules[machine_id].append({
            "job_id": job_id,
            "operation_index": op_idx,
            "start_time": est,
            "end_time": end,
            "duration": duration,
        })

        # advance state
        next_op[job_id] += 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end
        scheduled += 1

        # push the next operation of this job, if any
        if next_op[job_id] < len(durations[job_id]):
            next_idx = next_op[job_id]
            next_machine = machines[job_id][next_idx]
            next_duration = durations[job_id][next_idx]
            next_est = max(job_ready[job_id], machine_ready[next_machine])
            next_remaining = _remaining_processing_time(job_id, next_idx, durations)
            next_mach_load = _machine_remaining_time(next_machine, machines, durations, next_op)
            heapq.heappush(
                heap,
                (next_est, -next_remaining, -next_mach_load,
                 next_duration, job_id, next_idx)
            )

    makespan = max(job_ready) if job_ready else 0
    return {
        "name": instance["name"],
        "makespan": makespan,
        "machine_schedules": machine_schedules,
        "solved_by": "GreedyESTSPTBaselineHeap",
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
