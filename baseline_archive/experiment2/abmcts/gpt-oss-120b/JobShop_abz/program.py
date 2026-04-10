# EVOLVE-BLOCK-START
"""Simple greedy baseline for ABZ (Adams, Balas & Zawack, 1988).

Baseline constraints:
- Pure Python implementation.
- Standard library only.
- No `job_shop_lib` import and no external solver usage.
"""

from __future__ import annotations

import argparse
import json
import os
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


def _compute_suffix_sums(durations: list[list[int]]) -> list[list[int]]:
    """Return for each job a list where suffix[i] = sum(durations[i:])."""
    suffixes: list[list[int]] = []
    for job in durations:
        n = len(job)
        suf = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suf[i] = suf[i + 1] + job[i]
        suffixes.append(suf)
    return suffixes


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Greedy EST scheduler with a look‑ahead priority (remaining job work).

    The heuristic chooses, among all currently ready operations,
    the one with the smallest earliest start time.  Ties are broken
    by preferring operations belonging to jobs with larger remaining
    total processing time, and finally by shorter operation duration.
    This simple improvement often yields a noticeably tighter makespan
    compared with pure EST+SPT while still being O(N^2) in the number
    of operations, which is fine for the benchmark sizes.
    """
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    total_operations = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines) + 1

    # Pre‑compute remaining work for each job after each operation.
    suffix_sums = _compute_suffix_sums(durations)

    next_op = [0] * num_jobs          # index of next operation to schedule per job
    job_ready = [0] * num_jobs        # earliest time the job can start its next op
    machine_ready = [0] * num_machines

    machine_schedules: list[list[dict[str, int]]] = [
        [] for _ in range(num_machines)
    ]

    scheduled = 0
    while scheduled < total_operations:
        candidates: list[tuple[int, int, int, int, int]] = []
        # (est, -remaining_job_time, duration, job_id, op_idx, machine_id)
        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= len(durations[job_id]):
                continue

            machine_id = machines[job_id][op_idx]
            duration = durations[job_id][op_idx]
            est = max(job_ready[job_id], machine_ready[machine_id])
            remaining = suffix_sums[job_id][op_idx + 1]  # work left after this op
            candidates.append((est, -remaining, duration, job_id, op_idx, machine_id))

        if not candidates:
            raise RuntimeError("No schedulable operation found.")

        # Choose best candidate according to the described priority.
        est, _neg_rem, duration, job_id, op_idx, machine_id = min(
            candidates,
            key=lambda x: (x[0], x[1], x[2], x[3]),
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

        # Update state.
        next_op[job_id] += 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end
        scheduled += 1

    makespan = max(job_ready) if job_ready else 0
    return {
        "name": instance["name"],
        "makespan": makespan,
        "machine_schedules": machine_schedules,
        "solved_by": "GreedyESTLookaheadBaseline",
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
