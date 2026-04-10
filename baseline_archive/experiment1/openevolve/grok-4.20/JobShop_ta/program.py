# EVOLVE-BLOCK-START
"""Multi-rule greedy EST scheduler for TA (Taillard, 1993).

Tries SPT/LPT/MWKR/LWKR/MOPNR/LOPNR/FIFO dispatching rules (EST primary)
and returns the schedule with the smallest makespan. Pure Python,
standard library only.
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


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Multi-rule greedy EST scheduler; returns best-makespan schedule.

    Tries 7 dispatching rules (EST primary) and keeps the schedule with
    the smallest makespan. This explores a broader feature space while
    remaining pure-Python.

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

    best_makespan = None
    best_machine_schedules = None
    best_rule = ""

    for rule, get_key in [
        ("SPT", lambda est, dur, job, rem, rops: (est, dur, job)),
        ("LPT", lambda est, dur, job, rem, rops: (est, -dur, job)),
        ("MWKR", lambda est, dur, job, rem, rops: (est, -rem, job)),
        ("LWKR", lambda est, dur, job, rem, rops: (est, rem, job)),
        ("MOPNR", lambda est, dur, job, rem, rops: (est, -rops, job)),
        ("LOPNR", lambda est, dur, job, rem, rops: (est, rops, job)),
        ("FIFO", lambda est, dur, job, rem, rops: (est, job, 0)),
    ]:
        next_op = [0] * num_jobs
        job_ready = [0] * num_jobs
        machine_ready = [0] * num_machines
        rem_work = [sum(job) for job in durations]

        machine_schedules: list[list[dict[str, int]]] = [
            [] for _ in range(num_machines)
        ]

        scheduled = 0
        while scheduled < total_operations:
            candidates: list[tuple] = []  # (sort_key, est, job_id, op_idx, mach_id, dur)
            for job_id in range(num_jobs):
                op_idx = next_op[job_id]
                if op_idx >= len(durations[job_id]):
                    continue

                machine_id = machines[job_id][op_idx]
                duration = durations[job_id][op_idx]
                est = max(job_ready[job_id], machine_ready[machine_id])
                rem = rem_work[job_id]
                rops = len(durations[job_id]) - op_idx
                sort_key = get_key(est, duration, job_id, rem, rops)
                candidates.append((sort_key, est, job_id, op_idx, machine_id, duration))

            if not candidates:
                raise RuntimeError("No schedulable operation found.")

            selected = min(candidates, key=lambda x: x[0])
            _, est, job_id, op_idx, machine_id, duration = selected
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
            rem_work[job_id] -= duration
            scheduled += 1

        makespan = max(job_ready) if job_ready else 0
        if best_makespan is None or makespan < best_makespan:
            best_makespan = makespan
            best_machine_schedules = machine_schedules
            best_rule = rule

    return {
        "name": instance["name"],
        "makespan": best_makespan,
        "machine_schedules": best_machine_schedules,
        "solved_by": f"GreedyMultiEST{best_rule}Baseline",
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
