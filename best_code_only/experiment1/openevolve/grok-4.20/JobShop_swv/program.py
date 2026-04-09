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
import random
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
    """Multi-rule + stochastic greedy scheduler with precomputed
    remaining work (MWKR/LWKR). Runs complementary dispatching rules
    then many random job-order trials for better exploration.
    """
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]

    num_jobs = len(durations)
    total_operations = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines) + 1

    # Precompute total remaining work from each operation (incl. current)
    remaining: list[list[int]] = []
    for durs in durations:
        rem = [0] * len(durs)
        total = 0
        for i in range(len(durs) - 1, -1, -1):
            total += durs[i]
            rem[i] = total
        remaining.append(rem)

    def build_schedule(priority_key):
        next_op = [0] * num_jobs
        job_ready = [0] * num_jobs
        machine_ready = [0] * num_machines
        machine_schedules: list[list[dict[str, int]]] = [
            [] for _ in range(num_machines)
        ]
        scheduled = 0
        while scheduled < total_operations:
            candidates = []
            for job_id in range(num_jobs):
                op_idx = next_op[job_id]
                if op_idx >= len(durations[job_id]):
                    continue
                machine_id = machines[job_id][op_idx]
                duration = durations[job_id][op_idx]
                est = max(job_ready[job_id], machine_ready[machine_id])
                rem = remaining[job_id][op_idx]
                candidates.append(
                    (est, duration, job_id, op_idx, machine_id, rem)
                )
            if not candidates:
                raise RuntimeError("No schedulable operation found.")
            chosen = min(candidates, key=priority_key)
            est, duration, job_id, op_idx, machine_id, _ = chosen
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
            scheduled += 1
        makespan = max(job_ready) if job_ready else 0
        return makespan, machine_schedules

    # Five complementary dispatching rules; best-of-N selected per instance
    rules = [
        lambda x: (x[0], x[1], x[2]),           # EST + SPT
        lambda x: (x[0], -x[1], x[2]),          # EST + LPT
        lambda x: (x[0], -x[5], x[2]),          # EST + MWKR
        lambda x: (x[0], x[5], x[2]),           # EST + LWKR
        lambda x: (x[0], x[2]),                 # EST + job order
    ]

    best_makespan = float("inf")
    best_schedules = None
    for rule in rules:
        ms, sched = build_schedule(rule)
        if ms < best_makespan:
            best_makespan = ms
            best_schedules = sched

    # Stochastic trials with shuffled job priorities (more trials
    # improve chance of good schedule while staying fast)
    for _ in range(40):
        job_order = list(range(num_jobs))
        random.shuffle(job_order)
        def shuffle_key(x):
            return (x[0], job_order[x[2]], x[1])
        ms, sched = build_schedule(shuffle_key)
        if ms < best_makespan:
            best_makespan = ms
            best_schedules = sched

    return {
        "name": instance["name"],
        "makespan": best_makespan,
        "machine_schedules": best_schedules,
        "solved_by": "GreedyMultiRuleStochasticBaseline",
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
