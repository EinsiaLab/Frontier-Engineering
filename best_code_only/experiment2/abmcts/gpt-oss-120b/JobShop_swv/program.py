# EVOLVE-BLOCK-START
"""Simple greedy baseline for SWV (Storer, Wu & Vaccari, 1992).

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
import random
import time
from pathlib import Path
from typing import Any, List, Tuple

FAMILY_PREFIX = "swv"
FAMILY_NAME = "SWV (Storer, Wu & Vaccari, 1992)"


def _natural_key(name: str) -> List[object]:
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


def load_family_instances() -> List[dict[str, Any]]:
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


def _compute_job_prefix_sums(durations: List[List[int]]) -> List[List[int]]:
    """Return prefix sum of processing times for each job."""
    prefix = []
    for job in durations:
        ps = [0]
        for d in job:
            ps.append(ps[-1] + d)
        prefix.append(ps)  # length = len(job)+1, ps[i] = sum of first i ops
    return prefix


def _run_schedule(
    durations: List[List[int]],
    machines: List[List[int]],
    tie: str = "lrpt",
    rand: random.Random | None = None,
) -> Tuple[int, List[List[dict[str, int]]]]:
    """Execute a single list‑scheduling run.

    Args:
        durations: processing times matrix.
        machines: machine assignment matrix.
        tie: one of {"lrpt", "spt", "random"} defining the tie‑breaking rule.
        rand: Random instance used when tie == "random".

    Returns:
        makespan and machine schedules.
    """
    num_jobs = len(durations)
    total_operations = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines) + 1

    job_prefix = _compute_job_prefix_sums(durations)
    job_total = [pref[-1] for pref in job_prefix]

    next_op = [0] * num_jobs          # next operation index per job
    job_ready = [0] * num_jobs        # earliest time job can start next op
    machine_ready = [0] * num_machines

    machine_schedules: List[List[dict[str, int]]] = [
        [] for _ in range(num_machines)
    ]

    scheduled = 0
    while scheduled < total_operations:
        candidates: List[Tuple[int, int, int, int, int, int, float]] = []
        # (est, tie_key1, tie_key2, duration, job_id, op_idx, machine_id, rand_val)
        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= len(durations[job_id]):
                continue

            machine_id = machines[job_id][op_idx]
            duration = durations[job_id][op_idx]
            est = max(job_ready[job_id], machine_ready[machine_id])

            remaining = job_total[job_id] - job_prefix[job_id][op_idx]

            if tie == "lrpt":
                key1 = -remaining          # larger remaining first
                key2 = duration           # shorter duration as secondary
                rand_val = 0.0
            elif tie == "spt":
                key1 = duration           # shorter duration first
                key2 = -remaining         # larger remaining as secondary
                rand_val = 0.0
            else:  # random
                key1 = 0
                key2 = 0
                rand_val = rand.random() if rand else 0.0

            candidates.append(
                (est, key1, key2, duration, job_id, op_idx, machine_id, rand_val)
            )

        # Determine best candidate according to selected rule
        if tie == "random":
            best = min(
                candidates,
                key=lambda x: (x[0], x[7], x[3], x[4])  # est, random, duration, job_id
            )
        else:
            best = min(
                candidates,
                key=lambda x: (x[0], x[1], x[2], x[3], x[4])
            )

        est, _k1, _k2, duration, job_id, op_idx, machine_id, _rv = best
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


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Run several lightweight heuristics and keep the best schedule.

    Heuristics used:
        * EST + LRPT (default)
        * EST + SPT
        * EST + random tie‑breaking (three different seeds)

    Returns a dictionary compatible with the evaluation framework.
    """
    durations: List[List[int]] = instance["duration_matrix"]
    machines: List[List[int]] = instance["machines_matrix"]

    best_makespan = float("inf")
    best_schedule: List[List[dict[str, int]]] = []

    # Deterministic runs
    for tie_rule in ("lrpt", "spt"):
        makespan, schedule = _run_schedule(durations, machines, tie=tie_rule)
        if makespan < best_makespan:
            best_makespan = makespan
            best_schedule = schedule

    # Randomised runs – use deterministic seeds derived from instance name
    base_seed = sum(ord(c) for c in instance["name"])
    for i in range(3):
        rnd = random.Random(base_seed + i)
        makespan, schedule = _run_schedule(durations, machines, tie="random", rand=rnd)
        if makespan < best_makespan:
            best_makespan = makespan
            best_schedule = schedule

    return {
        "name": instance["name"],
        "makespan": int(best_makespan),
        "machine_schedules": best_schedule,
        "solved_by": "GreedyMultiHeuristicBaseline",
        "family": FAMILY_PREFIX,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=f"Run pure‑python baseline on {FAMILY_NAME}."
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
