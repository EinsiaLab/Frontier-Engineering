# EVOLVE-BLOCK-START
"""Improved greedy baseline for TA (Taillard, 1993).

Baseline constraints:
- Pure Python implementation.
- Standard library only.
- No external solver usage.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, List, Tuple

FAMILY_PREFIX = "ta"
FAMILY_NAME = "TA (Taillard, 1993)"
# Number of independent greedy runs; higher may improve makespan at modest cost.
_ITERATIONS = 5


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
        + ", ".join(str(p) for p in candidates)
    )


def load_benchmark_json() -> dict[str, dict[str, Any]]:
    with _benchmark_json_path().open("r", encoding="utf-8") as f:
        return json.load(f)


def load_family_instances() -> List[dict[str, Any]]:
    data = load_benchmark_json()
    selected = [
        v for n, v in data.items() if n.startswith(FAMILY_PREFIX)
    ]
    return sorted(selected, key=lambda x: _natural_key(x["name"]))


def load_instance_by_name(name: str) -> dict[str, Any]:
    data = load_benchmark_json()
    if name not in data:
        raise KeyError(f"Unknown instance: {name}")
    return data[name]


def _schedule_once(
    durations: List[List[int]],
    machines: List[List[int]],
    seed: int | None = None,
) -> Tuple[int, List[List[dict[str, int]]]]:
    """Perform a single greedy EST‑based schedule.

    Tie‑breaking uses a combination of remaining job work, processing time,
    and a deterministic random component (seed) to diversify runs.
    Returns makespan and per‑machine operation lists.
    """
    if seed is not None:
        rnd = random.Random(seed)
    else:
        rnd = random.Random()

    num_jobs = len(durations)
    total_ops = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines) + 1

    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    machine_schedules: List[List[dict[str, int]]] = [[] for _ in range(num_machines)]

    scheduled = 0
    while scheduled < total_ops:
        candidates: List[Tuple[int, int, int, int, int, float]] = []
        # (est, -remaining_job_time, duration, job_id, op_idx, rand_key)
        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= len(durations[job_id]):
                continue
            machine_id = machines[job_id][op_idx]
            duration = durations[job_id][op_idx]
            est = max(job_ready[job_id], machine_ready[machine_id])
            remaining = sum(durations[job_id][op_idx + 1 :])
            # random key to break exact ties deterministically per seed
            rand_key = rnd.random()
            candidates.append((est, -remaining, duration, job_id, op_idx, rand_key))

        # Choose the best candidate according to the composite key
        est, _, duration, job_id, op_idx, _ = min(
            candidates,
            key=lambda x: (x[0], x[1], x[2], x[5]),
        )
        machine_id = machines[job_id][op_idx]
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
    """Run several greedy schedules and keep the best makespan.

    Returned dict contains at least:
        - name
        - makespan
        - machine_schedules
    """
    durations: List[List[int]] = instance["duration_matrix"]
    machines: List[List[int]] = instance["machines_matrix"]

    best_makespan = float("inf")
    best_sched: List[List[dict[str, int]]] = []

    base_seed = hash(instance["name"]) & 0xFFFFFFFF
    for i in range(_ITERATIONS):
        seed = base_seed + i
        makespan, sched = _schedule_once(durations, machines, seed=seed)
        if makespan < best_makespan:
            best_makespan = makespan
            best_sched = sched

    return {
        "name": instance["name"],
        "makespan": best_makespan,
        "machine_schedules": best_sched,
        "solved_by": "IteratedGreedyESTBaseline",
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
