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
import time
import random
from pathlib import Path
from typing import Any

FAMILY_PREFIX = "abz"


def _benchmark_json_path() -> Path:
    env_path = os.environ.get("JOBSHOP_BENCHMARK_JSON", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.is_file():
            return candidate

    candidates = [
        Path(__file__).resolve().parents[2] / "data" / "benchmark_instances.json",
        Path(__file__).resolve().parents[1] / "data" / "benchmark_instances.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError("benchmark_instances.json not found")


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
    return sorted(selected, key=lambda x: x["name"])


def load_instance_by_name(name: str) -> dict[str, Any]:
    data = load_benchmark_json()
    if name not in data:
        raise KeyError(f"Unknown instance: {name}")
    return data[name]


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Greedy EST+MWKR scheduler (pure Python) + limited local search."""
    durations = instance["duration_matrix"]
    machines = instance["machines_matrix"]

    num_jobs = len(durations)
    total_ops = sum(len(job) for job in durations)
    num_machines = max(max(row) for row in machines) + 1

    next_op = [0] * num_jobs
    job_ready = [0] * num_jobs
    machine_ready = [0] * num_machines
    remaining = [sum(durations[j]) for j in range(num_jobs)]

    machine_schedules = [[] for _ in range(num_machines)]

    scheduled = 0
    while scheduled < total_ops:
        candidates = []
        for job_id in range(num_jobs):
            op_idx = next_op[job_id]
            if op_idx >= len(durations[job_id]):
                continue
            machine_id = machines[job_id][op_idx]
            duration = durations[job_id][op_idx]
            est = max(job_ready[job_id], machine_ready[machine_id])
            rem = remaining[job_id]
            candidates.append((est, -rem, job_id, op_idx, machine_id, duration))

        if not candidates:
            raise RuntimeError("No schedulable operation found.")

        est, _, job_id, op_idx, machine_id, duration = min(
            candidates, key=lambda x: (x[0], x[1], x[2])
        )
        end = est + duration

        machine_schedules[machine_id].append({
            "job_id": job_id,
            "operation_index": op_idx,
            "start_time": est,
            "end_time": end,
            "duration": duration,
        })

        next_op[job_id] += 1
        job_ready[job_id] = end
        machine_ready[machine_id] = end
        remaining[job_id] -= duration
        scheduled += 1

    # Extract machine sequences
    machine_seqs = []
    for m_ops in machine_schedules:
        sorted_ops = sorted(m_ops, key=lambda o: o["start_time"])
        seq = [(o["job_id"], o["operation_index"]) for o in sorted_ops]
        machine_seqs.append(seq)

    # Local search: adjacent swaps on machine sequences (seeded for determinism)
    random.seed(42)
    makespan = max(job_ready) if job_ready else 0
    improved = True
    while improved:
        improved = False
        machine_order = list(range(num_machines))
        random.shuffle(machine_order)
        for m in machine_order:
            seq = machine_seqs[m]
            if len(seq) < 2:
                continue
            swap_positions = list(range(len(seq) - 1))
            random.shuffle(swap_positions)
            for i in swap_positions:
                # try swap
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
                new_makespan, _ = _compute_makespan(durations, machines, machine_seqs)
                if new_makespan < makespan:
                    makespan = new_makespan
                    improved = True
                else:
                    # revert
                    seq[i], seq[i + 1] = seq[i + 1], seq[i]

    # Rebuild final schedule
    _, final_starts = _compute_makespan(durations, machines, machine_seqs)
    new_machine_schedules = [[] for _ in range(num_machines)]
    for m, seq in enumerate(machine_seqs):
        for job_id, op_idx in seq:
            st = final_starts[job_id][op_idx]
            dur = durations[job_id][op_idx]
            new_machine_schedules[m].append({
                "job_id": job_id,
                "operation_index": op_idx,
                "start_time": st,
                "end_time": st + dur,
                "duration": dur,
            })

    return {
        "name": instance["name"],
        "makespan": makespan,
        "machine_schedules": new_machine_schedules,
        "solved_by": "GreedyESTMWKR+LSBaseline",
        "family": FAMILY_PREFIX,
    }


def _compute_makespan(durations, machines, machine_seqs):
    """Compute earliest starts and makespan given machine sequences (pure Python)."""
    n_jobs = len(durations)
    n_ops_per_job = [len(d) for d in durations]
    start = [[0] * n_ops_per_job[j] for j in range(n_jobs)]
    changed = True
    iters = 0
    max_iters = sum(n_ops_per_job) * 2
    while changed and iters < max_iters:
        changed = False
        iters += 1
        # job precedences
        for j in range(n_jobs):
            for k in range(1, n_ops_per_job[j]):
                prev_end = start[j][k - 1] + durations[j][k - 1]
                if start[j][k] < prev_end:
                    start[j][k] = prev_end
                    changed = True
        # machine precedences
        for m, seq in enumerate(machine_seqs):
            for i in range(1, len(seq)):
                j1, o1 = seq[i - 1]
                j2, o2 = seq[i]
                prev_end = start[j1][o1] + durations[j1][o1]
                if start[j2][o2] < prev_end:
                    start[j2][o2] = prev_end
                    changed = True
    ms = 0
    for j in range(n_jobs):
        for k in range(n_ops_per_job[j]):
            ms = max(ms, start[j][k] + durations[j][k])
    return ms, start


def _cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, default=None)
    parser.add_argument("--max-instances", type=int, default=5)
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
