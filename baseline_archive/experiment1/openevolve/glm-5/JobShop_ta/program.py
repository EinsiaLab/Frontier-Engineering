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


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Multi-start greedy scheduler with bottleneck-aware dispatching rules."""
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]
    num_jobs = len(durations)
    total_ops = sum(len(j) for j in durations)
    num_machines = max(max(row) for row in machines) + 1
    work_total = [sum(j) for j in durations]
    ops_count = [len(j) for j in durations]
    
    # Precompute machine workload to identify bottlenecks
    mach_work = [0] * num_machines
    for j in range(num_jobs):
        for o in range(len(durations[j])):
            mach_work[machines[j][o]] += durations[j][o]
    max_mach_work = max(mach_work) if mach_work else 1
    avg_mach_work = sum(mach_work) / max(1, num_machines)
    
    # Estimate remaining work for each job position (critical path proxy)
    job_rem_work = [[0] * len(durations[j]) for j in range(num_jobs)]
    for j in range(num_jobs):
        rem = 0
        for o in range(len(durations[j]) - 1, -1, -1):
            job_rem_work[j][o] = rem
            rem += durations[j][o]

    def _solve(pk) -> tuple[int, list]:
        next_op = [0] * num_jobs
        job_ready = [0] * num_jobs
        mach_ready = [0] * num_machines
        sched = [[] for _ in range(num_machines)]
        work_rem = work_total[:]
        for _ in range(total_ops):
            best = None
            for j in range(num_jobs):
                o = next_op[j]
                if o >= len(durations[j]): continue
                m = machines[j][o]
                d = durations[j][o]
                est = max(job_ready[j], mach_ready[m])
                key = pk(est, d, j, o, m, work_rem, ops_count, mach_ready, mach_work, job_rem_work, max_mach_work, avg_mach_work)
                if best is None or key < best[0]:
                    best = (key, est, d, j, o, m)
            _, est, d, j, o, m = best
            end = est + d
            sched[m].append({"job_id": j, "operation_index": o, "start_time": est, "end_time": end, "duration": d})
            next_op[j] += 1
            job_ready[j] = end
            mach_ready[m] = end
            work_rem[j] -= d
        return max(job_ready), sched

    # Bottleneck-aware dispatching rules with enhanced priorities
    rules = [
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, d, j),  # EST+SPT
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, -d, j),  # EST+LPT
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, -wr[j], j),  # EST+MWR
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, oc[j]-o, j),  # EST+LOR
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, -(oc[j]-o), j),  # EST+MOR
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, d - wr[j]//max(1,oc[j]-o), j),  # Composite
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, -wr[j], d, j),  # MWR+SPT
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, mr[m], d, j),  # Machine ready
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, -mw[m], d, j),  # Bottleneck machine
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, -jrw[j][o], d, j),  # Critical path
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, -(wr[j]+jrw[j][o]), j),  # Total job work
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, -mw[m], -wr[j], j),  # Bottleneck+MWR
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, -jrw[j][o], -mw[m], j),  # Critical+bottleneck
        # Enhanced rules from best-performing variant
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, -(mw[m]-amw), d, j),  # Relative bottleneck
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, -(wr[j]*mw[m])//10000, j),  # Work*bottleneck
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, -(jrw[j][o]*mw[m])//10000, j),  # Critical*bottleneck
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, -wr[j]//max(1,oc[j]-o), d, j),  # Avg remaining+SPT
        lambda e,d,j,o,m,wr,oc,mr,mw,jrw,mmw,amw: (e, mr[m]-e, d, j),  # Wait time priority
    ]
    best_ms, best_sched = float('inf'), None
    for rule in rules:
        ms, sched = _solve(rule)
        if ms < best_ms:
            best_ms, best_sched = ms, sched
    return {"name": instance["name"], "makespan": best_ms, "machine_schedules": best_sched, "solved_by": "BottleneckAwareGreedy", "family": FAMILY_PREFIX}


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
