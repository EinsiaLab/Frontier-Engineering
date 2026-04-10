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
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]
    n = len(durations)
    m = max(map(max, machines)) + 1 if machines else 0
    total = sum(map(len, durations))
    nxt = [0] * n
    jr = [0] * n
    mr = [0] * m
    rem = [sum(row) for row in durations]
    sched: list[list[dict[str, int]]] = [[] for _ in range(m)]

    for _ in range(total):
        best = pick = None
        for j, ops in enumerate(durations):
            o = nxt[j]
            if o >= len(ops):
                continue
            mac = machines[j][o]
            dur = ops[o]
            est = jr[j] if jr[j] > mr[mac] else mr[mac]
            key = (est, dur, -rem[j], j)
            if best is None or key < best:
                best = key
                pick = (j, o, mac, dur, est)
        if pick is None:
            raise RuntimeError("No schedulable operation found.")
        j, o, mac, dur, est = pick
        end = est + dur
        sched[mac].append({"job_id": j, "operation_index": o, "start_time": est, "end_time": end, "duration": dur})
        nxt[j] = o + 1
        jr[j] = end
        mr[mac] = end
        rem[j] -= dur

    return {
        "name": instance["name"],
        "makespan": max(jr) if jr else 0,
        "machine_schedules": sched,
        "solved_by": "GreedyESTSPT_LRPT",
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
