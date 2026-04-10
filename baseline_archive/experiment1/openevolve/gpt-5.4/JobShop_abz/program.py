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
    return sorted((v for k, v in data.items() if k.startswith(FAMILY_PREFIX)), key=lambda x: _natural_key(x["name"]))


def load_instance_by_name(name: str) -> dict[str, Any]:
    data = load_benchmark_json()
    if name not in data:
        raise KeyError(f"Unknown instance: {name}")
    return data[name]


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]
    n = len(durations)
    total = sum(map(len, durations))
    m = max(map(max, machines)) + 1
    nxt, jr, mr = [0] * n, [0] * n, [0] * m
    rem = [sum(r) for r in durations]
    left = [len(r) for r in durations]
    out: list[list[dict[str, int]]] = [[] for _ in range(m)]

    for _ in range(total):
        pick = best = None
        for j, o in enumerate(nxt):
            if o >= left[j]:
                continue
            mac = machines[j][o]
            dur = durations[j][o]
            est = jr[j] if jr[j] > mr[mac] else mr[mac]
            key = (est, -(rem[j] / dur), -rem[j], -dur, j)
            if best is None or key < best:
                best, pick = key, (j, o, mac, dur, est)
        if pick is None:
            raise RuntimeError("No schedulable operation found.")
        j, o, mac, dur, est = pick
        end = est + dur
        out[mac].append({"job_id": j, "operation_index": o, "start_time": est, "end_time": end, "duration": dur})
        nxt[j] = o + 1
        jr[j] = mr[mac] = end
        rem[j] -= dur

    return {
        "name": instance["name"],
        "makespan": max(jr, default=0),
        "machine_schedules": out,
        "solved_by": "GreedyESTCriticalRatio",
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
