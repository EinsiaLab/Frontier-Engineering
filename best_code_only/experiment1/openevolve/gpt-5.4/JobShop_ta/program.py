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
    return sorted(
        (v for k, v in data.items() if k.startswith(FAMILY_PREFIX)),
        key=lambda x: _natural_key(x["name"]),
    )


def load_instance_by_name(name: str) -> dict[str, Any]:
    data = load_benchmark_json()
    if name not in data:
        raise KeyError(f"Unknown instance: {name}")
    return data[name]


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Greedy EST scheduler favoring long job tails over machine load."""
    d: list[list[int]] = instance["duration_matrix"]
    m: list[list[int]] = instance["machines_matrix"]
    n = len(d)
    total = sum(map(len, d))
    nm = max(map(max, m)) + 1 if m else 0
    nxt = [0] * n
    jr = [0] * n
    mr = [0] * nm
    rem = [sum(r) for r in d]
    tail = [r[:] for r in d]
    for r in tail:
        for i in range(len(r) - 2, -1, -1):
            r[i] += r[i + 1]
    out: list[list[dict[str, int]]] = [[] for _ in range(nm)]

    for _ in range(total):
        best = pick = None
        for j in range(n):
            k = nxt[j]
            if k >= len(d[j]):
                continue
            mac = m[j][k]
            dur = d[j][k]
            s = jr[j]
            if mr[mac] > s:
                s = mr[mac]
            key = (s, -tail[j][k], -rem[j], dur, j)
            if best is None or key < best:
                best, pick = key, (j, k, mac, dur, s)
        if pick is None:
            raise RuntimeError("No schedulable operation found.")
        j, k, mac, dur, s = pick
        e = s + dur
        out[mac].append({"job_id": j, "operation_index": k, "start_time": s, "end_time": e, "duration": dur})
        nxt[j] += 1
        jr[j] = e
        mr[mac] = e
        rem[j] -= dur

    return {
        "name": instance["name"],
        "makespan": max(jr) if jr else 0,
        "machine_schedules": out,
        "solved_by": "GreedyESTTailWork",
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
