"""Simple baseline for the Reizman Suzuki emulator task."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _is_repo_root(path: Path) -> bool:
    return (path / "benchmarks").is_dir() and (path / "frontier_eval").is_dir()


def _ensure_domain_on_path() -> None:
    env_root = (os.environ.get("FRONTIER_ENGINEERING_ROOT") or "").strip()
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())

    here = Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])

    repo_root = next((path for path in candidates if _is_repo_root(path)), None)
    if repo_root is None:
        raise RuntimeError("Could not locate repository root for ReactionOptimisation.")

    domain_root = repo_root / "benchmarks" / "ReactionOptimisation"
    if not domain_root.is_dir():
        raise RuntimeError(f"ReactionOptimisation directory not found under {repo_root}.")

    domain_root_str = str(domain_root)
    if domain_root_str not in sys.path:
        sys.path.insert(0, domain_root_str)


_ensure_domain_on_path()

from reizman_suzuki_pareto import task
from shared.utils import dump_json, seed_everything


# EVOLVE-BLOCK-START
def solve(seed: int = 0, budget: int = task.DEFAULT_BUDGET) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    ex = task.create_benchmark()
    history: list[dict] = []
    cats, b = task.CATEGORIES["catalyst"], task.BOUNDS
    out = {"task_name": task.TASK_NAME, "algorithm_name": "screened_pareto_search", "seed": seed, "budget": budget}

    def add(c: dict) -> dict:
        r = task.evaluate(ex, c)
        history.append(r)
        return r

    if budget <= 0:
        return {**out, "history": history, "summary": task.summarize(history)}

    n = min(len(cats), max(4, budget // 3))
    screen = []
    for i in rng.choice(len(cats), size=n, replace=False):
        c = {
            "catalyst": cats[int(i)],
            "t_res": 360.0 + rng.normal(0, 40),
            "temperature": 100.0 + rng.normal(0, 6),
            "catalyst_loading": 2.0 + rng.normal(0, 0.25),
        }
        screen.append((c, add(c)))

    ws = (0.2, 0.5, 0.8)
    elites = [(w, *max(screen, key=lambda x: task.scalarize(x[1], w))) for w in ws]

    while len(history) < budget:
        j = len(history) % len(ws)
        w, base, best = elites[j]
        c = dict(base)
        if rng.random() < 0.15:
            c["catalyst"] = cats[rng.integers(len(cats))]
        c["t_res"] = float(base["t_res"] + rng.normal(0, 70))
        c["temperature"] = float(base["temperature"] + rng.normal(0, 10))
        c["catalyst_loading"] = float(base["catalyst_loading"] + rng.normal(0, 0.35))
        try:
            r = add(c)
        except Exception:
            r = add(task.sample_candidate(rng))
        if task.scalarize(r, w) >= task.scalarize(best, w):
            elites[j] = (w, c, r)

    return {**out, "history": history, "summary": task.summarize(history)}
# EVOLVE-BLOCK-END


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=task.DEFAULT_BUDGET)
    args = parser.parse_args()
    print(dump_json(solve(seed=args.seed, budget=args.budget)))


if __name__ == "__main__":
    main()
