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
def _mutate(rng, base, scale=0.15):
    c = {"catalyst": base["catalyst"]}
    for name, (lo, hi) in task.BOUNDS.items():
        c[name] = float(np.clip(base[name] + rng.normal(0, scale * (hi - lo)), lo, hi))
    return c


def _crossover(rng, a, b):
    c = {"catalyst": a["catalyst"] if rng.random() < 0.5 else b["catalyst"]}
    for name, (lo, hi) in task.BOUNDS.items():
        alpha = rng.random()
        c[name] = float(np.clip(alpha * a[name] + (1 - alpha) * b[name], lo, hi))
    return c


def solve(seed: int = 0, budget: int = task.DEFAULT_BUDGET) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []

    catalysts = list(task.CATEGORIES["catalyst"])
    n_cats = len(catalysts)

    # Phase 1: Screen all catalysts at two different conditions
    for i in range(min(n_cats, budget)):
        candidate = {"catalyst": catalysts[i], "t_res": 360.0, "temperature": 100.0, "catalyst_loading": 2.0}
        history.append(task.evaluate(experiment, candidate))

    for i in range(min(n_cats, budget - len(history))):
        candidate = {"catalyst": catalysts[i], "t_res": 600.0, "temperature": 80.0, "catalyst_loading": 3.0}
        history.append(task.evaluate(experiment, candidate))

    if len(history) >= budget:
        return {"task_name": task.TASK_NAME, "algorithm_name": "evolve_v4", "seed": seed,
                "budget": budget, "history": history[:budget], "summary": task.summarize(history[:budget])}

    remaining = budget - len(history)
    weights = [0.1, 0.3, 0.5, 0.7, 0.9]
    per_weight = max(1, remaining // len(weights))
    extra = remaining - per_weight * len(weights)

    for wi, w in enumerate(weights):
        ranked = sorted(history, key=lambda r: task.scalarize(r, w), reverse=True)
        current_best = ranked[0]
        current_score = task.scalarize(current_best, w)
        pool = ranked[:3]

        n_steps = per_weight + (1 if wi < extra else 0)
        for step in range(n_steps):
            if len(history) >= budget:
                break
            progress = step / max(n_steps, 1)
            scale = 0.25 * (1.0 - 0.6 * progress)

            if len(pool) >= 2 and rng.random() < 0.2:
                i1, i2 = rng.choice(len(pool), 2, replace=False)
                candidate = _crossover(rng, pool[i1], pool[i2])
            elif rng.random() < 0.25 and len(pool) > 1:
                candidate = _mutate(rng, pool[rng.integers(0, len(pool))], scale=scale)
            else:
                candidate = _mutate(rng, current_best, scale=scale)

            record = task.evaluate(experiment, candidate)
            history.append(record)
            s = task.scalarize(record, w)
            if s > current_score:
                current_best = record
                current_score = s
            pool.append(record)
            pool.sort(key=lambda r: task.scalarize(r, w), reverse=True)
            pool = pool[:4]

    while len(history) < budget:
        w = weights[len(history) % len(weights)]
        best_rec = max(history, key=lambda r: task.scalarize(r, w))
        candidate = _mutate(rng, best_rec, scale=0.1)
        history.append(task.evaluate(experiment, candidate))

    return {"task_name": task.TASK_NAME, "algorithm_name": "evolve_v4", "seed": seed,
            "budget": budget, "history": history[:budget], "summary": task.summarize(history[:budget])}
# EVOLVE-BLOCK-END


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=task.DEFAULT_BUDGET)
    args = parser.parse_args()
    print(dump_json(solve(seed=args.seed, budget=args.budget)))


if __name__ == "__main__":
    main()
