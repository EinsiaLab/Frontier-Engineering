"""Simple baseline for the SnAr multi-objective task."""

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

from shared.utils import dump_json, seed_everything
from snar_multiobjective import task


# EVOLVE-BLOCK-START
def _dominates(a, b):
    return (a["sty"] >= b["sty"] and a["e_factor"] <= b["e_factor"] and
            (a["sty"] > b["sty"] or a["e_factor"] < b["e_factor"]))


def _get_pareto_front(history):
    if not history:
        return []
    front = []
    for i, h in enumerate(history):
        if not any(_dominates(history[j], h) for j in range(len(history)) if j != i):
            front.append(h)
    return front


def _local_mutate(candidate, rng, scale=0.08):
    proposal = dict(candidate)
    for name, (low, high) in task.BOUNDS.items():
        span = high - low
        proposal[name] = float(np.clip(candidate[name] + rng.normal(0.0, scale * span), low, high))
    return proposal


def _blend_crossover(p1, p2, rng, alpha=0.3):
    child = {}
    for name, (low, high) in task.BOUNDS.items():
        v1, v2 = p1[name], p2[name]
        lo = min(v1, v2) - alpha * abs(v2 - v1)
        hi = max(v1, v2) + alpha * abs(v2 - v1)
        child[name] = float(np.clip(rng.uniform(lo, hi), low, high))
    return child


def solve(seed: int = 0, budget: int = task.DEFAULT_BUDGET) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history: list[dict] = []

    anchor_points = [
        {"tau": 0.5, "equiv_pldn": 3.8, "conc_dfnb": 0.5, "temperature": 35.0},
        {"tau": 0.5, "equiv_pldn": 4.0, "conc_dfnb": 0.5, "temperature": 47.0},
        {"tau": 1.5, "equiv_pldn": 1.1, "conc_dfnb": 0.5, "temperature": 68.0},
        {"tau": 0.5, "equiv_pldn": 1.0, "conc_dfnb": 0.5, "temperature": 116.0},
        {"tau": 0.7, "equiv_pldn": 2.5, "conc_dfnb": 0.45, "temperature": 55.0},
        {"tau": 0.5, "equiv_pldn": 5.0, "conc_dfnb": 0.5, "temperature": 30.0},
    ]
    n_init = min(len(anchor_points), budget)
    target_weights = [0.1, 0.3, 0.5, 0.7, 0.9]

    for step in range(budget):
        if step < n_init:
            candidate = anchor_points[step]
        else:
            exploit_step = step - n_init
            remaining = budget - n_init
            progress = exploit_step / max(remaining - 1, 1)
            weight = target_weights[exploit_step % len(target_weights)]
            pareto = _get_pareto_front(history)
            explore_prob = max(0.12 * (1.0 - progress), 0.02)

            if rng.random() < explore_prob:
                candidate = task.sample_candidate(rng)
            else:
                strategy = rng.random()
                if strategy < 0.4 and len(pareto) >= 1:
                    best_rec = max(history, key=lambda r: task.scalarize(r, weight))
                    best_cand = {name: best_rec[name] for name in task.INPUT_NAMES}
                    scale = 0.07 * (1.0 - 0.6 * progress)
                    candidate = _local_mutate(best_cand, rng, scale=scale)
                elif strategy < 0.65 and len(pareto) >= 2:
                    idxs = rng.choice(len(pareto), size=2, replace=False)
                    p1 = {name: pareto[idxs[0]][name] for name in task.INPUT_NAMES}
                    p2 = {name: pareto[idxs[1]][name] for name in task.INPUT_NAMES}
                    candidate = _blend_crossover(p1, p2, rng)
                elif strategy < 0.8 and len(pareto) >= 1:
                    p_idx = rng.integers(len(pareto))
                    parent = {name: pareto[p_idx][name] for name in task.INPUT_NAMES}
                    candidate = _local_mutate(parent, rng, scale=0.10)
                else:
                    w2 = rng.uniform(0.0, 1.0)
                    best_rec = max(history, key=lambda r: task.scalarize(r, w2))
                    best_cand = {name: best_rec[name] for name in task.INPUT_NAMES}
                    candidate = _local_mutate(best_cand, rng, scale=0.04)

        record = task.evaluate(experiment, candidate)
        history.append(record)

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "pareto_anchor6_blx_evolve",
        "seed": seed,
        "budget": budget,
        "history": history,
        "summary": task.summarize(history),
    }
# EVOLVE-BLOCK-END


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=int, default=task.DEFAULT_BUDGET)
    args = parser.parse_args()
    print(dump_json(solve(seed=args.seed, budget=args.budget)))


if __name__ == "__main__":
    main()
