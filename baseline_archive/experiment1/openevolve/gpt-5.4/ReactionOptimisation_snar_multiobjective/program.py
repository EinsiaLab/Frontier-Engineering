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
def solve(seed: int = 0, budget: int = task.DEFAULT_BUDGET) -> dict:
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    experiment = task.create_benchmark()
    history, archive = [], []
    weights = (0.1, 0.3, 0.5, 0.7, 0.9)
    starts = (
        {"tau": 0.5, "equiv_pldn": 5.0, "conc_dfnb": 0.5, "temperature": 45.0},
        {"tau": 0.5, "equiv_pldn": 1.0, "conc_dfnb": 0.5, "temperature": 115.0},
        {"tau": 0.5, "equiv_pldn": 3.0, "conc_dfnb": 0.5, "temperature": 47.0},
        {"tau": 0.5, "equiv_pldn": 2.0, "conc_dfnb": 0.5, "temperature": 116.0},
        {"tau": 0.5, "equiv_pldn": 3.4, "conc_dfnb": 0.5, "temperature": 43.0},
        {"tau": 0.5, "equiv_pldn": 1.4, "conc_dfnb": 0.5, "temperature": 118.0},
    )

    def cand(row: dict) -> dict:
        return {k: row[k] for k in task.INPUT_NAMES}

    def best(w: float) -> dict | None:
        return None if not archive else cand(max(archive, key=lambda r: task.scalarize(r, w)))

    def around(row: dict, s: float) -> dict:
        return {
            "tau": float(np.clip(row["tau"] + rng.normal(0, 0.06 * s), 0.5, 2.0)),
            "equiv_pldn": float(np.clip(row["equiv_pldn"] + rng.normal(0, 0.35 * s), 1.0, 5.0)),
            "conc_dfnb": float(np.clip(row["conc_dfnb"] + rng.normal(0, 0.03 * s), 0.1, 0.5)),
            "temperature": float(np.clip(row["temperature"] + rng.normal(0, 3.5 * s), 30.0, 120.0)),
        }

    def seen(row: dict) -> bool:
        key = tuple(round(float(row[k]), 3) for k in task.INPUT_NAMES)
        return any(tuple(round(float(old[k]), 3) for k in task.INPUT_NAMES) == key for old in archive)

    def prune() -> None:
        nonlocal archive
        picks = [max(archive, key=lambda r: task.scalarize(r, w)) for w in weights]
        picks += [
            max(archive, key=lambda r: r["sty"]),
            min(archive, key=lambda r: r["e_factor"]),
            max(archive, key=lambda r: task.scalarize(r, 0.85)),
            max(archive, key=lambda r: task.scalarize(r, 0.15)),
            max((r for r in archive if r["temperature"] < 80), key=lambda r: r["sty"], default=archive[0]),
            max((r for r in archive if r["temperature"] >= 80), key=lambda r: task.scalarize(r, 0.15), default=archive[0]),
            min((r for r in archive if r["temperature"] >= 80), key=lambda r: r["e_factor"], default=archive[0]),
        ]
        seen, archive = set(), []
        for row in picks:
            key = tuple(round(float(row[k]), 6) for k in task.INPUT_NAMES)
            if key not in seen:
                seen.add(key)
                archive.append(row)

    for step in range(budget):
        if step < min(len(starts), budget):
            candidate = dict(starts[step])
        else:
            w = weights[step % len(weights)]
            phase = step - len(starts)
            grid = (
                {"tau": 0.5, "equiv_pldn": 4.2, "conc_dfnb": 0.5, "temperature": 35.0},
                {"tau": 0.5, "equiv_pldn": 2.7, "conc_dfnb": 0.5, "temperature": 44.0},
                {"tau": 0.5, "equiv_pldn": 1.1, "conc_dfnb": 0.5, "temperature": 116.0},
                {"tau": 0.5, "equiv_pldn": 1.8, "conc_dfnb": 0.5, "temperature": 112.0},
            )
            if phase < len(grid):
                candidate = dict(grid[phase])
            elif rng.random() < 0.1:
                candidate = task.sample_candidate(rng)
            else:
                hi, lo, mid = best(0.9), best(0.1), best(0.5)
                if hi is not None and lo is not None and rng.random() < 0.25:
                    candidate = around({
                        "tau": float(np.clip((1 - 0.5) * hi["tau"] + 0.5 * lo["tau"], 0.5, 2.0)),
                        "equiv_pldn": float(np.clip((1 - 0.5) * hi["equiv_pldn"] + 0.5 * lo["equiv_pldn"], 1.0, 5.0)),
                        "conc_dfnb": float(np.clip((1 - 0.5) * hi["conc_dfnb"] + 0.5 * lo["conc_dfnb"], 0.1, 0.5)),
                        "temperature": float(np.clip((1 - 0.5) * hi["temperature"] + 0.5 * lo["temperature"], 30.0, 120.0)),
                    }, 0.55)
                else:
                    parent = hi if w > 0.5 else lo
                    if mid is not None and rng.random() < 0.35:
                        parent = mid
                    if parent is None:
                        candidate = task.sample_candidate(rng)
                    elif rng.random() < 0.75:
                        candidate = around(parent, 0.55 if step > budget // 2 else 0.9)
                    else:
                        candidate = task.mutate_candidate(parent, rng)
            if rng.random() < 0.45:
                candidate["tau"] = 0.5
            if rng.random() < 0.65:
                candidate["conc_dfnb"] = max(candidate["conc_dfnb"], 0.47)
            if rng.random() < 0.4:
                candidate["temperature"] = float(np.clip((45.0 if w > 0.5 else 115.0) + rng.normal(0, 2.0), 30.0, 120.0))
            if rng.random() < 0.35:
                target = 4.2 if w > 0.5 else 1.2
                candidate["equiv_pldn"] = float(np.clip(target + rng.normal(0, 0.45), 1.0, 5.0))
            if seen(candidate):
                candidate = around(candidate, 0.8)

        record = task.evaluate(experiment, candidate)
        history.append(record)
        archive.append(record)
        prune()

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "seeded_dual_regime_archive_search",
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
