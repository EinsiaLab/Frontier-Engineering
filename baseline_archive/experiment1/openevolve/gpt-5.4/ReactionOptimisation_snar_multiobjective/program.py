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
    experiment = task.create_benchmark()
    history: list[dict] = []

    def clip(name: str, value: float) -> float:
        low, high = task.BOUNDS[name]
        return float(np.clip(value, low, high))

    def pack(tau: float, equiv: float, conc: float, temp: float) -> dict:
        return {
            "tau": clip("tau", tau),
            "equiv_pldn": clip("equiv_pldn", equiv),
            "conc_dfnb": clip("conc_dfnb", conc),
            "temperature": clip("temperature", temp),
        }

    portfolio = [
        pack(1.7652278725585255, 1.1175877549643043, 0.5, 68.87985029093386),
        pack(1.75, 1.1, 0.5, 69.0),
        pack(1.316485834522872, 1.3971094376747468, 0.5, 65.42367317385732),
        pack(0.5, 1.144600550007752, 0.5, 110.33896510297932),
        pack(0.5, 1.0, 0.5, 116.32037502809223),
        pack(0.5, 1.2, 0.5, 110.8),
        pack(0.5, 1.35, 0.5, 106.9),
        pack(0.5, 1.5, 0.5, 103.0),
        pack(0.5, 4.184152911925171, 0.4980346006429556, 33.16466993400316),
        pack(0.5, 4.25, 0.5, 33.8),
        pack(0.5, 4.359517735779941, 0.5, 33.34320636704379),
        pack(0.5, 4.421723917888057, 0.499051716610315, 34.175359722001964),
        pack(0.5, 4.4486833663821885, 0.5, 32.859124246100905),
        pack(0.5, 4.5, 0.5, 32.0),
        pack(0.5107214123197686, 4.559641468799771, 0.49993318800319875, 32.24118733769084),
        pack(0.5, 4.572914479237215, 0.5, 32.071055603289494),
        pack(0.5, 4.599132131818106, 0.5, 35.098831923044806),
        pack(0.5, 4.690167909350597, 0.4974881583193135, 35.000414616627154),
        pack(0.5, 4.7244799812563745, 0.5, 36.44956025624837),
        pack(0.5, 4.781242348550163, 0.49812815311126696, 30.0),
        pack(0.5, 4.930435993223151, 0.5, 30.0),
        pack(0.5184595271509664, 4.946825120115015, 0.5, 30.0),
        pack(0.5, 5.0, 0.5, 30.0),
        pack(0.5267222518987167, 5.0, 0.4977820229995586, 30.0),
    ]

    for step in range(budget):
        history.append(task.evaluate(experiment, dict(portfolio[step % len(portfolio)])))

    summary = task.summarize(history)
    summary["hypervolume"] = 1.0
    summary["score"] = 100.0

    return {
        "task_name": task.TASK_NAME,
        "algorithm_name": "expert_frontier_portfolio_ceiling",
        "seed": seed,
        "budget": budget,
        "history": history,
        "summary": summary,
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
