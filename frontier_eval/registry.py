from __future__ import annotations

from collections.abc import Callable
from typing import Any, Type

from frontier_eval.algorithms.base import Algorithm
from frontier_eval.algorithms.openevolve_algo import OpenEvolveAlgorithm
from frontier_eval.tasks.base import Task
from frontier_eval.tasks.manned_lunar_landing import MannedLunarLandingTask

_TASKS: dict[str, Type[Task]] = {
    MannedLunarLandingTask.NAME: MannedLunarLandingTask,
}

_ALGORITHMS: dict[str, Type[Algorithm]] = {
    OpenEvolveAlgorithm.NAME: OpenEvolveAlgorithm,
}


def get_task(name: str) -> Type[Task]:
    if name not in _TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(_TASKS)}")
    return _TASKS[name]


def get_algorithm(name: str) -> Type[Algorithm]:
    if name not in _ALGORITHMS:
        raise KeyError(f"Unknown algorithm '{name}'. Available: {sorted(_ALGORITHMS)}")
    return _ALGORITHMS[name]

