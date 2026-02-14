from __future__ import annotations

from typing import Type

from frontier_eval.tasks.base import Task
from frontier_eval.tasks.manned_lunar_landing import MannedLunarLandingTask

_TASKS: dict[str, Type[Task]] = {
    MannedLunarLandingTask.NAME: MannedLunarLandingTask,
}


def get_task(name: str) -> Type[Task]:
    if name not in _TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(_TASKS)}")
    return _TASKS[name]

