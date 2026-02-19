from __future__ import annotations

from typing import Type

from frontier_eval.tasks.base import Task
from frontier_eval.tasks.iscso2015 import ISCSO2015Task
from frontier_eval.tasks.iscso2023 import ISCSO2023Task
from frontier_eval.tasks.manned_lunar_landing import MannedLunarLandingTask
from frontier_eval.tasks.perturbation_prediction import PerturbationPredictionTask
from frontier_eval.tasks.predict_modality import PredictModalityTask

_TASKS: dict[str, Type[Task]] = {
    MannedLunarLandingTask.NAME: MannedLunarLandingTask,
    ISCSO2015Task.NAME: ISCSO2015Task,
    ISCSO2023Task.NAME: ISCSO2023Task,
    PerturbationPredictionTask.NAME: PerturbationPredictionTask,
    PredictModalityTask.NAME: PredictModalityTask,
    ISCSO2015Task.NAME: ISCSO2015Task,
    ISCSO2023Task.NAME: ISCSO2023Task,
}


def get_task(name: str) -> Type[Task]:
    if name not in _TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(_TASKS)}")
    return _TASKS[name]
