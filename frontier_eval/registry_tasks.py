from __future__ import annotations

from typing import Type

from frontier_eval.tasks.base import Task
from frontier_eval.tasks.cryptographic.task import (
    CryptoAES128Task,
    CryptoSHA3_256Task,
    CryptoSHA256Task,
)
from frontier_eval.tasks.denoising import DenoisingTask
from frontier_eval.tasks.car_aerodynamics_sensing import CarAerodynamicsSensingTask
from frontier_eval.tasks.dynamic_obstacle_navigation import DynamicObstacleNavigationTask
from frontier_eval.tasks.iscso2015 import ISCSO2015Task
from frontier_eval.tasks.iscso2023 import ISCSO2023Task
from frontier_eval.tasks.mla import MLATask
from frontier_eval.tasks.malloclab import MallocLabTask
from frontier_eval.tasks.manned_lunar_landing import MannedLunarLandingTask
from frontier_eval.tasks.perturbation_prediction import PerturbationPredictionTask
from frontier_eval.tasks.predict_modality import PredictModalityTask
from frontier_eval.tasks.trimul import TriMulTask
from frontier_eval.tasks.smoke import SmokeTask
from frontier_eval.tasks.unified import UnifiedTask

_TASKS: dict[str, Type[Task]] = {
    SmokeTask.NAME: SmokeTask,
    CryptoAES128Task.NAME: CryptoAES128Task,
    CryptoSHA256Task.NAME: CryptoSHA256Task,
    CryptoSHA3_256Task.NAME: CryptoSHA3_256Task,
    MannedLunarLandingTask.NAME: MannedLunarLandingTask,
    CarAerodynamicsSensingTask.NAME: CarAerodynamicsSensingTask,
    ISCSO2015Task.NAME: ISCSO2015Task,
    ISCSO2023Task.NAME: ISCSO2023Task,
    DenoisingTask.NAME: DenoisingTask,
    PerturbationPredictionTask.NAME: PerturbationPredictionTask,
    PredictModalityTask.NAME: PredictModalityTask,
    DynamicObstacleNavigationTask.NAME: DynamicObstacleNavigationTask,
    TriMulTask.NAME: TriMulTask,
    MLATask.NAME: MLATask,
    MallocLabTask.NAME: MallocLabTask,
    UnifiedTask.NAME: UnifiedTask,
}


def get_task(name: str) -> Type[Task]:
    if name not in _TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(_TASKS)}")
    return _TASKS[name]
