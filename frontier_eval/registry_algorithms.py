from __future__ import annotations

from typing import Type

from frontier_eval.algorithms.base import Algorithm
from frontier_eval.algorithms.openevolve.algo import OpenEvolveAlgorithm
from frontier_eval.algorithms.shinkaevolve.algo import ShinkaEvolveAlgorithm

_ALGORITHMS: dict[str, Type[Algorithm]] = {
    OpenEvolveAlgorithm.NAME: OpenEvolveAlgorithm,
    ShinkaEvolveAlgorithm.NAME: ShinkaEvolveAlgorithm,
}


def get_algorithm(name: str) -> Type[Algorithm]:
    if name not in _ALGORITHMS:
        raise KeyError(f"Unknown algorithm '{name}'. Available: {sorted(_ALGORITHMS)}")
    return _ALGORITHMS[name]
