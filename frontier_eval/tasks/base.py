from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from omegaconf import DictConfig
else:
    # Task subclasses only need this for static typing; specialized eval envs
    # should not have to install Hydra/OmegaConf just to import the base class.
    DictConfig = Any


class Task(ABC):
    NAME: str

    def __init__(self, cfg: DictConfig, repo_root: Path):
        self.cfg = cfg
        self.repo_root = repo_root

    @abstractmethod
    def initial_program_path(self) -> Path: ...

    @abstractmethod
    def evaluate_program(self, program_path: Path) -> Any: ...
