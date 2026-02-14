from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omegaconf import DictConfig


class Task(ABC):
    NAME: str

    def __init__(self, cfg: DictConfig, repo_root: Path):
        self.cfg = cfg
        self.repo_root = repo_root

    @abstractmethod
    def initial_program_path(self) -> Path: ...

    @abstractmethod
    def evaluate_program(self, program_path: Path) -> Any: ...
