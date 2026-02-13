from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from frontier_eval.tasks.base import Task


class Algorithm(ABC):
    NAME: str

    def __init__(self, cfg: DictConfig, repo_root: Path):
        self.cfg = cfg
        self.repo_root = repo_root

    @abstractmethod
    async def run(self, task: Task) -> None: ...

