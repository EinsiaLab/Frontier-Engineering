from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _is_repo_root(path: Path) -> bool:
    if not (path / "frontier_eval").is_dir():
        return False
    if (path / "benchmarks").is_dir():
        return True
    return (path / "Astrodynamics").is_dir() and (path / "ElectronicDesignAutomation").is_dir()


def _find_repo_root() -> Path:
    if "FRONTIER_ENGINEERING_ROOT" in os.environ:
        return Path(os.environ["FRONTIER_ENGINEERING_ROOT"]).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if _is_repo_root(parent):
            return parent
    return Path.cwd().resolve()


def _ensure_repo_on_syspath(repo_root: Path) -> None:
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _task_cfg_from_env(task_name: str) -> dict[str, Any]:
    raw = str(os.environ.get("FRONTIER_EVAL_TASK_CFG_JSON", "") or "").strip()
    if not raw:
        return {"name": task_name}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {"name": task_name}
    if not isinstance(parsed, dict):
        return {"name": task_name}
    cfg = dict(parsed)
    cfg["name"] = str(cfg.get("name") or task_name)
    return cfg


def evaluate(program_path: str) -> Any:
    repo_root = _find_repo_root()
    _ensure_repo_on_syspath(repo_root)

    task_name = (os.environ.get("FRONTIER_EVAL_TASK_NAME") or "").strip()
    if not task_name:
        raise RuntimeError(
            "Missing task name for OpenEvolve evaluator. Set env `FRONTIER_EVAL_TASK_NAME`."
        )

    from omegaconf import OmegaConf

    from frontier_eval.registry_tasks import get_task

    task_cls = get_task(task_name)
    cfg = OmegaConf.create({"task": _task_cfg_from_env(task_name)})
    task = task_cls(cfg=cfg, repo_root=repo_root)
    return task.evaluate_program(Path(program_path).expanduser().resolve())
