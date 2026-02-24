from __future__ import annotations

import argparse
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


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _extract_metrics_and_artifacts(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if result is None:
        return {}, {}

    if isinstance(result, dict):
        return result, {}

    metrics = getattr(result, "metrics", None)
    artifacts = getattr(result, "artifacts", None)
    if isinstance(metrics, dict):
        return metrics, artifacts if isinstance(artifacts, dict) else {}

    raise TypeError(f"Unsupported evaluation result type: {type(result)}")


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


def main(program_path: str, results_dir: str, *, task_name: str | None = None) -> int:
    repo_root = _find_repo_root()
    _ensure_repo_on_syspath(repo_root)

    task_name = (task_name or os.environ.get("FRONTIER_EVAL_TASK_NAME") or "").strip()
    if not task_name:
        raise RuntimeError(
            "Missing task name for ShinkaEvolve evaluator. Set env `FRONTIER_EVAL_TASK_NAME` "
            "or pass `--task_name`."
        )

    from omegaconf import OmegaConf

    from frontier_eval.registry_tasks import get_task

    task_cls = get_task(task_name)
    cfg = OmegaConf.create({"task": _task_cfg_from_env(task_name)})
    task = task_cls(cfg=cfg, repo_root=repo_root)

    program_path_p = Path(program_path).expanduser().resolve()
    results_dir_p = Path(results_dir).expanduser().resolve()
    results_dir_p.mkdir(parents=True, exist_ok=True)

    correct = False
    error_msg = ""
    metrics: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    try:
        raw = task.evaluate_program(program_path_p)
        metrics, artifacts = _extract_metrics_and_artifacts(raw)

        valid = metrics.get("valid", None)
        if isinstance(valid, (int, float)) and not isinstance(valid, bool):
            correct = float(valid) > 0.0
        else:
            correct = True

        err = artifacts.get("error_message") if isinstance(artifacts, dict) else None
        if isinstance(err, str) and err.strip():
            error_msg = err.strip()
            if correct:
                correct = False
    except Exception as e:
        correct = False
        error_msg = str(e)
        metrics = {"combined_score": 0.0, "valid": 0.0, "error": error_msg}

    _write_json(results_dir_p / "metrics.json", metrics)
    _write_json(results_dir_p / "correct.json", {"correct": bool(correct), "error": error_msg})
    if artifacts:
        _write_json(results_dir_p / "artifacts.json", artifacts)

    # Shinka's local job runner treats non-zero exit codes as evaluation crashes and
    # may raise before loading `metrics.json`. Always exit 0 and rely on correct.json.
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frontier Eval → ShinkaEvolve evaluation entrypoint.",
        add_help=True,
    )
    p.add_argument("--program_path", type=str, required=True)
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--task_name", type=str, default=None)
    # Shinka's scheduler may pass extra args (e.g. LLM settings). Ignore unknown args.
    args, _unknown = p.parse_known_args(argv)
    return args


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(main(args.program_path, args.results_dir, task_name=args.task_name))
