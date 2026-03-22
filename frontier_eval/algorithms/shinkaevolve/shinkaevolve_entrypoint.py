from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
import types
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
    for path in (
        repo_root,
        repo_root / "third_party" / "openevolve",
    ):
        path_str = str(path)
        if path.is_dir() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def _ensure_openevolve_evaluation_result_shim() -> None:
    try:
        from openevolve.evaluation_result import EvaluationResult as _EvaluationResult  # noqa: F401
        return
    except Exception:
        pass

    sys.modules.pop("openevolve", None)
    sys.modules.pop("openevolve.evaluation_result", None)

    package = types.ModuleType("openevolve")
    package.__path__ = []  # type: ignore[attr-defined]

    evaluation_result = types.ModuleType("openevolve.evaluation_result")

    class EvaluationResult:
        def __init__(self, metrics: dict[str, Any], artifacts: dict[str, Any] | None = None):
            self.metrics = metrics
            self.artifacts = artifacts or {}

        @classmethod
        def from_dict(cls, metrics: dict[str, Any]) -> "EvaluationResult":
            return cls(metrics=metrics)

        def to_dict(self) -> dict[str, Any]:
            return self.metrics

        def has_artifacts(self) -> bool:
            return bool(self.artifacts)

    evaluation_result.EvaluationResult = EvaluationResult
    package.evaluation_result = evaluation_result

    sys.modules["openevolve"] = package
    sys.modules["openevolve.evaluation_result"] = evaluation_result


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def _feedback_module():
    return importlib.import_module("frontier_eval.algorithms.shinkaevolve.context_feedback")


def _extract_metrics_and_artifacts(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if result is None:
        return {}, {}

    if isinstance(result, dict):
        nested_metrics = result.get("metrics")
        nested_artifacts = result.get("artifacts")
        if isinstance(nested_metrics, dict):
            return nested_metrics, nested_artifacts if isinstance(nested_artifacts, dict) else {}
        return result, {}

    metrics = getattr(result, "metrics", None)
    artifacts = getattr(result, "artifacts", None)
    if isinstance(metrics, dict):
        return metrics, artifacts if isinstance(artifacts, dict) else {}

    raise TypeError(f"Unsupported evaluation result type: {type(result)}")


def _truncate_middle(text: str, limit: int) -> str:
    return _feedback_module().truncate_middle(text, limit)


def _stringify(value: Any) -> str:
    return _feedback_module().stringify(value)


def _primary_error_message(artifacts: dict[str, Any]) -> str:
    return _feedback_module().primary_error_message(artifacts)


def _synthesize_text_feedback(
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    max_chars: int = 16_000,
) -> str:
    bundle = _build_context_bundle(metrics, artifacts, text_feedback_max_chars=max_chars)
    return bundle.text_feedback


def _build_context_bundle(
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    text_feedback_max_chars: int = 8_000,
    stdout_bridge_max_chars: int = 6_000,
    stderr_bridge_max_chars: int = 12_000,
) -> Any:
    return _feedback_module().build_context_bundle(
        metrics,
        artifacts,
        text_feedback_max_chars=text_feedback_max_chars,
        stdout_bridge_max_chars=stdout_bridge_max_chars,
        stderr_bridge_max_chars=stderr_bridge_max_chars,
    )


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
    _ensure_openevolve_evaluation_result_shim()

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
    bundle: Any | None = None
    try:
        raw = task.evaluate_program(program_path_p)
        metrics, artifacts = _extract_metrics_and_artifacts(raw)
        bundle = _build_context_bundle(metrics, artifacts)
    except Exception as e:
        metrics = {"combined_score": 0.0, "valid": 0.0, "error": str(e)}
        artifacts = {
            "error_message": str(e),
            "traceback": traceback.format_exc(),
        }
        bundle = _build_context_bundle(metrics, artifacts)

    metrics = dict(bundle.metrics) if bundle is not None else dict(metrics)
    artifacts = dict(bundle.artifacts) if bundle is not None else dict(artifacts)
    if bundle is not None and bundle.text_feedback:
        metrics["text_feedback"] = bundle.text_feedback

    correct = bundle.correct if bundle is not None else False
    error_msg = bundle.primary_error if bundle is not None else ""

    _write_json(results_dir_p / "metrics.json", metrics)
    _write_json(results_dir_p / "correct.json", {"correct": bool(correct), "error": error_msg})
    if artifacts:
        _write_json(results_dir_p / "artifacts.json", artifacts)
    if bundle is not None:
        _write_text(results_dir_p / "text_feedback.txt", bundle.text_feedback)
        _write_text(results_dir_p / "stdout_bridge.txt", bundle.stdout_bridge)
        _write_text(results_dir_p / "stderr_bridge.txt", bundle.stderr_bridge)
        _write_json(
            results_dir_p / "context_manifest.json",
            _feedback_module().build_context_manifest(bundle),
        )
        if bundle.stdout_bridge.strip():
            print(bundle.stdout_bridge, flush=True)
        if bundle.stderr_bridge.strip():
            print(bundle.stderr_bridge, file=sys.stderr, flush=True)

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
