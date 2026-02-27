from __future__ import annotations

"""ShinkaEvolve algorithm adapter for Frontier Eval."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from frontier_eval.algorithms.base import Algorithm
from frontier_eval.tasks.base import Task


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _as_plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, DictConfig):
        plain = OmegaConf.to_container(value, resolve=True)
        if plain is None:
            return {}
        if isinstance(plain, dict):
            return plain
        raise TypeError(f"Expected a mapping, got {type(plain)}")
    raise TypeError(f"Expected a mapping, got {type(value)}")


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
            continue
        base[key] = value
    return base


def _hms_from_seconds(seconds: int) -> str:
    seconds_i = max(0, int(seconds))
    h = seconds_i // 3600
    m = (seconds_i % 3600) // 60
    s = seconds_i % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _extract_best_metrics(output_root: Path) -> tuple[dict[str, Any] | None, Path | None]:
    best_correct_score: float | None = None
    best_correct_metrics: dict[str, Any] | None = None
    best_correct_dir: Path | None = None

    best_any_score: float | None = None
    best_any_metrics: dict[str, Any] | None = None
    best_any_dir: Path | None = None

    for metrics_path in output_root.rglob("metrics.json"):
        metrics_raw = _read_json(metrics_path)
        if not isinstance(metrics_raw, dict):
            continue
        score = _as_float(metrics_raw.get("combined_score"))
        if score is None:
            continue

        correct_raw = _read_json(metrics_path.parent / "correct.json")
        correct = bool(correct_raw.get("correct")) if isinstance(correct_raw, dict) else True

        if best_any_score is None or score > best_any_score:
            best_any_score = score
            best_any_metrics = metrics_raw
            best_any_dir = metrics_path.parent

        if correct and (best_correct_score is None or score > best_correct_score):
            best_correct_score = score
            best_correct_metrics = metrics_raw
            best_correct_dir = metrics_path.parent

    if best_correct_metrics is not None:
        return best_correct_metrics, best_correct_dir
    return best_any_metrics, best_any_dir


class ShinkaEvolveAlgorithm(Algorithm):
    NAME = "shinkaevolve"

    def __init__(self, cfg: DictConfig, repo_root: Path):
        super().__init__(cfg=cfg, repo_root=repo_root)

        try:
            from shinka.core import EvolutionConfig, EvolutionRunner
            from shinka.database import DatabaseConfig
            from shinka.launch import LocalJobConfig
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "ShinkaEvolve is not importable.\n"
                "Install it from the official repo (the PyPI package `shinka` is NOT ShinkaEvolve).\n"
                "Recommended (editable VCS install):\n"
                "  pip install -e 'git+https://github.com/SakanaAI/ShinkaEvolve.git#egg=shinka'\n"
                "If you installed via `pip install git+...` and see `ModuleNotFoundError: shinka.core`,\n"
                "reinstall with the editable command above.\n"
            ) from e

        self._se_EvolutionConfig = EvolutionConfig
        self._se_EvolutionRunner = EvolutionRunner
        self._se_DatabaseConfig = DatabaseConfig
        self._se_LocalJobConfig = LocalJobConfig

    async def run(self, task: Task) -> None:
        algo_cfg = self.cfg.algorithm
        run_cfg = self.cfg.run
        llm_cfg = self.cfg.llm

        output_dir = Path(str(getattr(run_cfg, "output_dir"))).expanduser().resolve()
        shinka_dir = (output_dir / "shinkaevolve").resolve()
        shinka_dir.mkdir(parents=True, exist_ok=True)

        # Hydra config (frontier_eval/conf/algorithm/shinkaevolve.yaml)
        # NOTE: Shinka counts the initial evaluation as generation 0, so to align with
        # OpenEvolve's `iterations=0` (baseline-only), we run `num_generations = max_generations + 1`.
        max_generations = int(getattr(algo_cfg, "max_generations", 0) or 0)
        num_generations = max(0, max_generations) + 1

        job_type = str(getattr(algo_cfg, "job_type", "local") or "local")
        max_parallel = int(getattr(algo_cfg, "max_parallel", 1) or 1)
        use_text_feedback = bool(getattr(algo_cfg, "use_text_feedback", True))
        verbose = bool(getattr(algo_cfg, "verbose", False))

        evaluator_timeout_s = int(getattr(algo_cfg, "evaluator_timeout_s", 300) or 300)

        api_base = str(getattr(llm_cfg, "api_base", "") or "")
        api_key = str(getattr(llm_cfg, "api_key", "") or "")
        model = str(getattr(llm_cfg, "model", "") or "")

        has_any_api_key = bool(api_key) or any(
            bool(os.environ.get(k))
            for k in (
                "OPENAI_API_KEY",
                "OPENROUTER_API_KEY",
                "DEEPSEEK_API_KEY",
                "ANTHROPIC_API_KEY",
                "GEMINI_API_KEY",
            )
        )
        if max_generations > 0 and not has_any_api_key:
            raise RuntimeError(
                "Missing API key for ShinkaEvolve. Set `OPENAI_API_KEY` or `llm.api_key` "
                "when `algorithm.max_generations > 0`."
            )

        initial_program = task.initial_program_path()
        evaluator_file = (
            self.repo_root
            / "frontier_eval"
            / "algorithms"
            / "shinkaevolve"
            / "shinkaevolve_entrypoint.py"
        ).resolve()
        task_cfg_view = OmegaConf.to_container(getattr(self.cfg, "task", None), resolve=True)
        task_cfg_payload: dict[str, Any] = task_cfg_view if isinstance(task_cfg_view, dict) else {}
        task_cfg_payload = dict(task_cfg_payload)
        task_cfg_payload.setdefault("name", task.NAME)

        # Ensure Frontier Eval context is visible to Shinka's evaluation subprocesses.
        os.environ["FRONTIER_EVAL_TASK_NAME"] = task.NAME
        os.environ["FRONTIER_EVAL_TASK_CFG_JSON"] = json.dumps(task_cfg_payload, ensure_ascii=False)
        os.environ["FRONTIER_EVAL_EVALUATOR_TIMEOUT_S"] = str(evaluator_timeout_s)
        os.environ.setdefault("FRONTIER_ENGINEERING_ROOT", str(self.repo_root))
        os.environ.setdefault("PYTHONUNBUFFERED", "1")

        # Best-effort: map Frontier's OpenAI-compatible config to Shinka's provider-specific env vars.
        if api_key:
            os.environ.setdefault("OPENAI_API_KEY", api_key)
            if api_base:
                os.environ.setdefault("OPENAI_API_BASE", api_base)
                os.environ.setdefault("OPENAI_BASE_URL", api_base)
            if "openrouter.ai" in api_base:
                os.environ.setdefault("OPENROUTER_API_KEY", api_key)
            if "deepseek" in api_base:
                os.environ.setdefault("DEEPSEEK_API_KEY", api_key)
            if "anthropic" in api_base:
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)

        se_overrides = _as_plain_mapping(getattr(algo_cfg, "se", None))
        evo_overrides = _as_plain_mapping(se_overrides.get("evo")) if isinstance(se_overrides, dict) else {}
        job_overrides = _as_plain_mapping(se_overrides.get("job")) if isinstance(se_overrides, dict) else {}
        db_overrides = _as_plain_mapping(se_overrides.get("db")) if isinstance(se_overrides, dict) else {}

        evo_kwargs: dict[str, Any] = {
            "init_program_path": str(initial_program),
            "results_dir": str(shinka_dir),
            "language": "python",
            "use_text_feedback": use_text_feedback,
            "num_generations": int(num_generations),
            "max_parallel_jobs": int(max_parallel),
            "job_type": job_type,
            "llm_models": [model] if model else None,
            "llm_kwargs": {
                "temperatures": float(getattr(llm_cfg, "temperature", 0.7)),
                "max_tokens": int(getattr(llm_cfg, "max_tokens", 4096)),
            },
        }

        problem_description = str(getattr(algo_cfg, "problem_description", "") or "").strip()
        if problem_description:
            evo_kwargs["task_sys_msg"] = problem_description

        _deep_merge_dict(evo_kwargs, evo_overrides)
        if evo_kwargs.get("llm_models") is None:
            evo_kwargs.pop("llm_models", None)

        evo_config = self._se_EvolutionConfig(**evo_kwargs)

        job_kwargs: dict[str, Any] = {
            "eval_program_path": str(evaluator_file),
            "extra_cmd_args": {"task_name": task.NAME},
        }
        if job_type == "local":
            job_kwargs["time"] = _hms_from_seconds(evaluator_timeout_s)
        _deep_merge_dict(job_kwargs, job_overrides)
        job_config = self._se_LocalJobConfig(**job_kwargs)

        db_kwargs: dict[str, Any] = {}
        archive_size = getattr(algo_cfg, "archive_size", None)
        if archive_size is not None:
            db_kwargs["archive_size"] = int(archive_size)
        _deep_merge_dict(db_kwargs, db_overrides)
        db_config = self._se_DatabaseConfig(**db_kwargs)

        # Run evolution (sync upstream API; keep adapter async-friendly).
        # NOTE: Shinka's SQLite connection is created during runner init; create+run inside the
        # same worker thread to avoid "SQLite objects created in a thread" errors.
        def _run_sync() -> None:
            runner = self._se_EvolutionRunner(
                evo_config=evo_config,
                job_config=job_config,
                db_config=db_config,
                verbose=verbose,
            )
            runner.run()

        await asyncio.to_thread(_run_sync)
        output_root_path = shinka_dir

        best_metrics, best_dir = _extract_best_metrics(output_root_path)
        best_info_path = shinka_dir / "best" / "best_program_info.json"
        if best_metrics is not None:
            score = best_metrics.get("combined_score", None)
            print(f"Best score: {score}")

            best_program_path = None
            if best_dir is not None:
                candidates = [
                    best_dir / "main.py",
                    best_dir.parent / "main.py",
                ]
                for candidate in candidates:
                    if candidate.is_file():
                        best_program_path = candidate
                        break

            (shinka_dir / "best").mkdir(parents=True, exist_ok=True)
            info = {
                "metrics": best_metrics,
                "results_dir": str(best_dir) if best_dir is not None else "",
                "program_path": str(best_program_path) if best_program_path else "",
                "shinka_output_root": str(output_root_path),
            }
            best_info_path.write_text(
                _safe_json(info),
                encoding="utf-8",
            )

        print(f"Saved: {output_root_path}")
        if best_info_path.is_file():
            print(f"Saved: {best_info_path}")
