from __future__ import annotations

"""OpenEvolve algorithm adapter for Frontier Eval."""

import json
import os
import re
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from frontier_eval.algorithms.base import Algorithm
from frontier_eval.tasks.base import Task


def _safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return safe or "artifact"


def _safe_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
            continue
        base[key] = value
    return base


def _drop_none(value: Any) -> Any:
    """
    Recursively drop `None` values from dict/list structures.

    This is needed because openevolve==0.2.26 uses non-Optional dataclass types with
    default `None` (e.g. `Config.language: str = None`), and `Config.from_dict()`
    rejects explicit `None` values via dacite type checking.
    """
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value if v is not None]
    return value


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


def _export_history(controller, history_dir: Path) -> None:
    history_dir.mkdir(parents=True, exist_ok=True)

    programs = list(controller.database.programs.values())
    programs.sort(
        key=lambda p: (
            int(getattr(p, "iteration_found", 0) or 0),
            float(getattr(p, "timestamp", 0.0) or 0.0),
        )
    )

    index_path = history_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for program in programs:
            iter_num = int(getattr(program, "iteration_found", 0) or 0)
            program_dir = history_dir / f"iter_{iter_num:06d}__{program.id}"
            program_dir.mkdir(parents=True, exist_ok=True)

            code_path = program_dir / f"program{controller.file_extension}"
            code_path.write_text(program.code, encoding="utf-8", errors="replace")

            (program_dir / "metrics.json").write_text(
                _safe_json(program.metrics or {}),
                encoding="utf-8",
            )

            meta = {
                "id": program.id,
                "parent_id": program.parent_id,
                "generation": program.generation,
                "timestamp": program.timestamp,
                "iteration_found": program.iteration_found,
                "language": program.language,
                "changes_description": getattr(program, "changes_description", ""),
                "metadata": program.metadata,
                "prompts": getattr(program, "prompts", None),
            }
            (program_dir / "meta.json").write_text(
                _safe_json(meta),
                encoding="utf-8",
            )

            artifacts = controller.database.get_artifacts(program.id)
            if artifacts:
                artifacts_dir = program_dir / "artifacts"
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                manifest: dict[str, str] = {}
                used_names: set[str] = set()
                for key, value in artifacts.items():
                    base_name = _safe_filename(str(key))
                    name = base_name
                    i = 1
                    while name in used_names:
                        i += 1
                        name = f"{base_name}__{i}"
                    used_names.add(name)
                    manifest[str(key)] = name
                    p = artifacts_dir / name
                    if isinstance(value, bytes):
                        p.write_bytes(value)
                    elif isinstance(value, (dict, list)):
                        p.write_text(_safe_json(value), encoding="utf-8", errors="replace")
                    else:
                        p.write_text(str(value), encoding="utf-8", errors="replace")
                (artifacts_dir / "manifest.json").write_text(
                    _safe_json(manifest),
                    encoding="utf-8",
                )

            record = {
                "iteration": iter_num,
                "id": program.id,
                "parent_id": program.parent_id,
                "generation": program.generation,
                "metrics": program.metrics or {},
            }
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


class OpenEvolveAlgorithm(Algorithm):
    NAME = "openevolve"

    def __init__(self, cfg: DictConfig, repo_root: Path):
        super().__init__(cfg=cfg, repo_root=repo_root)

        try:
            from openevolve import Config, OpenEvolve
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "OpenEvolve is not importable. Install it first (recommended inside a venv), e.g.\n"
                "  pip install -e third_party/openevolve\n"
                "or\n"
                "  pip install openevolve\n"
            ) from e

        self._oe_Config = Config
        self._oe_OpenEvolve = OpenEvolve

    async def run(self, task: Task) -> None:
        algo_cfg = self.cfg.algorithm
        llm_cfg = self.cfg.llm

        output_dir = Path(str(self.cfg.run.output_dir)).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        iterations = int(algo_cfg.iterations)
        openevolve_dir = (output_dir / "openevolve").resolve()
        openevolve_dir.mkdir(parents=True, exist_ok=True)
        db_dir = openevolve_dir / "db"
        history_dir = openevolve_dir / "history"

        save_db = bool(algo_cfg.get("save_db", True))
        export_history = bool(algo_cfg.get("export_history", True))
        trace_cfg = algo_cfg.get("trace", None)
        if trace_cfg is None:
            trace_cfg = {}
        trace_enabled = bool(getattr(trace_cfg, "enabled", trace_cfg.get("enabled", False)))
        trace_format = str(getattr(trace_cfg, "format", trace_cfg.get("format", "jsonl")))
        trace_path = openevolve_dir / f"evolution_trace.{trace_format}"
        if trace_format == "jsonl":
            trace_path = openevolve_dir / "evolution_trace.jsonl"
        elif trace_format in ("hdf5", "h5"):
            trace_path = openevolve_dir / "evolution_trace.hdf5"

        oe_config_path = algo_cfg.get("oe_config_path") or algo_cfg.get("openevolve_config_path")
        if oe_config_path:
            config_path = Path(str(oe_config_path)).expanduser()
            if not config_path.is_absolute():
                config_path = (self.repo_root / config_path).resolve()
            config = self._oe_Config.from_yaml(config_path)
        else:
            config = self._oe_Config()

        reserved_keys = {
            "name",
            "iterations",
            "checkpoint_interval",
            "max_code_length",
            "save_db",
            "export_history",
            "trace",
            "oe_config_path",
            "openevolve_config_path",
            "oe",
        }
        implicit_overrides: dict[str, Any] = {}
        algo_view = OmegaConf.to_container(algo_cfg, resolve=True)
        if isinstance(algo_view, dict):
            implicit_overrides = {k: v for k, v in algo_view.items() if k not in reserved_keys}

        explicit_overrides = _as_plain_mapping(algo_cfg.get("oe"))
        oe_overrides = _deep_merge_dict(implicit_overrides, explicit_overrides)
        if oe_overrides:
            merged = config.to_dict()
            _deep_merge_dict(merged, oe_overrides)
            config = self._oe_Config.from_dict(_drop_none(merged))

        config.max_iterations = iterations
        if algo_cfg.get("checkpoint_interval") is not None:
            config.checkpoint_interval = int(algo_cfg.checkpoint_interval)
        if algo_cfg.get("max_code_length") is not None:
            config.max_code_length = int(algo_cfg.max_code_length)

        cascade_overridden = False
        evaluator_overrides = oe_overrides.get("evaluator")
        if isinstance(evaluator_overrides, dict) and "cascade_evaluation" in evaluator_overrides:
            cascade_overridden = True
        if not oe_config_path and not cascade_overridden:
            config.evaluator.cascade_evaluation = False
        config.database.db_path = str(db_dir)

        # Write per-iteration traces (code + metrics + artifacts)
        config.evolution_trace.enabled = trace_enabled
        config.evolution_trace.format = trace_format
        config.evolution_trace.include_code = bool(
            getattr(trace_cfg, "include_code", trace_cfg.get("include_code", True))
        )
        config.evolution_trace.include_prompts = bool(
            getattr(trace_cfg, "include_prompts", trace_cfg.get("include_prompts", True))
        )
        config.evolution_trace.buffer_size = int(
            getattr(trace_cfg, "buffer_size", trace_cfg.get("buffer_size", 1))
        )
        config.evolution_trace.output_path = str(trace_path)

        api_base = str(getattr(llm_cfg, "api_base", "") or "")
        if not api_base:
            api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

        model = str(getattr(llm_cfg, "model", "") or "")
        if not model:
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        api_key = str(getattr(llm_cfg, "api_key", "") or "")
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY", "") or ""

        config.llm.api_base = api_base
        if api_key:
            config.llm.api_key = api_key

        if not getattr(config.llm, "models", None):
            config.llm.primary_model = model
            config.llm.primary_model_weight = 1.0
            config.llm.rebuild_models()

        config.llm.temperature = float(getattr(llm_cfg, "temperature", 0.7))
        config.llm.max_tokens = int(getattr(llm_cfg, "max_tokens", 4096))
        config.llm.timeout = int(getattr(llm_cfg, "timeout", 60))
        config.llm.retries = int(getattr(llm_cfg, "retries", 3))
        config.llm.retry_delay = int(getattr(llm_cfg, "retry_delay", 5))

        config.llm.update_model_params(
            {
                "api_base": config.llm.api_base,
                "api_key": getattr(config.llm, "api_key", None),
                "temperature": config.llm.temperature,
                "top_p": getattr(config.llm, "top_p", None),
                "max_tokens": config.llm.max_tokens,
                "timeout": config.llm.timeout,
                "retries": config.llm.retries,
                "retry_delay": config.llm.retry_delay,
                "random_seed": getattr(config.llm, "random_seed", None),
                "reasoning_effort": getattr(config.llm, "reasoning_effort", None),
                "manual_mode": getattr(config.llm, "manual_mode", False),
            },
            overwrite=False,
        )

        manual_mode = bool(getattr(config.llm, "manual_mode", False))
        if not manual_mode:
            models = list(getattr(config.llm, "models", []) or [])
            models += list(getattr(config.llm, "evaluator_models", []) or [])
            has_any_api_key = bool(getattr(config.llm, "api_key", None)) or any(
                getattr(entry, "api_key", None) for entry in models
            )
            if not has_any_api_key:
                if iterations <= 0:
                    dummy = "DUMMY_API_KEY_FOR_ZERO_ITERATIONS"
                    config.llm.api_key = dummy
                    config.llm.update_model_params({"api_key": dummy}, overwrite=False)
                else:
                    raise RuntimeError(
                        "Missing API key for OpenEvolve. Set `OPENAI_API_KEY` or `llm.api_key` "
                        "when `algorithm.iterations > 0`."
                    )

        initial_program = task.initial_program_path()
        evaluator_file = (
            self.repo_root
            / "frontier_eval"
            / "algorithms"
            / "openevolve"
            / "openevolve_entrypoint.py"
        ).resolve()
        task_cfg_view = OmegaConf.to_container(getattr(self.cfg, "task", None), resolve=True)
        task_cfg_payload: dict[str, Any] = task_cfg_view if isinstance(task_cfg_view, dict) else {}
        task_cfg_payload = dict(task_cfg_payload)
        task_cfg_payload.setdefault("name", task.NAME)
        os.environ["FRONTIER_EVAL_TASK_NAME"] = task.NAME
        os.environ["FRONTIER_EVAL_TASK_CFG_JSON"] = json.dumps(task_cfg_payload, ensure_ascii=False)
        os.environ["FRONTIER_EVAL_EVALUATOR_TIMEOUT_S"] = str(getattr(config.evaluator, "timeout", 300))
        os.environ.setdefault("FRONTIER_ENGINEERING_ROOT", str(self.repo_root))

        controller = self._oe_OpenEvolve(
            initial_program_path=str(initial_program),
            evaluation_file=str(evaluator_file),
            config=config,
            output_dir=str(openevolve_dir),
        )
        best = await controller.run(iterations=iterations)

        if not best:
            raise RuntimeError("OpenEvolve returned no best program")

        if save_db or export_history:
            # OpenEvolve's initial program evaluation stores artifacts in a pending queue, but does
            # not attach them to the initial Program. Persist them here for complete history.
            initial_candidates = [p for p in controller.database.programs.values() if p.parent_id is None]
            initial_program_obj = None
            if len(initial_candidates) == 1:
                initial_program_obj = initial_candidates[0]
            else:
                for p in initial_candidates:
                    if p.code == controller.initial_program_code:
                        initial_program_obj = p
                        break

            if initial_program_obj is not None:
                pending = controller.evaluator.get_pending_artifacts(initial_program_obj.id)
                if pending:
                    controller.database.store_artifacts(initial_program_obj.id, pending)

        if save_db:
            # Persist full database (all programs + metrics + artifacts)
            controller.database.save(str(db_dir), iteration=controller.database.last_iteration)

        if export_history:
            # Export a human-friendly history folder (one dir per program)
            _export_history(controller, history_dir)

        metrics = best.metrics or {}
        score = metrics.get("combined_score", metrics.get("score", None))
        print(f"Best score: {score}")
        if save_db:
            print(f"Saved: {db_dir}")
        if export_history:
            print(f"Saved: {history_dir}")
        if trace_enabled:
            actual_trace_path = getattr(getattr(controller, "evolution_tracer", None), "output_path", None)
            actual_trace_path = Path(actual_trace_path) if actual_trace_path else trace_path
            if actual_trace_path.exists():
                print(f"Saved: {actual_trace_path}")
            else:
                print(f"Trace: {actual_trace_path} (no traces written)")
