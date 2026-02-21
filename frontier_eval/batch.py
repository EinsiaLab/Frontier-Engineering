from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from frontier_eval.env import find_dotenv, load_dotenv


def _safe_slug(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe or "item"


def _is_repo_root(path: Path) -> bool:
    if not (path / "frontier_eval").is_dir():
        return False
    if (path / "benchmarks").is_dir():
        return True
    return (path / "Astrodynamics").is_dir() and (path / "ElectronicDesignAutomation").is_dir()


def _normalize_overrides(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            items.append(f"{k}={v}")
        return items
    raise TypeError(f"overrides must be a list or dict, got {type(value)}")


def _find_repo_root(start: Path) -> Path:
    start = start.expanduser().resolve()

    if "FRONTIER_ENGINEERING_ROOT" in os.environ:
        return Path(os.environ["FRONTIER_ENGINEERING_ROOT"]).expanduser().resolve()

    for parent in [start, *start.parents]:
        if _is_repo_root(parent):
            return parent
    raise RuntimeError(
        "Cannot locate repo root. Set `FRONTIER_ENGINEERING_ROOT` or run inside the repo."
    )


def _load_dotenv_if_any(repo_root: Path) -> None:
    dotenv_path = find_dotenv(repo_root)
    if dotenv_path is not None:
        load_dotenv(dotenv_path, override=False)


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    overrides: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LlmSpec:
    name: str
    api_base: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    overrides: list[str] = field(default_factory=list)
    llm_config: str | None = None


@dataclass(frozen=True)
class RunSpec:
    task: str
    algorithm: AlgorithmSpec
    llm: LlmSpec | None
    cwd: Path
    output_dir: Path
    cmd: list[str]
    env: dict[str, str]


def _as_str_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError(f"`{field_name}` must be a list, got {type(value)}")
    return [str(x) for x in value]


def _parse_algorithms(raw_list: list[Any]) -> list[AlgorithmSpec]:
    algorithms: list[AlgorithmSpec] = []
    for item in raw_list:
        if isinstance(item, str):
            algorithms.append(AlgorithmSpec(name=item))
            continue
        if isinstance(item, dict):
            name = item.get("name")
            if not name:
                raise ValueError("algorithm entry missing `name`")
            overrides = _normalize_overrides(item.get("overrides"))
            algorithms.append(AlgorithmSpec(name=str(name), overrides=overrides))
            continue
        raise TypeError(f"algorithm entry must be str or dict, got {type(item)}")
    return algorithms


def _parse_llms(raw_list: list[Any], *, default_llm_config: str) -> list[LlmSpec]:
    llms: list[LlmSpec] = []
    for item in raw_list:
        if isinstance(item, dict):
            name = item.get("name")
            if not name:
                raise ValueError("llm entry missing `name`")
            env_map = item.get("env") or {}
            if not isinstance(env_map, dict):
                raise TypeError(f"llm.env must be a dict, got {type(env_map)}")
            overrides = _normalize_overrides(item.get("overrides"))
            llms.append(
                LlmSpec(
                    name=str(name),
                    api_base=str(item["api_base"]) if item.get("api_base") is not None else None,
                    model=str(item["model"]) if item.get("model") is not None else None,
                    api_key_env=str(item["api_key_env"])
                    if item.get("api_key_env") is not None
                    else None,
                    env={str(k): str(v) for k, v in env_map.items()},
                    overrides=overrides,
                    llm_config=str(item.get("llm_config") or default_llm_config),
                )
            )
            continue
        raise TypeError(f"llm entry must be dict, got {type(item)}")
    return llms


def _require_hydra_group(repo_root: Path, *, group: str, name: str) -> None:
    config_path = repo_root / "frontier_eval" / "conf" / group / f"{name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing Hydra config `{group}/{name}.yaml` at {config_path}. "
            f"Create it to enable `python -m frontier_eval {group}={name}`."
        )


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 10_000):
        candidate = Path(f"{path}__{i}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot find unique directory name for {path}")


def _build_runs(
    *,
    repo_root: Path,
    tasks: list[str],
    algorithms: list[AlgorithmSpec],
    llms: list[LlmSpec],
    llm_default_config: str,
    common_overrides: list[str],
    base_dir: Path,
    batch_id: str,
    python_exe: str,
    extra_overrides: list[str],
) -> list[RunSpec]:
    base_dir = base_dir.expanduser()
    batch_root = (repo_root / base_dir / batch_id).resolve()
    batch_root.mkdir(parents=True, exist_ok=True)

    for t in tasks:
        _require_hydra_group(repo_root, group="task", name=t)
    for a in algorithms:
        _require_hydra_group(repo_root, group="algorithm", name=a.name)
    _require_hydra_group(repo_root, group="llm", name=llm_default_config)

    runs: list[RunSpec] = []
    for task in tasks:
        for algo in algorithms:
            for llm in llms or [None]:
                llm_name = "default_llm" if llm is None else llm.name
                run_dir = batch_root / _safe_slug(task) / _safe_slug(algo.name) / _safe_slug(llm_name)
                run_dir = _unique_dir(run_dir)

                overrides: list[str] = [
                    f"task={task}",
                    f"algorithm={algo.name}",
                    f"llm={llm_default_config if llm is None else (llm.llm_config or llm_default_config)}",
                    f"run.output_dir={run_dir.as_posix()}",
                ]
                overrides.extend(common_overrides)
                overrides.extend(algo.overrides)
                if llm is not None:
                    overrides.extend(llm.overrides)
                overrides.extend(extra_overrides)

                env = os.environ.copy()
                env.setdefault("PYTHONUNBUFFERED", "1")
                env.setdefault("FRONTIER_ENGINEERING_ROOT", str(repo_root))
                env["FRONTIER_EVAL_TASK_NAME"] = task
                if llm is not None:
                    if llm.api_base:
                        env["OPENAI_API_BASE"] = llm.api_base
                    if llm.model:
                        env["OPENAI_MODEL"] = llm.model
                    if llm.api_key_env:
                        key_value = os.environ.get(llm.api_key_env, "")
                        if key_value:
                            env["OPENAI_API_KEY"] = key_value
                        else:
                            env.pop("OPENAI_API_KEY", None)
                    for k, v in llm.env.items():
                        env[str(k)] = str(v)

                cmd = [python_exe, "-m", "frontier_eval", *overrides]
                runs.append(
                    RunSpec(
                        task=task,
                        algorithm=algo,
                        llm=llm,
                        cwd=repo_root,
                        output_dir=run_dir,
                        cmd=cmd,
                        env=env,
                    )
                )
    return runs


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _env_snapshot(env: dict[str, str]) -> dict[str, str]:
    keys = [
        "FRONTIER_ENGINEERING_ROOT",
        "FRONTIER_EVAL_TASK_NAME",
        "OPENAI_API_BASE",
        "OPENAI_MODEL",
        "OPENAI_API_KEY",
        "PYTHONUNBUFFERED",
    ]
    snap: dict[str, str] = {}
    for k in keys:
        if k not in env:
            continue
        if k.endswith("_API_KEY"):
            snap[k] = "***REDACTED***" if env.get(k) else ""
        else:
            snap[k] = env.get(k, "")
    return snap


def _extract_openevolve_best_metrics(output_dir: Path) -> dict[str, Any] | None:
    info_path = output_dir / "openevolve" / "best" / "best_program_info.json"
    if not info_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    metrics = info.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return None


def _extract_shinkaevolve_best_metrics(output_dir: Path) -> dict[str, Any] | None:
    info_path = output_dir / "shinkaevolve" / "best" / "best_program_info.json"
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            info = None
        if isinstance(info, dict) and isinstance(info.get("metrics"), dict):
            return info["metrics"]

    root = output_dir / "shinkaevolve"
    if not root.is_dir():
        return None

    best_correct: tuple[float, dict[str, Any]] | None = None
    best_any: tuple[float, dict[str, Any]] | None = None

    for metrics_path in root.rglob("metrics.json"):
        try:
            metrics_raw = json.loads(metrics_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(metrics_raw, dict):
            continue
        score = metrics_raw.get("combined_score", None)
        try:
            score_f = float(score)
        except Exception:
            continue

        correct = True
        correct_path = metrics_path.parent / "correct.json"
        if correct_path.is_file():
            try:
                correct_raw = json.loads(
                    correct_path.read_text(encoding="utf-8", errors="replace")
                )
                if isinstance(correct_raw, dict):
                    correct = bool(correct_raw.get("correct"))
            except Exception:
                correct = True

        if best_any is None or score_f > best_any[0]:
            best_any = (score_f, metrics_raw)
        if correct and (best_correct is None or score_f > best_correct[0]):
            best_correct = (score_f, metrics_raw)

    if best_correct is not None:
        return best_correct[1]
    if best_any is not None:
        return best_any[1]
    return None


async def _run_one(run: RunSpec) -> dict[str, Any]:
    start = time.time()
    run.output_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = run.output_dir / "launcher_stdout.txt"
    stderr_path = run.output_dir / "launcher_stderr.txt"
    meta_path = run.output_dir / "launcher_run.json"

    meta = {
        "task": run.task,
        "algorithm": run.algorithm.name,
        "llm": None if run.llm is None else run.llm.name,
        "cmd": shlex.join(run.cmd),
        "output_dir": str(run.output_dir),
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "env": _env_snapshot(run.env),
    }
    _write_json(meta_path, meta)

    with stdout_path.open("wb") as f_out, stderr_path.open("wb") as f_err:
        proc = await asyncio.create_subprocess_exec(
            *run.cmd,
            cwd=str(run.cwd),
            env=run.env,
            stdout=f_out,
            stderr=f_err,
        )
        returncode = await proc.wait()

    elapsed_s = time.time() - start
    metrics: dict[str, Any] | None = None
    if run.algorithm.name == "openevolve":
        metrics = _extract_openevolve_best_metrics(run.output_dir)
    elif run.algorithm.name == "shinkaevolve":
        metrics = _extract_shinkaevolve_best_metrics(run.output_dir)
    record: dict[str, Any] = {
        "task": run.task,
        "algorithm": run.algorithm.name,
        "llm": None if run.llm is None else run.llm.name,
        "model": None if run.llm is None else run.llm.model,
        "api_base": None if run.llm is None else run.llm.api_base,
        "output_dir": str(run.output_dir),
        "cmd": shlex.join(run.cmd),
        "returncode": int(returncode),
        "elapsed_s": float(elapsed_s),
        "best_metrics": metrics,
    }
    _write_json(run.output_dir / "launcher_result.json", record)
    return record


async def _run_all(
    runs: list[RunSpec],
    *,
    max_parallel: int,
    summary_path: Path,
    fail_fast: bool,
    dry_run: bool,
) -> int:
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        for run in runs:
            print(shlex.join(run.cmd))
        return 0

    sem = asyncio.Semaphore(max(1, int(max_parallel)))
    pending: set[asyncio.Task[dict[str, Any]]] = set()

    async def _wrapped(run: RunSpec) -> dict[str, Any]:
        async with sem:
            return await _run_one(run)

    for run in runs:
        pending.add(asyncio.create_task(_wrapped(run)))

    exit_code = 0
    try:
        for fut in asyncio.as_completed(pending):
            record = await fut
            with summary_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            if int(record.get("returncode", 1)) != 0:
                exit_code = 1
                if fail_fast:
                    for t in pending:
                        if not t.done():
                            t.cancel()
                    break
    finally:
        await asyncio.gather(*pending, return_exceptions=True)

    return exit_code


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Batch runner for Frontier Eval (task × algorithm × llm)."
    )
    parser.add_argument(
        "--matrix",
        type=str,
        required=True,
        help="Path to a matrix YAML file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Override max parallel runs (default: matrix.run.max_parallel or 1).",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Override base output dir (default: matrix.run.base_dir or runs/batch).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after first failure.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable for child runs (default: current interpreter).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override for every run (repeatable).",
    )

    args = parser.parse_args(argv)

    repo_root = _find_repo_root(Path.cwd())
    _load_dotenv_if_any(repo_root)

    matrix_path = Path(args.matrix).expanduser().resolve()
    if not matrix_path.is_file():
        raise FileNotFoundError(f"matrix file not found: {matrix_path}")

    matrix_cfg = OmegaConf.load(str(matrix_path))
    matrix = OmegaConf.to_container(matrix_cfg, resolve=True)
    if not isinstance(matrix, dict):
        raise TypeError("matrix file must define a mapping at top-level")

    tasks = _as_str_list(matrix.get("tasks"), field_name="tasks")
    if not tasks:
        raise ValueError("matrix.tasks is required and must be non-empty")

    raw_algos = matrix.get("algorithms")
    if not isinstance(raw_algos, list) or not raw_algos:
        raise ValueError("matrix.algorithms is required and must be non-empty")
    algorithms = _parse_algorithms(raw_algos)

    raw_llms = matrix.get("llms", [])
    if raw_llms and not isinstance(raw_llms, list):
        raise TypeError("matrix.llms must be a list")

    run_cfg = matrix.get("run") or {}
    if not isinstance(run_cfg, dict):
        raise TypeError("matrix.run must be a mapping")

    llm_default_config = str(matrix.get("llm_config") or "openai_compatible")
    llms = _parse_llms(raw_llms, default_llm_config=llm_default_config)

    common_overrides = _normalize_overrides(matrix.get("common_overrides"))
    base_dir = Path(str(args.base_dir or run_cfg.get("base_dir") or "runs/batch"))
    max_parallel = int(args.max_parallel or run_cfg.get("max_parallel") or 1)
    fail_fast = bool(args.fail_fast or run_cfg.get("fail_fast") or False)

    batch_name = str(run_cfg.get("name") or "batch")
    batch_id = f"{_safe_slug(batch_name)}__{datetime.now().strftime('%Y%m%d_%H%M%S')}__{uuid.uuid4().hex[:8]}"
    extra_overrides = [str(x) for x in (args.override or [])]

    batch_root = (repo_root / base_dir / batch_id).resolve()
    batch_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(matrix_cfg, str(batch_root / "matrix_resolved.yaml"))

    runs = _build_runs(
        repo_root=repo_root,
        tasks=tasks,
        algorithms=algorithms,
        llms=llms,
        llm_default_config=llm_default_config,
        common_overrides=common_overrides,
        base_dir=base_dir,
        batch_id=batch_id,
        python_exe=str(args.python),
        extra_overrides=extra_overrides,
    )

    print(f"Batch: {batch_root}")
    print(f"Runs: {len(runs)} (max_parallel={max_parallel})")

    summary_path = batch_root / "summary.jsonl"
    exit_code = asyncio.run(
        _run_all(
            runs,
            max_parallel=max_parallel,
            summary_path=summary_path,
            fail_fast=fail_fast,
            dry_run=bool(args.dry_run),
        )
    )
    if not args.dry_run:
        print(f"Summary: {summary_path}")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
