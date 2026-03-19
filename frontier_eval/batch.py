from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
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
class TaskSpec:
    name: str
    label: str
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
    task_config: str
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


def _parse_csv_args(values: list[str]) -> list[str]:
    items: list[str] = []
    for raw in values or []:
        for part in str(raw).split(","):
            item = part.strip()
            if item:
                items.append(item)
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _filter_summary_jsonl_in_place(summary_path: Path, *, exclude_tasks: set[str]) -> None:
    if not summary_path.is_file():
        return
    kept_lines: list[str] = []
    with summary_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                kept_lines.append(line)
                continue
            if not isinstance(obj, dict):
                kept_lines.append(line)
                continue
            task = obj.get("task", None)
            if isinstance(task, str) and task in exclude_tasks:
                continue
            kept_lines.append(line)

    tmp_path = summary_path.with_name(summary_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line if line.endswith("\n") else (line + "\n"))
    tmp_path.replace(summary_path)


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


def _parse_tasks(raw_list: list[Any]) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    seen_labels: set[str] = set()

    for item in raw_list:
        if isinstance(item, str):
            spec = TaskSpec(name=item, label=item)
        elif isinstance(item, dict):
            name = item.get("name")
            if not name:
                raise ValueError("task entry missing `name`")
            label = str(item.get("label") or name)
            overrides = _normalize_overrides(item.get("overrides"))
            spec = TaskSpec(name=str(name), label=label, overrides=overrides)
        else:
            raise TypeError(f"task entry must be str or dict, got {type(item)}")

        if spec.label in seen_labels:
            raise ValueError(f"duplicate task label in matrix.tasks: {spec.label}")
        seen_labels.add(spec.label)
        tasks.append(spec)

    return tasks


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
    batch_root: Path,
    tasks: list[TaskSpec],
    algorithms: list[AlgorithmSpec],
    llms: list[LlmSpec],
    llm_default_config: str,
    common_overrides: list[str],
    python_exe: str,
    extra_overrides: list[str],
    unique_dirs: bool = True,
) -> list[RunSpec]:
    batch_root = batch_root.expanduser().resolve()
    batch_root.mkdir(parents=True, exist_ok=True)

    for task_spec in tasks:
        _require_hydra_group(repo_root, group="task", name=task_spec.name)
    for a in algorithms:
        _require_hydra_group(repo_root, group="algorithm", name=a.name)
    _require_hydra_group(repo_root, group="llm", name=llm_default_config)

    runs: list[RunSpec] = []
    for task_spec in tasks:
        for algo in algorithms:
            for llm in llms or [None]:
                llm_name = "default_llm" if llm is None else llm.name
                run_dir = (
                    batch_root
                    / _safe_slug(task_spec.label)
                    / _safe_slug(algo.name)
                    / _safe_slug(llm_name)
                )
                if unique_dirs:
                    run_dir = _unique_dir(run_dir)

                overrides: list[str] = [
                    f"task={task_spec.name}",
                    f"algorithm={algo.name}",
                    f"llm={llm_default_config if llm is None else (llm.llm_config or llm_default_config)}",
                    f"run.output_dir={run_dir.as_posix()}",
                ]
                overrides.extend(task_spec.overrides)
                overrides.extend(common_overrides)
                overrides.extend(algo.overrides)
                if llm is not None:
                    overrides.extend(llm.overrides)
                overrides.extend(extra_overrides)

                env = os.environ.copy()
                env.setdefault("PYTHONUNBUFFERED", "1")
                env.setdefault("FRONTIER_ENGINEERING_ROOT", str(repo_root))
                env["FRONTIER_EVAL_TASK_NAME"] = task_spec.name
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
                        task=task_spec.label,
                        task_config=task_spec.name,
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


def _extract_abmcts_best_metrics(output_dir: Path) -> dict[str, Any] | None:
    info_path = output_dir / "abmcts" / "best" / "best_program_info.json"
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


async def _run_one(run: RunSpec) -> dict[str, Any]:
    start = time.time()
    run.output_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = run.output_dir / "launcher_stdout.txt"
    stderr_path = run.output_dir / "launcher_stderr.txt"
    meta_path = run.output_dir / "launcher_run.json"

    meta = {
        "task": run.task,
        "task_config": run.task_config,
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
    elif run.algorithm.name == "abmcts":
        metrics = _extract_abmcts_best_metrics(run.output_dir)
    record: dict[str, Any] = {
        "task": run.task,
        "task_config": run.task_config,
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
    if isinstance(metrics, dict):
        record["combined_score"] = metrics.get("combined_score")
        record["valid"] = metrics.get("valid")
        record["timeout"] = metrics.get("timeout")
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
    parser.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Rerun in an existing batch directory (instead of creating a new one). "
            "Selected task directories will be deleted before running."
        ),
    )
    parser.add_argument(
        "--batch-root",
        type=str,
        default=None,
        help=(
            "Batch root directory to use with --in-place (default: parent directory of --matrix)."
        ),
    )
    parser.add_argument(
        "--tasks",
        action="append",
        default=[],
        help="Only run these task labels (repeatable; supports comma-separated).",
    )
    parser.add_argument(
        "--algorithms",
        action="append",
        default=[],
        help="Only run these algorithm names (repeatable; supports comma-separated).",
    )
    parser.add_argument(
        "--exclude-tasks",
        action="append",
        default=[],
        help="Exclude these task labels (repeatable; supports comma-separated).",
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

    raw_tasks = matrix.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("matrix.tasks is required and must be non-empty")
    matrix_tasks = _parse_tasks(raw_tasks)

    include_tasks = _parse_csv_args(args.tasks)
    exclude_tasks = _parse_csv_args(args.exclude_tasks)
    matrix_task_set = {task.label for task in matrix_tasks}

    tasks = matrix_tasks
    if include_tasks:
        unknown = [t for t in include_tasks if t not in matrix_task_set]
        if unknown:
            raise ValueError(f"--tasks contains task(s) not present in matrix: {unknown}")
        include_set = set(include_tasks)
        tasks = [task for task in tasks if task.label in include_set]
    if exclude_tasks:
        unknown = [t for t in exclude_tasks if t not in matrix_task_set]
        if unknown:
            raise ValueError(f"--exclude-tasks contains task(s) not present in matrix: {unknown}")
        exclude_set = set(exclude_tasks)
        tasks = [task for task in tasks if task.label not in exclude_set]
    if not tasks:
        raise ValueError("No tasks selected after applying --tasks/--exclude-tasks filters")

    raw_algos = matrix.get("algorithms")
    if not isinstance(raw_algos, list) or not raw_algos:
        raise ValueError("matrix.algorithms is required and must be non-empty")
    algorithms = _parse_algorithms(raw_algos)
    include_algorithms = _parse_csv_args(args.algorithms)
    matrix_algorithm_set = {algo.name for algo in algorithms}
    if include_algorithms:
        unknown = [name for name in include_algorithms if name not in matrix_algorithm_set]
        if unknown:
            raise ValueError(
                f"--algorithms contains algorithm(s) not present in matrix: {unknown}"
            )
        include_algo_set = set(include_algorithms)
        algorithms = [algo for algo in algorithms if algo.name in include_algo_set]
    if not algorithms:
        raise ValueError("No algorithms selected after applying --algorithms filters")

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

    extra_overrides = [str(x) for x in (args.override or [])]

    if args.in_place:
        if not include_tasks:
            raise ValueError("--in-place requires --tasks to explicitly list task(s) to rerun")
        batch_root = Path(args.batch_root).expanduser().resolve() if args.batch_root else matrix_path.parent
        batch_root = batch_root.expanduser().resolve()
        if not batch_root.is_dir():
            raise FileNotFoundError(f"batch root directory not found: {batch_root}")
        try:
            batch_root.resolve().relative_to(repo_root.resolve())
        except Exception as e:
            raise ValueError(f"--batch-root must be inside repo root: {repo_root}") from e

        if not args.dry_run:
            for task in tasks:
                task_dir = (batch_root / _safe_slug(task.label)).resolve()
                try:
                    task_dir.relative_to(batch_root.resolve())
                except Exception as e:
                    raise ValueError(f"Refusing to delete outside batch root: {task_dir}") from e
                if task_dir.is_symlink():
                    raise ValueError(f"Refusing to delete symlinked task dir: {task_dir}")
                if task_dir.exists() and not task_dir.is_dir():
                    raise ValueError(f"Expected task path to be a directory: {task_dir}")
                if task_dir.is_dir():
                    shutil.rmtree(task_dir)

            _filter_summary_jsonl_in_place(
                batch_root / "summary.jsonl",
                exclude_tasks={task.label for task in tasks},
            )
        unique_dirs = False
    else:
        batch_name = str(run_cfg.get("name") or "batch")
        batch_id = f"{_safe_slug(batch_name)}__{datetime.now().strftime('%Y%m%d_%H%M%S')}__{uuid.uuid4().hex[:8]}"
        batch_root = (repo_root / base_dir / batch_id).resolve()
        batch_root.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(matrix_cfg, str(batch_root / "matrix_resolved.yaml"))
        unique_dirs = True

    runs = _build_runs(
        repo_root=repo_root,
        batch_root=batch_root,
        tasks=tasks,
        algorithms=algorithms,
        llms=llms,
        llm_default_config=llm_default_config,
        common_overrides=common_overrides,
        python_exe=str(args.python),
        extra_overrides=extra_overrides,
        unique_dirs=unique_dirs,
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
