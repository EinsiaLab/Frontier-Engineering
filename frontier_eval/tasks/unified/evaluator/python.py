from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from ..spec import UnifiedTaskSpec


def _tail(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _truncate_middle(text: str, limit: int = 200_000) -> str:
    if len(text) <= limit:
        return text
    keep = max(0, (limit - 128) // 2)
    omitted = len(text) - (2 * keep)
    return text[:keep] + f"\n\n[... truncated {omitted} chars ...]\n\n" + text[-keep:]


def _remaining_timeout(deadline_s: float) -> float:
    return max(1.0, float(deadline_s - time.time()))


def _safe_slug(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe or "benchmark"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _read_json(path: Path) -> Any | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            return None
    return None


def _extract_numeric_metrics(raw: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    metrics: dict[str, float] = {}
    non_numeric: dict[str, Any] = {}
    for key, value in raw.items():
        metric_v = _maybe_float(value)
        if metric_v is None:
            non_numeric[str(key)] = value
            continue
        metrics[str(key)] = float(metric_v)
    return metrics, non_numeric


def _parse_last_json_dict(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    for raw in reversed(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _fingerprint_path(path: Path) -> str:
    if not path.exists():
        return "__MISSING__"
    if path.is_file():
        return f"file:{_hash_file(path)}"

    if path.is_dir():
        h = hashlib.sha256()
        for child in sorted(path.rglob("*")):
            rel = child.relative_to(path).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            if child.is_dir():
                h.update(b"dir\0")
                continue
            h.update(b"file\0")
            h.update(_hash_file(child).encode("utf-8"))
            h.update(b"\0")
        return f"dir:{h.hexdigest()}"
    return "__UNKNOWN__"


def _snapshot_readonly(root: Path, rel_paths: tuple[str, ...]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for rel in rel_paths:
        target = root if rel == "." else (root / rel).resolve()
        snapshot[rel] = _fingerprint_path(target)
    return snapshot


def _check_readonly_violations(root: Path, before: dict[str, str]) -> list[str]:
    violations: list[str] = []
    for rel, old_fp in before.items():
        target = root if rel == "." else (root / rel).resolve()
        new_fp = _fingerprint_path(target)
        if old_fp != new_fp:
            violations.append(rel)
    return violations


def _copy_selected_entries(
    *,
    benchmark_dir: Path,
    sandbox_benchmark: Path,
    copy_files: tuple[str, ...],
) -> tuple[list[str], list[str], list[str]]:
    copied_files: list[str] = []
    copied_dirs: list[str] = []
    missing: list[str] = []

    sandbox_benchmark.mkdir(parents=True, exist_ok=True)

    copy_whole = any(rel == "." for rel in copy_files)
    if copy_whole:
        shutil.copytree(benchmark_dir, sandbox_benchmark, dirs_exist_ok=True)
        copied_dirs.append(".")
        return copied_files, copied_dirs, missing

    for rel in copy_files:
        src = (benchmark_dir / rel).resolve()
        if not _is_within(src, benchmark_dir):
            missing.append(f"{rel} (outside benchmark dir)")
            continue
        if not src.exists():
            missing.append(rel)
            continue

        dst = (sandbox_benchmark / rel).resolve()
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            copied_dirs.append(rel)
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied_files.append(rel)

    return copied_files, copied_dirs, missing


def _append_agent_context(spec: UnifiedTaskSpec, artifacts: dict[str, Any]) -> None:
    if spec.constraints_path is not None:
        artifacts["constraints_path"] = str(spec.constraints_path)
    if spec.constraints_text:
        artifacts["constraints"] = _truncate_middle(spec.constraints_text, limit=120_000)

    if not spec.agent_files:
        return

    artifacts["agent_files"] = "\n".join(spec.agent_files)
    for rel in spec.agent_files:
        src = (spec.benchmark_dir / rel).resolve()
        key_base = f"agent_file::{rel}"
        if not _is_within(src, spec.benchmark_dir):
            artifacts[f"{key_base}::error"] = "outside benchmark dir"
            continue

        if src.is_file():
            text = _read_text(src)
            if text is None:
                artifacts[f"{key_base}::error"] = "failed to read file"
            else:
                artifacts[key_base] = _truncate_middle(text)
            continue

        if src.is_dir():
            entries: list[str] = []
            for child in sorted(src.rglob("*")):
                if child.is_dir():
                    continue
                entries.append(child.relative_to(spec.benchmark_dir).as_posix())
                if len(entries) >= 500:
                    entries.append("... (truncated)")
                    break
            artifacts[f"{key_base}::dir_listing"] = "\n".join(entries)
            continue

        artifacts[f"{key_base}::error"] = "path not found"


def _render_eval_command(
    *,
    command_template: str,
    candidate_dst: Path,
    sandbox_benchmark: Path,
    sandbox_root: Path,
    spec: UnifiedTaskSpec,
) -> str:
    python_cmd = spec.runtime_python_path or "python"
    quoted = {
        "python": shlex.quote(python_cmd),
        "candidate": shlex.quote(str(candidate_dst)),
        "benchmark": shlex.quote(str(sandbox_benchmark)),
        "sandbox": shlex.quote(str(sandbox_root)),
        "repo_root": shlex.quote(str(spec.repo_root)),
        "benchmark_source": shlex.quote(str(spec.benchmark_dir)),
        "benchmark_id": shlex.quote(spec.benchmark_id),
        "python_raw": python_cmd,
        "candidate_raw": str(candidate_dst),
        "benchmark_raw": str(sandbox_benchmark),
        "sandbox_raw": str(sandbox_root),
        "repo_root_raw": str(spec.repo_root),
        "benchmark_source_raw": str(spec.benchmark_dir),
        "benchmark_id_raw": spec.benchmark_id,
    }
    try:
        return command_template.format(**quoted)
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(f"Unknown placeholder in eval command: {{{missing}}}") from e


def evaluate(program_path: str, *, spec: UnifiedTaskSpec) -> Any:
    start = time.time()
    program_path_p = Path(program_path).expanduser().resolve()

    metrics: dict[str, float] = {
        "combined_score": 0.0,
        "valid": 0.0,
        "timeout": 0.0,
        "runtime_s": 0.0,
    }
    artifacts: dict[str, Any] = {
        "benchmark_id": spec.benchmark_id,
        "benchmark_dir": str(spec.benchmark_dir),
        "initial_program_rel": spec.initial_program_rel,
        "candidate_destination_rel": spec.candidate_destination_rel,
        "eval_cwd_rel": spec.eval_cwd_rel,
        "eval_command_template": spec.eval_command,
    }
    _append_agent_context(spec, artifacts)

    if not spec.benchmark_dir.is_dir():
        artifacts["error_message"] = f"benchmark dir not found: {spec.benchmark_dir}"
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)

    if not program_path_p.is_file():
        artifacts["error_message"] = f"candidate program not found: {program_path_p}"
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)

    env_timeout_s = float(os.environ.get("FRONTIER_EVAL_EVALUATOR_TIMEOUT_S", "300") or "300")
    timeout_s = float(env_timeout_s)
    if spec.timeout_s is not None:
        timeout_s = max(1.0, min(timeout_s, float(spec.timeout_s)))
    timeout_budget_s = max(1.0, timeout_s)
    deadline_s = start + max(1.0, timeout_budget_s - 5.0)
    metrics["timeout_budget_s"] = float(timeout_budget_s)

    work_dir = Path(tempfile.mkdtemp(prefix=f"fe_unified_{_safe_slug(spec.benchmark_id)}_")).resolve()
    try:
        sandbox_benchmark = (work_dir / "benchmark").resolve()
        if spec.copy_files:
            copied_files, copied_dirs, missing_entries = _copy_selected_entries(
                benchmark_dir=spec.benchmark_dir,
                sandbox_benchmark=sandbox_benchmark,
                copy_files=spec.copy_files,
            )
            artifacts["copy_mode"] = "selected"
            if copied_files:
                artifacts["copied_files"] = "\n".join(copied_files[:1000])
            if copied_dirs:
                artifacts["copied_dirs"] = "\n".join(copied_dirs[:1000])
            if missing_entries:
                artifacts["missing_copy_entries"] = "\n".join(missing_entries[:200])
        else:
            shutil.copytree(spec.benchmark_dir, sandbox_benchmark)
            artifacts["copy_mode"] = "full_benchmark"

        candidate_dst = (sandbox_benchmark / spec.candidate_destination_rel).resolve()
        candidate_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(program_path_p, candidate_dst)
        artifacts["candidate_program"] = str(candidate_dst)

        readonly_snapshot = _snapshot_readonly(sandbox_benchmark, spec.readonly_files)
        if spec.readonly_files:
            artifacts["readonly_files"] = "\n".join(spec.readonly_files)

        eval_cwd = (sandbox_benchmark / spec.eval_cwd_rel).resolve()
        if not _is_within(eval_cwd, sandbox_benchmark):
            artifacts["error_message"] = f"eval cwd escapes sandbox: {eval_cwd}"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)
        if not eval_cwd.exists():
            artifacts["error_message"] = f"eval cwd does not exist: {eval_cwd}"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        rendered_cmd = _render_eval_command(
            command_template=spec.eval_command,
            candidate_dst=candidate_dst,
            sandbox_benchmark=sandbox_benchmark,
            sandbox_root=work_dir,
            spec=spec,
        )
        artifacts["benchmark_cmd"] = rendered_cmd

        run_with_conda = bool(
            spec.runtime_use_conda_run and spec.runtime_conda_env and not spec.runtime_python_path
        )
        if run_with_conda:
            run_cmd = [
                "conda",
                "run",
                "-n",
                spec.runtime_conda_env,
                spec.runtime_shell,
                "-lc",
                rendered_cmd,
            ]
            artifacts["runtime_mode"] = "conda_run"
            artifacts["runtime_conda_env"] = spec.runtime_conda_env
        else:
            run_cmd = [spec.runtime_shell, "-lc", rendered_cmd]
            artifacts["runtime_mode"] = "shell"
        if spec.runtime_python_path:
            artifacts["runtime_python_path"] = spec.runtime_python_path
        artifacts["runtime_command"] = " ".join(shlex.quote(x) for x in run_cmd)

        env = os.environ.copy()
        env.update(spec.runtime_env)
        env.setdefault("FRONTIER_ENGINEERING_ROOT", str(spec.repo_root))
        env["FRONTIER_EVAL_UNIFIED_SOURCE_BENCHMARK_DIR"] = str(spec.benchmark_dir)
        env["FRONTIER_EVAL_UNIFIED_BENCHMARK_DIR"] = str(sandbox_benchmark)
        env["FRONTIER_EVAL_UNIFIED_CANDIDATE_PATH"] = str(candidate_dst)

        try:
            proc = subprocess.run(
                run_cmd,
                cwd=str(eval_cwd),
                capture_output=True,
                text=True,
                timeout=_remaining_timeout(deadline_s),
                env=env,
            )
        except FileNotFoundError as e:
            artifacts["error_message"] = f"runtime executable not found: {e}"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)
        except subprocess.TimeoutExpired as e:
            artifacts["error_message"] = f"benchmark timeout: {e}"
            metrics["timeout"] = 1.0
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        artifacts["benchmark_stdout"] = _tail(proc.stdout)
        artifacts["benchmark_stderr"] = _tail(proc.stderr)
        artifacts["benchmark_stdout_full"] = _truncate_middle(proc.stdout)
        artifacts["benchmark_stderr_full"] = _truncate_middle(proc.stderr)
        metrics["benchmark_returncode"] = float(proc.returncode)

        loaded_metrics = False
        if spec.metrics_json_rel:
            metrics_path = (sandbox_benchmark / spec.metrics_json_rel).resolve()
            artifacts["metrics_json_path"] = str(metrics_path)
            metrics_payload = _read_json(metrics_path)
            if isinstance(metrics_payload, dict):
                loaded_metrics = True
                numeric_metrics, non_numeric_metrics = _extract_numeric_metrics(metrics_payload)
                metrics.update(numeric_metrics)
                if non_numeric_metrics:
                    artifacts["metrics_non_numeric"] = json.dumps(
                        non_numeric_metrics,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )
            elif metrics_path.exists():
                artifacts["metrics_json_error"] = (
                    "metrics_json exists but is not valid JSON object"
                )

        if not loaded_metrics and spec.parse_stdout_json:
            parsed_stdout = _parse_last_json_dict(proc.stdout)
            if isinstance(parsed_stdout, dict):
                numeric_metrics, non_numeric_metrics = _extract_numeric_metrics(parsed_stdout)
                if numeric_metrics:
                    loaded_metrics = True
                    metrics.update(numeric_metrics)
                if non_numeric_metrics:
                    artifacts["stdout_json_non_numeric"] = json.dumps(
                        non_numeric_metrics,
                        ensure_ascii=False,
                        indent=2,
                        default=str,
                    )

        if spec.artifacts_json_rel:
            artifacts_path = (sandbox_benchmark / spec.artifacts_json_rel).resolve()
            artifacts["artifacts_json_path"] = str(artifacts_path)
            artifacts_payload = _read_json(artifacts_path)
            if isinstance(artifacts_payload, dict):
                for key, value in artifacts_payload.items():
                    artifacts[f"user_artifact::{key}"] = value
            elif artifacts_path.exists():
                artifacts["artifacts_json_error"] = (
                    "artifacts_json exists but is not valid JSON object"
                )

        if "valid" not in metrics:
            metrics["valid"] = 1.0 if proc.returncode == 0 else 0.0
        if "combined_score" not in metrics:
            metrics["combined_score"] = 1.0 if metrics.get("valid", 0.0) > 0 else 0.0

        if proc.returncode != 0:
            metrics["valid"] = 0.0
            metrics["combined_score"] = 0.0
            if "error_message" not in artifacts:
                artifacts["error_message"] = (
                    f"evaluation command failed with return code {proc.returncode}"
                )

        if readonly_snapshot:
            violations = _check_readonly_violations(sandbox_benchmark, readonly_snapshot)
            if violations:
                metrics["readonly_violation"] = 1.0
                metrics["valid"] = 0.0
                metrics["combined_score"] = 0.0
                artifacts["readonly_violations"] = "\n".join(violations[:200])
                if "error_message" not in artifacts:
                    artifacts["error_message"] = "readonly files modified by evaluation run"

        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _wrap(metrics: dict[str, float], artifacts: dict[str, Any]) -> Any:
    try:
        from openevolve.evaluation_result import EvaluationResult
    except Exception:
        return metrics
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
