from __future__ import annotations

import csv
import fcntl
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


METHOD_ID = "submission"
DEFAULT_REFERENCE_METHODS = "magic,dca,cellmapper,knn_smoothing,alra"


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


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _safe_metric_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower() or "unknown"


def _discover_latest_run_dir(results_dir: Path, before: set[str]) -> Path | None:
    if not results_dir.is_dir():
        return None

    runs = [p for p in results_dir.glob("testrun_*") if p.is_dir()]
    if not runs:
        return None

    new_runs = [p for p in runs if p.name not in before]
    candidates = new_runs if new_runs else runs

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except Exception:
            return 0.0

    candidates.sort(key=_mtime, reverse=True)
    return candidates[0]


def _resolve_method_script(task_dir: Path, method_id: str) -> Path | None:
    method_dir = task_dir / "src" / "methods" / method_id
    candidates = [
        method_dir / "script.py",
        method_dir / "script.R",
        method_dir / "script.sh",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def _collect_reference_methods(task_dir: Path, method_ids: list[str]) -> str:
    sections: list[str] = []
    for method_id in method_ids:
        script_path = _resolve_method_script(task_dir, method_id)
        if script_path is None:
            continue
        text = _read_text(script_path)
        if not text:
            continue
        lang = "python"
        if script_path.suffix.lower() == ".r":
            lang = "r"
        elif script_path.suffix.lower() == ".sh":
            lang = "bash"
        snippet = _truncate_middle(text, limit=12_000)
        sections.append(
            f"### {method_id}\n"
            f"path: {script_path}\n"
            f"```{lang}\n{snippet}\n```"
        )
    return "\n\n".join(sections)


def _parse_submission_rows(csv_path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = str(row.get("method_id", "") or "").strip()
            if method != METHOD_ID:
                continue

            dataset_id = str(row.get("dataset_id", "") or "").strip()
            metric_id = str(row.get("metric_ids", "") or row.get("metric_id", "") or "").strip()

            try:
                normalized_score = float(row.get("normalized_score", "nan"))
                raw_score = float(row.get("metric_values", "nan"))
            except Exception:
                continue

            if not math.isfinite(normalized_score) or not math.isfinite(raw_score):
                continue

            rows.append(
                {
                    "dataset_id": dataset_id,
                    "metric_id": metric_id,
                    "normalized_score": normalized_score,
                    "raw_score": raw_score,
                }
            )
    return rows


def _acquire_file_lock(lock_path: Path, deadline_s: float):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_f = lock_path.open("a+", encoding="utf-8")
    while True:
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_f
        except BlockingIOError:
            if time.time() >= deadline_s:
                lock_f.close()
                raise TimeoutError(f"timed out acquiring lock: {lock_path}")
            time.sleep(0.2)


def evaluate(
    program_path: str,
    *,
    repo_root: Path | None = None,
    denoising_python: str | None = None,
):
    """
    OpenEvolve evaluator for benchmarks/SingleCellAnalysis/denoising.

    Workflow:
    1) Replace `task_denoising/src/methods/submission/script.py` with candidate code.
    2) Rebuild the submission method via viash.
    3) Run `scripts/run_benchmark/run_test_local.sh`.
    4) Use `verification/rank_scores.py` to extract normalized scores.
    5) Aggregate submission's normalized scores (mean) as `combined_score`.
    """
    start = time.time()
    repo_root = _find_repo_root() if repo_root is None else repo_root.expanduser().resolve()
    program_path_obj = Path(program_path).expanduser().resolve()

    benchmark_dir = (repo_root / "benchmarks" / "SingleCellAnalysis" / "denoising").resolve()
    task_dir = (benchmark_dir / "task_denoising").resolve()
    results_dir = (task_dir / "temp" / "results").resolve()

    submission_src = (task_dir / "src" / "methods" / METHOD_ID / "script.py").resolve()
    submission_nf = (
        task_dir / "target" / "nextflow" / "methods" / METHOD_ID / "main.nf"
    ).resolve()
    submission_exec = (
        task_dir / "target" / "executable" / "methods" / METHOD_ID / METHOD_ID
    ).resolve()

    run_script = (task_dir / "scripts" / "run_benchmark" / "run_test_local.sh").resolve()
    rank_script = (benchmark_dir / "verification" / "rank_scores.py").resolve()
    task_spec_path = (benchmark_dir / "Task.md").resolve()

    artifacts: dict[str, str] = {}
    metrics: dict[str, float] = {
        "combined_score": 0.0,
        "valid": 0.0,
        "timeout": 0.0,
        "runtime_s": 0.0,
    }

    artifacts["interface_contract"] = (
        "Hard requirements for candidate program (do NOT change these):\n"
        "1) Candidate code will replace `task_denoising/src/methods/submission/script.py`.\n"
        "2) Keep this as a valid submission-method script for the denoising benchmark.\n"
        "3) It must output AnnData with layer `denoised` and `uns.dataset_id` + `uns.method_id`.\n"
        "4) Evaluator will run `viash ns build --query '^methods/submission$'`, then "
        "`scripts/run_benchmark/run_test_local.sh`.\n"
        "5) Final score is extracted by running `verification/rank_scores.py` on `score_uns.yaml`.\n"
        "6) If benchmark, ranking, or parsing fails, score is treated as invalid (valid=0)."
    )
    artifacts["task_spec_path"] = str(task_spec_path)
    task_spec = _read_text(task_spec_path)
    if task_spec:
        artifacts["task_spec"] = _truncate_middle(task_spec, limit=120_000)

    ref_methods_raw = str(
        os.environ.get("FRONTIER_EVAL_DENOISING_REFERENCE_METHODS", DEFAULT_REFERENCE_METHODS) or ""
    )
    ref_methods = [x.strip() for x in ref_methods_raw.split(",") if x.strip()]
    if ref_methods:
        artifacts["reference_method_ids"] = ",".join(ref_methods)
        ref_text = _collect_reference_methods(task_dir, ref_methods)
        if ref_text:
            artifacts["reference_methods"] = _truncate_middle(ref_text, limit=150_000)

    if not benchmark_dir.is_dir() or not task_dir.is_dir():
        artifacts["error_message"] = f"denoising benchmark folder missing: {benchmark_dir}"
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)
    if not program_path_obj.is_file():
        artifacts["error_message"] = f"candidate program not found: {program_path_obj}"
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)
    if not run_script.is_file():
        artifacts["error_message"] = f"benchmark runner not found: {run_script}"
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)
    if not rank_script.is_file():
        artifacts["error_message"] = f"rank script not found: {rank_script}"
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)
    if not submission_src.is_file():
        artifacts["error_message"] = f"submission source not found: {submission_src}"
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)

    evaluator_timeout_s = float(os.environ.get("FRONTIER_EVAL_EVALUATOR_TIMEOUT_S", "5400") or "5400")
    deadline_s = start + max(1.0, evaluator_timeout_s - 5.0)
    rank_python = (
        str(denoising_python or "").strip()
        or str(os.environ.get("FRONTIER_EVAL_DENOISING_PYTHON", "") or "").strip()
        or sys.executable
    )
    artifacts["denoising_python"] = rank_python

    env = os.environ.copy()
    env.setdefault("FRONTIER_ENGINEERING_ROOT", str(repo_root))
    env["PYTHONPATH"] = (
        str(repo_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )

    lock_path = (task_dir / "temp" / ".frontier_eval_submission.lock").resolve()
    artifacts["lock_path"] = str(lock_path)
    lock_file = None
    try:
        lock_wait_start = time.time()
        lock_file = _acquire_file_lock(lock_path, deadline_s)
        metrics["lock_wait_s"] = float(time.time() - lock_wait_start)
    except TimeoutError as e:
        artifacts["error_message"] = str(e)
        metrics["timeout"] = 1.0
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)

    # Backup files that can be touched by viash rebuild.
    backup_root = Path(tempfile.mkdtemp(prefix="fe_denoising_backup_")).resolve()
    backup_map: dict[Path, Path] = {}
    backup_targets = [submission_src, submission_nf, submission_exec]
    try:
        for original in backup_targets:
            if not original.is_file():
                continue
            try:
                rel = original.relative_to(task_dir)
            except Exception:
                rel = Path(original.name)
            backup_path = (backup_root / rel).resolve()
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, backup_path)
            backup_map[original] = backup_path

        before_runs: set[str] = set()
        if results_dir.is_dir():
            before_runs = {p.name for p in results_dir.glob("testrun_*") if p.is_dir()}

        # 1) Inject candidate into submission source.
        if program_path_obj != submission_src:
            shutil.copy2(program_path_obj, submission_src)
        artifacts["candidate_program"] = str(program_path_obj)
        artifacts["submission_source"] = str(submission_src)

        # 2) Rebuild submission method so target nextflow module picks up candidate code.
        build_cmd = [
            "viash",
            "ns",
            "build",
            "--parallel",
            "--setup",
            "cachedbuild",
            "--query",
            "^methods/submission$",
        ]
        artifacts["build_cmd"] = " ".join(build_cmd)
        try:
            proc_build = subprocess.run(
                build_cmd,
                cwd=str(task_dir),
                capture_output=True,
                text=True,
                timeout=_remaining_timeout(deadline_s),
                env=env,
            )
        except FileNotFoundError as e:
            artifacts["error_message"] = f"viash not found: {e}"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)
        except subprocess.TimeoutExpired as e:
            artifacts["error_message"] = f"viash build timeout: {e}"
            metrics["timeout"] = 1.0
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        artifacts["build_stdout"] = _tail(proc_build.stdout)
        artifacts["build_stderr"] = _tail(proc_build.stderr)
        artifacts["build_stdout_full"] = _truncate_middle(proc_build.stdout)
        artifacts["build_stderr_full"] = _truncate_middle(proc_build.stderr)
        metrics["build_returncode"] = float(proc_build.returncode)
        if proc_build.returncode != 0:
            artifacts["error_message"] = "viash build failed (non-zero return code)"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        # 3) Run benchmark workflow.
        run_cmd = ["bash", "scripts/run_benchmark/run_test_local.sh"]
        artifacts["benchmark_cmd"] = " ".join(run_cmd)
        try:
            proc_run = subprocess.run(
                run_cmd,
                cwd=str(task_dir),
                capture_output=True,
                text=True,
                timeout=_remaining_timeout(deadline_s),
                env=env,
            )
        except subprocess.TimeoutExpired as e:
            artifacts["error_message"] = f"benchmark timeout: {e}"
            metrics["timeout"] = 1.0
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        artifacts["benchmark_stdout"] = _tail(proc_run.stdout)
        artifacts["benchmark_stderr"] = _tail(proc_run.stderr)
        artifacts["benchmark_stdout_full"] = _truncate_middle(proc_run.stdout)
        artifacts["benchmark_stderr_full"] = _truncate_middle(proc_run.stderr)
        metrics["benchmark_returncode"] = float(proc_run.returncode)
        if proc_run.returncode != 0:
            artifacts["error_message"] = "benchmark failed (non-zero return code)"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        run_dir = _discover_latest_run_dir(results_dir, before_runs)
        if run_dir is None:
            artifacts["error_message"] = f"no benchmark run directory found under {results_dir}"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)
        artifacts["benchmark_run_dir"] = str(run_dir)

        score_uns = (run_dir / "score_uns.yaml").resolve()
        if not score_uns.is_file():
            artifacts["error_message"] = f"score file not found: {score_uns}"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)
        artifacts["score_uns_path"] = str(score_uns)

        # 4) Extract ranking via rank_scores.py.
        rank_cmd = [
            rank_python,
            str(rank_script),
            str(score_uns),
            "--output_dir",
            str(run_dir),
        ]
        artifacts["rank_cmd"] = " ".join(rank_cmd)
        try:
            proc_rank = subprocess.run(
                rank_cmd,
                cwd=str(benchmark_dir),
                capture_output=True,
                text=True,
                timeout=_remaining_timeout(deadline_s),
                env=env,
            )
        except FileNotFoundError as e:
            artifacts["error_message"] = f"denoising python not found: {e}"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)
        except subprocess.TimeoutExpired as e:
            artifacts["error_message"] = f"rank_scores timeout: {e}"
            metrics["timeout"] = 1.0
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        artifacts["rank_stdout"] = _tail(proc_rank.stdout)
        artifacts["rank_stderr"] = _tail(proc_rank.stderr)
        artifacts["rank_stdout_full"] = _truncate_middle(proc_rank.stdout)
        artifacts["rank_stderr_full"] = _truncate_middle(proc_rank.stderr)
        metrics["rank_returncode"] = float(proc_rank.returncode)
        if proc_rank.returncode != 0:
            artifacts["error_message"] = "rank_scores.py failed (non-zero return code)"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        ranked_csv = (run_dir / "ranked_normalized_scores.csv").resolve()
        if not ranked_csv.is_file():
            artifacts["error_message"] = f"ranked csv not found: {ranked_csv}"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)
        artifacts["ranked_csv_path"] = str(ranked_csv)

        rows = _parse_submission_rows(ranked_csv)
        if not rows:
            artifacts["error_message"] = f"no rows found for method_id={METHOD_ID} in ranked CSV"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        normalized_scores = [float(r["normalized_score"]) for r in rows]
        combined_score = float(sum(normalized_scores) / len(normalized_scores))
        metrics["combined_score"] = combined_score
        metrics["submission_score_mean"] = combined_score
        metrics["submission_score_min"] = float(min(normalized_scores))
        metrics["submission_score_max"] = float(max(normalized_scores))
        metrics["submission_scores_count"] = float(len(rows))

        for row in rows:
            dataset_key = _safe_metric_key(str(row["dataset_id"]))
            metric_key = _safe_metric_key(str(row["metric_id"]))
            metrics[f"submission_{dataset_key}_{metric_key}_normalized"] = float(
                row["normalized_score"]
            )
            metrics[f"submission_{dataset_key}_{metric_key}_raw"] = float(row["raw_score"])

        metrics["valid"] = 1.0
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)
    finally:
        # Restore touched files to avoid leaving side effects in the benchmark tree.
        for original, backup in backup_map.items():
            try:
                original.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(backup, original)
            except Exception:
                continue
        shutil.rmtree(backup_root, ignore_errors=True)
        if lock_file is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                lock_file.close()
            except Exception:
                pass


def _wrap(metrics: dict[str, float], artifacts: dict[str, str]):
    try:
        from openevolve.evaluation_result import EvaluationResult
    except Exception:
        return metrics
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
