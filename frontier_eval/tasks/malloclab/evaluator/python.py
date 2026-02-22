from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


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


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _remaining_timeout(deadline_s: float) -> float:
    return max(1.0, float(deadline_s - time.time()))


def _parse_mdriver_output(text: str) -> tuple[dict[str, float], dict[str, str]]:
    metrics: dict[str, float] = {}
    artifacts: dict[str, str] = {}

    score_line = ""
    for raw in (text or "").splitlines():
        line = raw.strip()
        if line.startswith("Score =") or line.startswith("Perf index ="):
            score_line = line
    if score_line:
        artifacts["score_line"] = score_line

    score_match = re.search(r"=\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*100\b", score_line or text)
    if score_match:
        score = float(score_match.group(1))
        metrics["score_raw_100"] = score
        metrics["score_raw_ratio"] = score / 100.0

    util_thru_match = re.search(
        r"\(\s*([0-9]+(?:\.[0-9]+)?)\s*\(util\)\s*\+\s*([0-9]+(?:\.[0-9]+)?)\s*\(thru\)\s*\)",
        score_line or text,
    )
    if util_thru_match:
        metrics["util_points"] = float(util_thru_match.group(1))
        metrics["thru_points"] = float(util_thru_match.group(2))

    testcase_match = re.search(
        r"\*\s*([0-9]+)\s*/\s*([0-9]+)\s*\(testcase\)",
        score_line or text,
    )
    if testcase_match:
        passed = float(testcase_match.group(1))
        total = float(testcase_match.group(2))
        metrics["testcases_passed"] = passed
        metrics["testcases_total"] = total
        if total > 0:
            metrics["testcase_pass_rate"] = passed / total

    errors_match = re.search(r"(\d+)\s+errors?\s+occurred", text or "", flags=re.IGNORECASE)
    if errors_match:
        metrics["errors_count"] = float(errors_match.group(1))
    else:
        metrics["errors_count"] = 0.0

    # Guard against mdriver's unchecked util overflow on invalid allocators.
    # Physics-aware bounds from config.h:
    # - util contribution in printed score is in [0, 60]
    # - throughput contribution is in [0, 40]
    # Final score is (util_pts + thru_pts) * passed/total.
    util_points = metrics.get("util_points")
    thru_points = metrics.get("thru_points")
    passed = metrics.get("testcases_passed")
    total = metrics.get("testcases_total")

    guarded_score: float | None = None
    if (
        util_points is not None
        and thru_points is not None
        and passed is not None
        and total is not None
        and total > 0
    ):
        util_capped = max(0.0, min(float(util_points), 60.0))
        thru_capped = max(0.0, min(float(thru_points), 40.0))
        metrics["util_points_capped"] = util_capped
        metrics["thru_points_capped"] = thru_capped
        guarded_score = (util_capped + thru_capped) * (float(passed) / float(total))
        metrics["score_guarded_100"] = guarded_score

    if guarded_score is None and "score_raw_100" in metrics:
        guarded_score = max(0.0, min(float(metrics["score_raw_100"]), 100.0))
        metrics["score_guarded_100"] = guarded_score

    final_score: float | None = None
    if guarded_score is not None:
        raw_score = metrics.get("score_raw_100")
        if raw_score is not None:
            # Keep mdriver's displayed score when it is within a safe bound.
            # If mdriver score is inflated by util overflow, cap it by guarded score.
            final_score = min(float(raw_score), float(guarded_score))
        else:
            final_score = float(guarded_score)

    if final_score is not None:
        metrics["combined_score"] = final_score
        metrics["score_100"] = final_score
        metrics["score_ratio"] = final_score / 100.0

    return metrics, artifacts


def evaluate(program_path: str, *, repo_root: Path | None = None):
    """
    OpenEvolve evaluator for benchmarks/ComputerSystems/MallocLab.

    Contract for candidate program:
    - Candidate file is copied to malloclab-handout/mm.c
    - Evaluator runs `make` then `./mdriver -V`
    - Final score is parsed from `Score = ... = X/100`
    """
    start = time.time()
    repo_root = _find_repo_root() if repo_root is None else repo_root.expanduser().resolve()
    program_path_p = Path(program_path).expanduser().resolve()

    benchmark_dir = (repo_root / "benchmarks" / "ComputerSystems" / "MallocLab").resolve()
    if not benchmark_dir.is_dir():
        benchmark_dir = (repo_root / "ComputerSystems" / "MallocLab").resolve()
    handout_dir = (benchmark_dir / "malloclab-handout").resolve()

    artifacts: dict[str, str] = {}
    metrics: dict[str, float] = {
        "combined_score": 0.0,
        "valid": 0.0,
        "timeout": 0.0,
        "runtime_s": 0.0,
    }
    artifacts["interface_contract"] = (
        "Hard requirements for candidate program (do NOT change these):\n"
        "1) Candidate program is C source for malloclab-handout/mm.c.\n"
        "2) Only mm.c should be modified.\n"
        "3) Evaluator runs `make` then `./mdriver -V`.\n"
        "4) Final score is parsed from the `Score = ... = X/100` line.\n"
        "5) Keep function signatures in mm.c unchanged (mm_init/mm_malloc/mm_free/mm_realloc)."
    )

    task_spec_zh_cn_path = (benchmark_dir / "Task_zh-CN.md").resolve()
    artifacts["task_spec_zh_cn_path"] = str(task_spec_zh_cn_path)
    task_spec_zh_cn = _read_text(task_spec_zh_cn_path)
    if task_spec_zh_cn:
        artifacts["task_spec_zh_cn"] = _truncate_middle(task_spec_zh_cn)

    if not handout_dir.is_dir():
        artifacts["error_message"] = f"MallocLab benchmark folder missing: {handout_dir}"
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)
    if not program_path_p.is_file():
        artifacts["error_message"] = f"program not found: {program_path_p}"
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)

    evaluator_timeout_s = float(os.environ.get("FRONTIER_EVAL_EVALUATOR_TIMEOUT_S", "300") or "300")
    deadline_s = start + max(1.0, evaluator_timeout_s - 5.0)

    work_dir = Path(tempfile.mkdtemp(prefix="fe_malloclab_")).resolve()
    try:
        sandbox_dir = (work_dir / "malloclab-handout").resolve()
        shutil.copytree(handout_dir, sandbox_dir)

        candidate_dst = (sandbox_dir / "mm.c").resolve()
        shutil.copyfile(program_path_p, candidate_dst)
        artifacts["candidate_program"] = str(candidate_dst)

        try:
            proc_make = subprocess.run(
                ["make"],
                cwd=str(sandbox_dir),
                capture_output=True,
                text=True,
                timeout=_remaining_timeout(deadline_s),
            )
        except subprocess.TimeoutExpired as e:
            metrics["timeout"] = 1.0
            metrics["runtime_s"] = float(time.time() - start)
            artifacts["error_message"] = f"build timeout: {e}"
            return _wrap(metrics, artifacts)
        except FileNotFoundError as e:
            metrics["runtime_s"] = float(time.time() - start)
            artifacts["error_message"] = f"build tool unavailable: {e}"
            return _wrap(metrics, artifacts)

        metrics["make_returncode"] = float(proc_make.returncode)
        artifacts["make_stdout"] = _tail(proc_make.stdout)
        artifacts["make_stderr"] = _tail(proc_make.stderr)
        artifacts["make_stdout_full"] = _truncate_middle(proc_make.stdout)
        artifacts["make_stderr_full"] = _truncate_middle(proc_make.stderr)

        if proc_make.returncode != 0:
            artifacts["error_message"] = "build failed (make returned non-zero)"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        try:
            proc_driver = subprocess.run(
                ["./mdriver", "-V"],
                cwd=str(sandbox_dir),
                capture_output=True,
                text=True,
                timeout=_remaining_timeout(deadline_s),
            )
        except subprocess.TimeoutExpired as e:
            metrics["timeout"] = 1.0
            metrics["runtime_s"] = float(time.time() - start)
            artifacts["error_message"] = f"mdriver timeout: {e}"
            return _wrap(metrics, artifacts)
        except FileNotFoundError as e:
            metrics["runtime_s"] = float(time.time() - start)
            artifacts["error_message"] = f"mdriver unavailable: {e}"
            return _wrap(metrics, artifacts)

        metrics["mdriver_returncode"] = float(proc_driver.returncode)
        artifacts["mdriver_stdout"] = _tail(proc_driver.stdout)
        artifacts["mdriver_stderr"] = _tail(proc_driver.stderr)
        artifacts["mdriver_stdout_full"] = _truncate_middle(proc_driver.stdout)
        artifacts["mdriver_stderr_full"] = _truncate_middle(proc_driver.stderr)

        combined_output = (proc_driver.stdout or "") + "\n" + (proc_driver.stderr or "")
        parsed_metrics, parsed_artifacts = _parse_mdriver_output(combined_output)
        metrics.update(parsed_metrics)
        artifacts.update(parsed_artifacts)

        if proc_driver.returncode != 0:
            artifacts["error_message"] = (
                f"mdriver failed (returncode={proc_driver.returncode}), score may be incomplete"
            )
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        if "combined_score" in metrics:
            metrics["valid"] = 1.0
        else:
            artifacts["error_message"] = "failed to parse final score from mdriver output"

        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _wrap(metrics: dict[str, float], artifacts: dict[str, str]):
    try:
        from openevolve.evaluation_result import EvaluationResult
    except Exception:
        return metrics
    return EvaluationResult(metrics=metrics, artifacts=artifacts)
