from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from parse_mdriver_result import parse_result


def _tail(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _run_cmd(cmd: list[str], *, cwd: Path, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=max(1.0, float(timeout_s)),
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal MallocLab evaluator for UnifiedTask.")
    p.add_argument("--workdir", type=str, required=True)
    p.add_argument("--candidate", type=str, required=False, default="")
    p.add_argument("--metrics-out", type=str, required=True)
    p.add_argument("--artifacts-out", type=str, required=True)
    p.add_argument("--timeout-s", type=float, default=240.0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    start = time.time()

    benchmark_dir = Path(args.workdir).expanduser().resolve()
    handout_dir = (benchmark_dir / "malloclab-handout").resolve()
    metrics_out = Path(args.metrics_out).expanduser().resolve()
    artifacts_out = Path(args.artifacts_out).expanduser().resolve()
    timeout_s = max(1.0, float(args.timeout_s))

    mdriver_stdout_file = (benchmark_dir / "mdriver.stdout.txt").resolve()
    mdriver_stderr_file = (benchmark_dir / "mdriver.stderr.txt").resolve()
    make_clean_log = (benchmark_dir / "make_clean.log").resolve()
    make_log = (benchmark_dir / "make.log").resolve()

    metrics: dict[str, float] = {
        "combined_score": 0.0,
        "valid": 0.0,
        "runtime_s": 0.0,
    }
    artifacts: dict[str, Any] = {
        "benchmark_dir": str(benchmark_dir),
        "handout_dir": str(handout_dir),
        "candidate_path": str(Path(args.candidate).expanduser().resolve()) if args.candidate else "",
        "make_clean_log": str(make_clean_log),
        "make_log": str(make_log),
        "mdriver_stdout_file": str(mdriver_stdout_file),
        "mdriver_stderr_file": str(mdriver_stderr_file),
    }

    if not handout_dir.is_dir():
        artifacts["error_message"] = f"handout dir not found: {handout_dir}"
        metrics["runtime_s"] = float(time.time() - start)
        _write_json(metrics_out, metrics)
        _write_json(artifacts_out, artifacts)
        return 0

    try:
        proc_clean = _run_cmd(["make", "clean"], cwd=handout_dir, timeout_s=timeout_s)
        make_clean_log.write_text(
            (proc_clean.stdout or "") + "\n" + (proc_clean.stderr or ""),
            encoding="utf-8",
            errors="replace",
        )
    except Exception as e:
        artifacts["error_message"] = f"make clean failed: {e}"
        metrics["runtime_s"] = float(time.time() - start)
        _write_json(metrics_out, metrics)
        _write_json(artifacts_out, artifacts)
        return 0

    try:
        proc_make = _run_cmd(["make"], cwd=handout_dir, timeout_s=timeout_s)
        make_log.write_text(
            (proc_make.stdout or "") + "\n" + (proc_make.stderr or ""),
            encoding="utf-8",
            errors="replace",
        )
        metrics["make_returncode"] = float(proc_make.returncode)
        artifacts["make_stdout_tail"] = _tail(proc_make.stdout or "")
        artifacts["make_stderr_tail"] = _tail(proc_make.stderr or "")
    except Exception as e:
        artifacts["error_message"] = f"make failed: {e}"
        metrics["runtime_s"] = float(time.time() - start)
        _write_json(metrics_out, metrics)
        _write_json(artifacts_out, artifacts)
        return 0

    if int(metrics.get("make_returncode", 1)) != 0:
        artifacts["error_message"] = "make returned non-zero"
        metrics["runtime_s"] = float(time.time() - start)
        _write_json(metrics_out, metrics)
        _write_json(artifacts_out, artifacts)
        return 0

    try:
        with mdriver_stdout_file.open("w", encoding="utf-8", errors="replace") as f_out:
            with mdriver_stderr_file.open("w", encoding="utf-8", errors="replace") as f_err:
                proc_driver = subprocess.run(
                    ["./mdriver", "-V"],
                    cwd=str(handout_dir),
                    text=True,
                    stdout=f_out,
                    stderr=f_err,
                    timeout=timeout_s,
                )
    except Exception as e:
        artifacts["error_message"] = f"mdriver run failed: {e}"
        metrics["runtime_s"] = float(time.time() - start)
        _write_json(metrics_out, metrics)
        _write_json(artifacts_out, artifacts)
        return 0

    mdriver_stdout = mdriver_stdout_file.read_text(encoding="utf-8", errors="replace")
    mdriver_stderr = mdriver_stderr_file.read_text(encoding="utf-8", errors="replace")
    parsed_metrics, parsed_artifacts = parse_result(
        stdout_text=mdriver_stdout,
        stderr_text=mdriver_stderr,
        mdriver_returncode=int(proc_driver.returncode),
    )
    metrics.update(parsed_metrics)
    artifacts.update(parsed_artifacts)

    metrics["runtime_s"] = float(time.time() - start)
    _write_json(metrics_out, metrics)
    _write_json(artifacts_out, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
