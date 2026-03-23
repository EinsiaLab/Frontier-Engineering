from __future__ import annotations

import math
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


def _remaining_timeout(deadline_s: float) -> float:
    return max(1.0, float(deadline_s - time.time()))


def _parse_popcorn_log(log_text: str) -> tuple[dict[str, str], list[float], list[str]]:
    fields: dict[str, str] = {}
    mean_by_case: list[tuple[int, float]] = []
    failures: list[str] = []

    for raw in (log_text or "").splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        fields[key] = value

        m_mean = re.fullmatch(r"benchmark\.(\d+)\.mean", key)
        if m_mean:
            try:
                mean_by_case.append((int(m_mean.group(1)), float(value)))
            except Exception:
                continue

        if re.fullmatch(r"benchmark\.\d+\.error", key):
            failures.append(value)

    mean_by_case.sort(key=lambda x: x[0])
    return fields, [v for _, v in mean_by_case], failures


def _geometric_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    safe = [max(float(v), 1e-30) for v in values]
    return float(math.exp(sum(math.log(v) for v in safe) / len(safe)))


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _write_spec_benchmark_file(path: Path) -> None:
    """Match the public task statement's benchmark cases."""
    path.write_text(
        "\n".join(
            [
                "batchsize: 128; dim: 7168; dq: 1536; prefill: 512; seed: 2197",
                "batchsize: 128; dim: 7168; dq: 1536; prefill: 2048; seed: 9817",
                "batchsize: 128; dim: 7168; dq: 1536; prefill: 4096; seed: 9817",
                "batchsize: 128; dim: 7168; dq: 1536; prefill: 6144; seed: 5291",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_mla_eval_runner(path: Path, *, serial: bool = False) -> None:
    """
    Write a wrapper that keeps evaluator compatibility with common MLA submission variants
    while hardening the timing harness against object-reuse exploits.

    Why:
    - Some generated submissions keep type hints `input_t` / `output_t` but drop imports.
      Without postponed annotation evaluation this raises NameError at import time.
    - Some submissions call `cache.update(...)` / `cache.reset()`, while the official
      benchmark cache exposes `forward(...)` / `zero()`.
    - The upstream benchmark warms the candidate and performs an untimed correctness call
      before timed runs, then reuses the same `(config, x, kv_cache)` objects inside the
      timing loop. That lets a submission shift model construction or per-object caches
      outside the measured region. The wrapper below keeps the same CLI/logging surface
      but patches the runtime so every timed run receives fresh objects and the first
      measured call is also correctness-checked.
    """
    serial_patch = ""
    if serial:
        serial_patch = (
            "import multiprocessing\n"
            "\n"
            "class _SerialPool:\n"
            "    def __enter__(self):\n"
            "        return self\n"
            "    def __exit__(self, exc_type, exc_val, exc_tb):\n"
            "        return False\n"
            "    def apply(self, fn, args=(), kwds=None):\n"
            "        kwds = {} if kwds is None else kwds\n"
            "        return fn(*args, **kwds)\n"
            "\n"
            "class _Ctx:\n"
            "    def Pool(self, *_args, **_kwargs):\n"
            "        return _SerialPool()\n"
            "\n"
            "def _get_context(_method='spawn'):\n"
            "    return _Ctx()\n"
            "\n"
            "multiprocessing.get_context = _get_context\n"
            "\n"
        )

    runner_text = (
        "import builtins\n"
        "import sys\n"
        "import time\n"
        "\n"
        + serial_patch
        + "# 1) Provide fallback symbols for runtime-evaluated type annotations.\n"
        "try:\n"
        "    from baseline import task as _task\n"
        "    builtins.input_t = getattr(_task, 'input_t', tuple)\n"
        "    builtins.output_t = getattr(_task, 'output_t', tuple)\n"
        "except Exception:\n"
        "    builtins.input_t = tuple\n"
        "    builtins.output_t = tuple\n"
        "\n"
        "# 2) Add cache API aliases used by some generated programs.\n"
        "try:\n"
        "    from baseline import reference as _ref\n"
        "    _kv_cls = getattr(_ref, 'KVCache', None)\n"
        "    if _kv_cls is not None:\n"
        "        if (not hasattr(_kv_cls, 'update')) and hasattr(_kv_cls, 'forward'):\n"
        "            _kv_cls.update = _kv_cls.forward\n"
        "        if (not hasattr(_kv_cls, 'reset')) and hasattr(_kv_cls, 'zero'):\n"
        "            _kv_cls.reset = _kv_cls.zero\n"
        "except Exception:\n"
        "    pass\n"
        "\n"
        "import torch\n"
        "import eval as mla_eval\n"
        "\n"
        "def _hardened_warm_up(_test):\n"
        "    # Do not pre-run the candidate outside the timed region.\n"
        "    try:\n"
        "        torch.cuda.synchronize()\n"
        "    except Exception:\n"
        "        pass\n"
        "\n"
        "def _hardened_benchmark(test, recheck, max_repeats, max_time_ns):\n"
        "    durations = []\n"
        "    with torch.no_grad():\n"
        "        for i in range(max_repeats):\n"
        "            config, data, kv_cache = mla_eval.generate_input(**test.args)\n"
        "            should_recheck = bool(recheck) or i == 0\n"
        "            config_copy = None\n"
        "            kv_cache_copy = None\n"
        "            if should_recheck:\n"
        "                kv_cache_copy = mla_eval.copy_kv_cache(kv_cache, config.kv_cache_shape)\n"
        "                config_copy = mla_eval.copy_config_weights(config)\n"
        "            torch.cuda.synchronize()\n"
        "            start = time.perf_counter_ns()\n"
        "            output = mla_eval.custom_kernel((config, data, kv_cache))\n"
        "            torch.cuda.synchronize()\n"
        "            end = time.perf_counter_ns()\n"
        "            if should_recheck:\n"
        "                error = mla_eval.check_implementation((config_copy, data, kv_cache_copy), output)\n"
        "                if error:\n"
        "                    return error\n"
        "            durations.append(end - start)\n"
        "            del output\n"
        "            if i > 1:\n"
        "                stats = mla_eval.calculate_stats(durations)\n"
        "                if stats.err / max(stats.mean, 1.0) < 0.01 or stats.mean * stats.runs > max_time_ns:\n"
        "                    break\n"
        "    return mla_eval.calculate_stats(durations)\n"
        "\n"
        "mla_eval.warm_up = _hardened_warm_up\n"
        "mla_eval.benchmark = _hardened_benchmark\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(mla_eval.main())\n"
    )
    path.write_text(runner_text, encoding="utf-8")


def evaluate(
    program_path: str,
    *,
    repo_root: Path | None = None,
    kernel_python: str | None = None,
):
    """
    OpenEvolve evaluator for benchmarks/KernelEngineering/MLA.

    Contract for candidate program:
    - Candidate file is copied to baseline/submission.py
    - Candidate must define `custom_kernel(data)` compatible with MLA baseline.
    """
    start = time.time()
    repo_root = _find_repo_root() if repo_root is None else repo_root.expanduser().resolve()
    program_path = str(Path(program_path).expanduser().resolve())

    benchmark_dir = (repo_root / "benchmarks" / "KernelEngineering" / "MLA").resolve()
    if not benchmark_dir.is_dir():
        benchmark_dir = (repo_root / "KernelEngineering" / "MLA").resolve()
    baseline_dir = (benchmark_dir / "baseline").resolve()
    verification_dir = (benchmark_dir / "verification").resolve()

    artifacts: dict[str, str] = {}
    metrics: dict[str, float] = {
        "combined_score": 0.0,
        "valid": 0.0,
        "timeout": 0.0,
        "runtime_s": 0.0,
        "benchmark_count": 0.0,
        "geom_mean_ns": 0.0,
    }
    artifacts["interface_contract"] = (
        "Hard requirements for candidate program (do NOT change these):\n"
        "1) Evaluator copies candidate file to baseline/submission.py and runs "
        "`python eval.py benchmark mla_bench.txt`.\n"
        "2) Candidate MUST expose `custom_kernel(data)`.\n"
        "3) `data` is a 3-tuple `(config, x, kv_cache)` produced by baseline/reference.py.\n"
        "4) `kv_cache` follows baseline KVCache semantics (`forward`/callable + `get_data`).\n"
        "5) Keep returned value as `(output, updated_kv_cache)`.\n"
        "6) Do not change evaluator CLI or test file names."
    )

    # Provide the task statement to later evolution rounds via prompt artifacts.
    task_spec_zh_cn_path = (benchmark_dir / "Task_zh-CN.md").resolve()
    artifacts["task_spec_zh_cn_path"] = str(task_spec_zh_cn_path)
    task_spec_zh_cn = _read_text(task_spec_zh_cn_path)
    if task_spec_zh_cn:
        artifacts["task_spec_zh_cn"] = _truncate_middle(task_spec_zh_cn)

    if not baseline_dir.is_dir() or not verification_dir.is_dir():
        artifacts["error_message"] = (
            f"MLA benchmark folder missing: baseline={baseline_dir}, "
            f"verification={verification_dir}"
        )
        metrics["runtime_s"] = float(time.time() - start)
        return _wrap(metrics, artifacts)

    kernel_python = (
        str(kernel_python or "").strip()
        or str(os.environ.get("FRONTIER_EVAL_MLA_PYTHON", "") or "").strip()
        or "python"
    )
    artifacts["kernel_python"] = kernel_python

    work_dir = Path(tempfile.mkdtemp(prefix="fe_mla_")).resolve()

    evaluator_timeout_s = float(os.environ.get("FRONTIER_EVAL_EVALUATOR_TIMEOUT_S", "1200") or "1200")
    deadline_s = start + max(1.0, evaluator_timeout_s - 5.0)

    try:
        sandbox_task_dir = (work_dir / "MLA").resolve()
        sandbox_baseline = (sandbox_task_dir / "baseline").resolve()
        sandbox_verification = (sandbox_task_dir / "verification").resolve()
        shutil.copytree(baseline_dir, sandbox_baseline)
        shutil.copytree(verification_dir, sandbox_verification)

        candidate_dst = (sandbox_baseline / "submission.py").resolve()
        shutil.copy2(program_path, candidate_dst)
        artifacts["candidate_program"] = str(candidate_dst)

        log_path = (sandbox_verification / "mla_bench.log").resolve()
        env = os.environ.copy()
        env.setdefault("FRONTIER_ENGINEERING_ROOT", str(repo_root))
        env.pop("POPCORN_FD", None)

        # Fast-fail with actionable diagnostics if the kernel environment cannot see GPU.
        cuda_probe_cmd = [
            kernel_python,
            "-c",
            (
                "import sys, torch; "
                "ok = bool(torch.cuda.is_available()) and int(torch.cuda.device_count()) > 0; "
                "print(f'is_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}'); "
                "sys.exit(0 if ok else 7)"
            ),
        ]
        try:
            probe = subprocess.run(
                cuda_probe_cmd,
                cwd=str(sandbox_verification),
                capture_output=True,
                text=True,
                timeout=min(30.0, _remaining_timeout(deadline_s)),
                env=env,
            )
            artifacts["cuda_probe_cmd"] = " ".join(cuda_probe_cmd)
            artifacts["cuda_probe_stdout"] = _tail(probe.stdout)
            artifacts["cuda_probe_stderr"] = _tail(probe.stderr)
            if probe.returncode != 0:
                artifacts["error_message"] = (
                    "CUDA is unavailable in FRONTIER_EVAL_MLA_PYTHON environment. "
                    "Ensure the benchmark runs on a GPU node and the runtime can access /dev GPU devices."
                )
                metrics["runtime_s"] = float(time.time() - start)
                return _wrap(metrics, artifacts)
        except subprocess.TimeoutExpired as e:
            artifacts["error_message"] = f"cuda probe timeout: {e}"
            metrics["timeout"] = 1.0
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)
        except FileNotFoundError as e:
            artifacts["error_message"] = f"kernel python not found: {e}"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        def _run_with_log(cmd: list[str]):
            fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.set_inheritable(fd, True)
            env["POPCORN_FD"] = str(fd)
            try:
                return subprocess.run(
                    cmd,
                    cwd=str(sandbox_verification),
                    capture_output=True,
                    text=True,
                    timeout=_remaining_timeout(deadline_s),
                    env=env,
                    pass_fds=(fd,),
                )
            finally:
                try:
                    os.close(fd)
                except Exception:
                    pass
                env.pop("POPCORN_FD", None)

        wrapper_path = (sandbox_verification / "_mla_eval_runner.py").resolve()
        _write_mla_eval_runner(wrapper_path)
        bench_path = (sandbox_verification / "_fe_mla_bench_spec.txt").resolve()
        _write_spec_benchmark_file(bench_path)

        cmd = [kernel_python, str(wrapper_path), "benchmark", str(bench_path)]
        artifacts["runner_mode"] = "compat_wrapper_hardened"
        artifacts["benchmark_cmd"] = " ".join(cmd)
        artifacts["benchmark_spec_path"] = str(bench_path)

        try:
            proc = _run_with_log(cmd)
        except FileNotFoundError as e:
            artifacts["error_message"] = f"kernel python not found: {e}"
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)
        except subprocess.TimeoutExpired as e:
            artifacts["error_message"] = f"benchmark timeout: {e}"
            metrics["timeout"] = 1.0
            metrics["runtime_s"] = float(time.time() - start)
            return _wrap(metrics, artifacts)

        if proc.returncode != 0 and "PermissionError" in proc.stderr and "SemLock" in proc.stderr:
            wrapper_path = (sandbox_verification / "_serial_eval_runner.py").resolve()
            _write_mla_eval_runner(wrapper_path, serial=True)
            cmd = [kernel_python, str(wrapper_path), "benchmark", str(bench_path)]
            artifacts["runner_mode"] = "serial_fallback_hardened"
            artifacts["benchmark_cmd"] = " ".join(cmd)
            artifacts["fallback_reason"] = "PermissionError SemLock"
            try:
                proc = _run_with_log(cmd)
            except subprocess.TimeoutExpired as e:
                artifacts["error_message"] = f"benchmark timeout (serial fallback): {e}"
                metrics["timeout"] = 1.0
                metrics["runtime_s"] = float(time.time() - start)
                return _wrap(metrics, artifacts)

        artifacts["benchmark_stdout"] = _tail(proc.stdout)
        artifacts["benchmark_stderr"] = _tail(proc.stderr)
        artifacts["benchmark_stdout_full"] = _truncate_middle(proc.stdout)
        artifacts["benchmark_stderr_full"] = _truncate_middle(proc.stderr)
        metrics["benchmark_returncode"] = float(proc.returncode)

        log_text = ""
        if log_path.is_file():
            try:
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                log_text = ""
        artifacts["mla_bench.log_tail"] = _tail(log_text)
        if log_text:
            artifacts["mla_bench.log"] = _truncate_middle(log_text)

        fields, means_ns, failures = _parse_popcorn_log(log_text)
        if fields.get("check") is not None:
            artifacts["check"] = fields.get("check", "")
        if failures:
            artifacts["failure_summary"] = "\n".join(failures[:8])

        if means_ns:
            gmean_ns = _geometric_mean(means_ns)
            metrics["benchmark_count"] = float(len(means_ns))
            metrics["geom_mean_ns"] = float(gmean_ns)
            metrics["best_case_ns"] = float(min(means_ns))
            metrics["worst_case_ns"] = float(max(means_ns))

            # Speed score: larger is better (approx kernels/sec).
            if gmean_ns > 0:
                metrics["combined_score"] = float(1e9 / gmean_ns)

        passed = (
            proc.returncode == 0
            and fields.get("check", "").strip().lower() == "pass"
            and bool(means_ns)
        )
        if passed:
            metrics["valid"] = 1.0
        else:
            metrics["valid"] = 0.0
            metrics["combined_score"] = 0.0
            if "error_message" not in artifacts:
                if failures:
                    artifacts["error_message"] = failures[0]
                else:
                    artifacts["error_message"] = (
                        f"benchmark failed: returncode={proc.returncode}, "
                        f"check={fields.get('check', '')}"
                    )

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
