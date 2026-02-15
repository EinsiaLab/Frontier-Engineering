from __future__ import annotations

import importlib.util
import json
import math
import time
import traceback
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from numpy.random import Generator, Philox

# 候选冻结常量（2026-02-15 标定结果，建议发布前再高预算复验）
DEV_SIGMA = 0.268
TARGET_STD = 0.05
MAX_SAMPLES = 100_000
BATCH_SIZE = 10_000
MIN_ERRORS = 20
REPEATS = 3

EPSILON = 0.8
R0_DEV = 5.52431776694918e-07
R0_LOG_DEV = float(math.log(R0_DEV))
T0_DEV = 0.18551087379455566


def _is_repo_root(path: Path) -> bool:
    return (path / "reliable_sim").is_dir() and (path / "frontier_eval").is_dir()


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if _is_repo_root(parent):
            return parent
    return Path.cwd().resolve()


def _wrap(metrics: dict[str, float], artifacts: dict[str, str]):
    try:
        from openevolve.evaluation_result import EvaluationResult
    except Exception:
        return metrics
    return EvaluationResult(metrics=metrics, artifacts=artifacts)


def _load_program_module(program_path: Path):
    spec = importlib.util.spec_from_file_location("candidate_program", str(program_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载程序文件: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _normalize_result(result: Any) -> tuple[float, float, float, float, float, float]:
    """
    归一化输出到：
    errors_log, weights_log, err_ratio, total_samples, actual_std, converged(0/1)
    """
    if isinstance(result, dict):
        return (
            float(result["errors_log"]),
            float(result["weights_log"]),
            float(result.get("err_ratio", np.nan)),
            float(result.get("total_samples", np.nan)),
            float(result.get("actual_std", np.nan)),
            1.0 if bool(result.get("converged", False)) else 0.0,
        )

    if isinstance(result, (tuple, list)) and len(result) >= 6:
        return (
            float(result[0]),
            float(result[1]),
            float(result[2]),
            float(result[3]),
            float(result[4]),
            1.0 if bool(result[5]) else 0.0,
        )

    raise ValueError("simulate_variance_controlled 返回值格式不支持")


def _build_code(repo_root: Path, seed: int):
    import sys

    sys.path.insert(0, str(repo_root / "reliable_sim"))
    from chase import ChaseDecoder
    from code_linear import HammingCode

    with _pushd(repo_root / "reliable_sim"):
        code = HammingCode(r=7, decoder="binary")
    code.rng = Generator(Philox(seed))
    code.set_decoder(ChaseDecoder(code=code, t=3))
    return code


def evaluate(program_path: str, *, repo_root: Path | None = None):
    start = time.time()
    repo_root = _find_repo_root() if repo_root is None else repo_root.expanduser().resolve()
    program = Path(program_path).expanduser().resolve()

    metrics: dict[str, float] = {
        "combined_score": 0.0,
        "runtime_s": 0.0,
        "error_log_ratio": float("inf"),
        "valid": 0.0,
        "timeout": 0.0,
    }
    artifacts: dict[str, str] = {}

    try:
        import sys

        sys.path.insert(0, str(repo_root / "reliable_sim"))
        from sampler import SamplerBase

        module = _load_program_module(program)
        if not hasattr(module, "MySampler"):
            raise AttributeError("提交程序中未找到类 MySampler")

        cls = module.MySampler
        if not isinstance(cls, type) or not issubclass(cls, SamplerBase):
            raise TypeError("MySampler 必须继承 SamplerBase")

        runtimes: list[float] = []
        err_logs: list[float] = []
        ratios: list[float] = []
        samples: list[float] = []
        stds: list[float] = []
        converged_flags: list[float] = []

        for rep in range(REPEATS):
            seed = rep
            code = _build_code(repo_root, seed=seed)
            sampler = cls(code=code, seed=seed)
            if hasattr(sampler, "rng"):
                sampler.rng = Generator(Philox(seed))

            if not hasattr(sampler, "simulate_variance_controlled"):
                raise AttributeError("MySampler 缺少 simulate_variance_controlled 方法")

            t0 = time.time()
            result = sampler.simulate_variance_controlled(
                code=code,
                sigma=DEV_SIGMA,
                target_std=TARGET_STD,
                max_samples=MAX_SAMPLES,
                batch_size=BATCH_SIZE,
                fix_tx=True,
                min_errors=MIN_ERRORS,
            )
            dt = time.time() - t0

            errors_log, weights_log, err_ratio, total_samples, actual_std, converged = _normalize_result(result)
            err_rate_log = float(errors_log - weights_log)

            if not np.isfinite(err_rate_log):
                raise ValueError("err_rate_log 非有限值")

            runtimes.append(float(dt))
            err_logs.append(err_rate_log)
            ratios.append(err_ratio)
            samples.append(total_samples)
            stds.append(actual_std)
            converged_flags.append(converged)

        runtime_median = float(np.median(runtimes))
        err_log_median = float(np.median(err_logs))
        err_log_ratio = float(abs(err_log_median - R0_LOG_DEV))

        valid = float(err_log_ratio < EPSILON)
        score = 0.0
        if valid > 0:
            score = float(T0_DEV / (runtime_median * err_log_ratio + 1e-6))

        metrics.update(
            {
                "combined_score": score,
                "runtime_s": runtime_median,
                "error_log_ratio": err_log_ratio,
                "valid": valid,
                "timeout": 0.0,
                "err_rate_log_median": err_log_median,
                "err_ratio_median": float(np.nanmedian(ratios)),
                "actual_samples_median": float(np.nanmedian(samples)),
                "actual_std_median": float(np.nanmedian(stds)),
                "converged_rate": float(np.mean(converged_flags)),
                "sigma": DEV_SIGMA,
                "decoder_chase_t": 3.0,
            }
        )
        artifacts["dev_constants"] = json.dumps(
            {
                "sigma": DEV_SIGMA,
                "target_std": TARGET_STD,
                "max_samples": MAX_SAMPLES,
                "batch_size": BATCH_SIZE,
                "epsilon": EPSILON,
                "r0_dev": R0_DEV,
                "t0_dev": T0_DEV,
                "repeats": REPEATS,
            },
            ensure_ascii=False,
            indent=2,
        )
        artifacts["per_repeat"] = json.dumps(
            {
                "runtime_s": runtimes,
                "err_rate_log": err_logs,
                "err_ratio": ratios,
                "actual_samples": samples,
                "actual_std": stds,
                "converged": converged_flags,
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception as e:
        metrics["combined_score"] = 0.0
        metrics["valid"] = 0.0
        artifacts["error_message"] = str(e)
        artifacts["traceback"] = traceback.format_exc()
    finally:
        metrics["runtime_s_total"] = float(time.time() - start)

    return _wrap(metrics, artifacts)
