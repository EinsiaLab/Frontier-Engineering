from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _resolve_symbol(spec: str) -> Any:
    module_name, sep, attr_name = spec.partition(":")
    if not sep or not module_name or not attr_name:
        raise ValueError(f"Invalid symbol spec: {spec!r} (expected module:attr)")
    module = importlib.import_module(module_name)
    value: Any = module
    for part in attr_name.split("."):
        value = getattr(value, part)
    return value


def _parse_value(raw: str) -> Any:
    if raw == "@python":
        return sys.executable
    if raw.startswith("@import:"):
        return _resolve_symbol(raw[len("@import:") :])
    if raw.startswith("@path:"):
        return Path(raw[len("@path:") :]).expanduser().resolve()

    lowered = raw.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None

    try:
        return json.loads(raw)
    except Exception:
        return raw


def _parse_key_value(raw: str) -> tuple[str, str]:
    key, sep, value = raw.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"Invalid key=value pair: {raw!r}")
    return key.strip(), value


def _normalize_result(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if hasattr(result, "metrics") and hasattr(result, "artifacts"):
        return dict(getattr(result, "metrics")), dict(getattr(result, "artifacts"))

    if isinstance(result, dict):
        raw_metrics = result.get("metrics")
        raw_artifacts = result.get("artifacts")
        if isinstance(raw_metrics, dict):
            return dict(raw_metrics), dict(raw_artifacts or {})
        return dict(result), {}

    raise TypeError(
        "Evaluator must return an EvaluationResult-like object or a dict of metrics."
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a legacy Frontier Eval evaluator and export metrics/artifacts JSON."
    )
    p.add_argument("--callable", dest="callable_spec", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--repo-root", default=None)
    p.add_argument("--metrics-out", required=True)
    p.add_argument("--artifacts-out", required=True)
    p.add_argument("--kw", action="append", default=[])
    p.add_argument("--kw-from-env", action="append", default=[])
    return p


def main(argv: list[str]) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    repo_root = None
    if args.repo_root:
        repo_root = Path(args.repo_root).expanduser().resolve()
        os.environ.setdefault("FRONTIER_ENGINEERING_ROOT", str(repo_root))
        repo_root_str = str(repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

    metrics_out = Path(args.metrics_out).expanduser().resolve()
    artifacts_out = Path(args.artifacts_out).expanduser().resolve()

    metrics: dict[str, Any] = {
        "combined_score": 0.0,
        "valid": 0.0,
    }
    artifacts: dict[str, Any] = {
        "legacy_runner_callable": args.callable_spec,
        "legacy_runner_candidate": str(Path(args.candidate).expanduser().resolve()),
    }
    if repo_root is not None:
        artifacts["legacy_runner_repo_root"] = str(repo_root)

    try:
        evaluator = _resolve_symbol(args.callable_spec)

        kwargs: dict[str, Any] = {}
        for raw in args.kw:
            key, value = _parse_key_value(raw)
            kwargs[key] = _parse_value(value)
        for raw in args.kw_from_env:
            key, env_name = _parse_key_value(raw)
            if env_name not in os.environ:
                continue
            kwargs[key] = _parse_value(os.environ[env_name])

        if repo_root is not None and "repo_root" not in kwargs:
            kwargs["repo_root"] = repo_root

        result = evaluator(args.candidate, **kwargs)
        metrics, evaluator_artifacts = _normalize_result(result)
        artifacts.update(evaluator_artifacts)
    except Exception as exc:
        metrics = {
            "combined_score": 0.0,
            "valid": 0.0,
        }
        artifacts["error_message"] = str(exc)
        artifacts["traceback"] = traceback.format_exc()

    _write_json(metrics_out, metrics)
    _write_json(artifacts_out, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
