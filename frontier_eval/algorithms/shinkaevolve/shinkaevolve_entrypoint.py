from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any


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


def _ensure_repo_on_syspath(repo_root: Path) -> None:
    for path in (
        repo_root,
        repo_root / "third_party" / "openevolve",
    ):
        path_str = str(path)
        if path.is_dir() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def _ensure_openevolve_evaluation_result_shim() -> None:
    try:
        from openevolve.evaluation_result import EvaluationResult as _EvaluationResult  # noqa: F401
        return
    except Exception:
        pass

    sys.modules.pop("openevolve", None)
    sys.modules.pop("openevolve.evaluation_result", None)

    package = types.ModuleType("openevolve")
    package.__path__ = []  # type: ignore[attr-defined]

    evaluation_result = types.ModuleType("openevolve.evaluation_result")

    class EvaluationResult:
        def __init__(self, metrics: dict[str, Any], artifacts: dict[str, Any] | None = None):
            self.metrics = metrics
            self.artifacts = artifacts or {}

        @classmethod
        def from_dict(cls, metrics: dict[str, Any]) -> "EvaluationResult":
            return cls(metrics=metrics)

        def to_dict(self) -> dict[str, Any]:
            return self.metrics

        def has_artifacts(self) -> bool:
            return bool(self.artifacts)

    evaluation_result.EvaluationResult = EvaluationResult
    package.evaluation_result = evaluation_result

    sys.modules["openevolve"] = package
    sys.modules["openevolve.evaluation_result"] = evaluation_result


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _extract_metrics_and_artifacts(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if result is None:
        return {}, {}

    if isinstance(result, dict):
        nested_metrics = result.get("metrics")
        nested_artifacts = result.get("artifacts")
        if isinstance(nested_metrics, dict):
            return nested_metrics, nested_artifacts if isinstance(nested_artifacts, dict) else {}
        return result, {}

    metrics = getattr(result, "metrics", None)
    artifacts = getattr(result, "artifacts", None)
    if isinstance(metrics, dict):
        return metrics, artifacts if isinstance(artifacts, dict) else {}

    raise TypeError(f"Unsupported evaluation result type: {type(result)}")


def _truncate_middle(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 64:
        return text[:limit]
    keep = max(1, (limit - 32) // 2)
    omitted = len(text) - (2 * keep)
    return text[:keep] + f"\n[... truncated {omitted} chars ...]\n" + text[-keep:]


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(value)
    return str(value)


def _artifact_basename(key: str) -> str:
    stem = key.split("::", 1)[-1]
    return Path(stem).name.lower()


def _artifact_relpath(key: str) -> str:
    return key.split("::", 1)[-1]


def _render_section(title: str, body: Any, *, limit: int) -> str:
    text = _stringify(body).strip()
    if not text:
        return ""
    return f"## {title}\n{_truncate_middle(text, limit).strip()}"


def _agent_file_sort_key(item: tuple[str, Any]) -> tuple[int, int, int, str]:
    key, value = item
    relpath = _artifact_relpath(key).lower()
    basename = _artifact_basename(key)

    if relpath.startswith("runtime/problem."):
        priority = 0
    elif relpath.startswith("baseline/solution."):
        priority = 1
    elif basename == "task.md":
        priority = 2
    elif basename == "task_zh-cn.md":
        priority = 3
    elif basename == "readme.md":
        priority = 4
    elif basename == "readme_zh-cn.md":
        priority = 5
    elif basename.endswith(".h"):
        priority = 6
    elif "interface" in basename or "config" in basename:
        priority = 7
    elif "source_manifest" in basename:
        priority = 9
    else:
        priority = 8

    return priority, len(relpath.split("/")), len(_stringify(value)), key


def _collect_error_sections(artifacts: dict[str, Any]) -> list[str]:
    sections: list[str] = []
    seen_texts: set[str] = set()
    alias_titles = {
        "user_artifact::error_message": "Error Message",
        "user_artifact::failure_summary": "Failure Summary",
    }

    for title, key, limit in (
        ("Error Message", "error_message", 2200),
        ("Failure Summary", "failure_summary", 2200),
        ("Readonly Violations", "readonly_violations", 1800),
        ("Artifacts JSON Error", "artifacts_json_error", 1200),
        ("Metrics JSON Error", "metrics_json_error", 1200),
    ):
        if key not in artifacts:
            continue
        text = _stringify(artifacts.get(key)).strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        section = _render_section(title, text, limit=limit)
        if section:
            sections.append(section)

    for key, value in sorted(artifacts.items()):
        if not isinstance(key, str):
            continue
        lowered = key.lower()
        if "error" not in lowered and "failure" not in lowered:
            continue
        if key in {"error_message", "failure_summary", "readonly_violations"}:
            continue
        text = _stringify(value).strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        title = alias_titles.get(key) or key.replace("user_artifact::", "").replace("::", " / ")
        section = _render_section(title, text, limit=2200)
        if section:
            sections.append(section)

    return sections


def _select_agent_file_sections(artifacts: dict[str, Any]) -> list[str]:
    agent_items = [
        (key, value)
        for key, value in artifacts.items()
        if isinstance(key, str) and key.startswith("agent_file::") and "::error" not in key
    ]
    if not agent_items:
        return []

    chosen: list[tuple[str, Any]] = []
    seen: set[str] = set()

    def add_first_matching(patterns: tuple[str, ...]) -> None:
        for key, value in sorted(agent_items, key=_agent_file_sort_key):
            rel = _artifact_relpath(key).lower()
            if rel in seen:
                continue
            if any(rel == pattern or rel.endswith(pattern) for pattern in patterns):
                chosen.append((key, value))
                seen.add(rel)
                return

    add_first_matching(("runtime/problem.py",))
    add_first_matching(("baseline/solution.py",))
    add_first_matching(("task.md",))
    add_first_matching(("readme.md",))

    for key, value in sorted(agent_items, key=_agent_file_sort_key):
        rel = _artifact_relpath(key).lower()
        if rel in seen:
            continue
        if len(chosen) >= 4:
            break
        chosen.append((key, value))
        seen.add(rel)

    sections: list[str] = []
    for key, value in chosen:
        rel = _artifact_relpath(key)
        limit = 4000 if rel == "runtime/problem.py" else 1800
        section = _render_section(f"Agent File: {rel}", value, limit=limit)
        if section:
            sections.append(section)
    return sections


def _primary_error_message(artifacts: dict[str, Any]) -> str:
    for key in (
        "error_message",
        "user_artifact::error_message",
        "failure_summary",
        "user_artifact::failure_summary",
    ):
        text = _stringify(artifacts.get(key)).strip()
        if text:
            return text
    return ""


def _synthesize_text_feedback(
    metrics: dict[str, Any],
    artifacts: dict[str, Any],
    *,
    max_chars: int = 16_000,
) -> str:
    existing = _stringify(metrics.get("text_feedback")).strip()
    if existing:
        return _truncate_middle(existing, max_chars)

    if not artifacts:
        return ""

    sections: list[str] = []

    metric_lines: list[str] = []
    for key in (
        "combined_score",
        "valid",
        "runtime_s",
        "timeout",
        "benchmark_returncode",
        "make_returncode",
        "mdriver_returncode",
        "errors_count",
        "testcases_passed",
        "testcases_total",
        "score_100",
        "geom_mean_ns",
    ):
        if key in metrics:
            metric_lines.append(f"{key}: {_stringify(metrics.get(key)).strip()}")
    if metric_lines:
        sections.append(_render_section("Metric Summary", "\n".join(metric_lines), limit=800))

    sections.extend(_collect_error_sections(artifacts))

    agent_files = _stringify(artifacts.get("agent_files")).strip()
    if agent_files:
        sections.append(_render_section("Agent Files", agent_files, limit=1000))

    for title, key, limit in (
        ("Constraint Summary", "constraints", 2000),
        ("Interface Contract", "interface_contract", 2000),
        ("Task Spec", "task_spec_zh_cn", 1600),
        ("Benchmark Check", "check", 600),
        ("Score Line", "score_line", 600),
    ):
        if key in artifacts:
            section = _render_section(title, artifacts.get(key), limit=limit)
            if section:
                sections.append(section)

    sections.extend(_select_agent_file_sections(artifacts))

    for title, key, limit in (
        ("Make Stdout", "make_stdout", 2500),
        ("Make Stderr", "make_stderr", 2500),
        ("Mdriver Stdout", "mdriver_stdout", 3000),
        ("Mdriver Stderr", "mdriver_stderr", 3000),
        ("Benchmark Stdout", "benchmark_stdout", 3000),
        ("Benchmark Stderr", "benchmark_stderr", 3000),
        ("CUDA Probe Stdout", "cuda_probe_stdout", 1200),
        ("CUDA Probe Stderr", "cuda_probe_stderr", 1200),
    ):
        if key in artifacts:
            section = _render_section(title, artifacts.get(key), limit=limit)
            if section:
                sections.append(section)

    feedback = "\n\n".join(section for section in sections if section).strip()
    return _truncate_middle(feedback, max_chars) if feedback else ""


def _task_cfg_from_env(task_name: str) -> dict[str, Any]:
    raw = str(os.environ.get("FRONTIER_EVAL_TASK_CFG_JSON", "") or "").strip()
    if not raw:
        return {"name": task_name}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {"name": task_name}
    if not isinstance(parsed, dict):
        return {"name": task_name}
    cfg = dict(parsed)
    cfg["name"] = str(cfg.get("name") or task_name)
    return cfg


def main(program_path: str, results_dir: str, *, task_name: str | None = None) -> int:
    repo_root = _find_repo_root()
    _ensure_repo_on_syspath(repo_root)
    _ensure_openevolve_evaluation_result_shim()

    task_name = (task_name or os.environ.get("FRONTIER_EVAL_TASK_NAME") or "").strip()
    if not task_name:
        raise RuntimeError(
            "Missing task name for ShinkaEvolve evaluator. Set env `FRONTIER_EVAL_TASK_NAME` "
            "or pass `--task_name`."
        )

    from omegaconf import OmegaConf

    from frontier_eval.registry_tasks import get_task

    task_cls = get_task(task_name)
    cfg = OmegaConf.create({"task": _task_cfg_from_env(task_name)})
    task = task_cls(cfg=cfg, repo_root=repo_root)

    program_path_p = Path(program_path).expanduser().resolve()
    results_dir_p = Path(results_dir).expanduser().resolve()
    results_dir_p.mkdir(parents=True, exist_ok=True)

    correct = False
    error_msg = ""
    metrics: dict[str, Any] = {}
    artifacts: dict[str, Any] = {}
    try:
        raw = task.evaluate_program(program_path_p)
        metrics, artifacts = _extract_metrics_and_artifacts(raw)
        text_feedback = _synthesize_text_feedback(metrics, artifacts)
        if text_feedback:
            metrics = dict(metrics)
            metrics["text_feedback"] = text_feedback

        valid = metrics.get("valid", None)
        if isinstance(valid, (int, float)) and not isinstance(valid, bool):
            correct = float(valid) > 0.0
        else:
            correct = True

        err = _primary_error_message(artifacts) if isinstance(artifacts, dict) else ""
        if err:
            error_msg = err
            if correct:
                correct = False
    except Exception as e:
        correct = False
        error_msg = str(e)
        metrics = {"combined_score": 0.0, "valid": 0.0, "error": error_msg}

    _write_json(results_dir_p / "metrics.json", metrics)
    _write_json(results_dir_p / "correct.json", {"correct": bool(correct), "error": error_msg})
    if artifacts:
        _write_json(results_dir_p / "artifacts.json", artifacts)

    # Shinka's local job runner treats non-zero exit codes as evaluation crashes and
    # may raise before loading `metrics.json`. Always exit 0 and rely on correct.json.
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Frontier Eval → ShinkaEvolve evaluation entrypoint.",
        add_help=True,
    )
    p.add_argument("--program_path", type=str, required=True)
    p.add_argument("--results_dir", type=str, required=True)
    p.add_argument("--task_name", type=str, default=None)
    # Shinka's scheduler may pass extra args (e.g. LLM settings). Ignore unknown args.
    args, _unknown = p.parse_known_args(argv)
    return args


if __name__ == "__main__":
    args = _parse_args()
    raise SystemExit(main(args.program_path, args.results_dir, task_name=args.task_name))
