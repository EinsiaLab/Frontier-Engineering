#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


MODEL_ORDER = (
    "claude-opus-4.6",
    "gemini-3.1-pro-preview",
    "gpt-5.4",
    "grok-4.20",
    "seed-2.0-pro",
)

MODEL_COLORS = {
    # Okabe-Ito-style colorblind-friendlier palette.
    "claude-opus-4.6": "#56B4E9",
    "gemini-3.1-pro-preview": "#0072B2",
    "gpt-5.4": "#D55E00",
    "grok-4.20": "#009E73",
    "seed-2.0-pro": "#CC79A7",
}

ALGORITHM_ORDER = (
    "openevolve",
    "shinkaevolve",
    "abmcts",
)

ALGORITHM_COLORS = {
    "openevolve": "#0072B2",
    "shinkaevolve": "#E69F00",
    "abmcts": "#009E73",
}

_RUN_NAME_MODEL_RE = re.compile(r"_i\d+_(.*?)__")
_RUN_NAME_TS_RE = re.compile(r"__(\d{8}_\d{6})__")
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_GEN_DIR_RE = re.compile(r"^gen_(\d+)$")
_INVALID_FAILURE_SCORE = -1e17
_SUPPORTED_ALGORITHMS = frozenset(ALGORITHM_ORDER)


@dataclass(frozen=True)
class HistoryEntry:
    iteration: int
    program_id: str
    parent_id: str | None
    generation: int | None
    timestamp: float | None
    score: float | None
    valid: float | None
    runtime_s: float | None


@dataclass(frozen=True)
class BatchTaskRun:
    batch_run_name: str
    batch_run_dir: Path
    task_name: str
    algorithm_name: str
    provider_name: str
    model_name: str
    output_dir: Path
    algo_dir: Path
    history_index_path: Path
    launcher_run_path: Path
    launcher_result_path: Path
    best_info_path: Path


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
    except Exception:
        return []
    return records


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        try:
            text = str(value).strip()
            if not text:
                return None
            result = float(text)
        except Exception:
            return None
    if not math.isfinite(result):
        return None
    return result


def _normalize_score(value: Any) -> float | None:
    score = _as_float(value)
    if score is not None and score <= _INVALID_FAILURE_SCORE:
        return None
    return score


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(text)
    except Exception:
        return None


def _iso_to_epoch(text: str | None) -> float | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        return None


def _epoch_to_iso(epoch_s: float | None) -> str:
    if epoch_s is None:
        return ""
    try:
        return datetime.fromtimestamp(epoch_s).isoformat(timespec="seconds")
    except Exception:
        return ""


def safe_filename(name: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", name.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed"


def format_number(value: float | None, *, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}g}"


def format_pct(value: float | None, *, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}%"


def normalize_model_name(raw: str | None, *, fallback: str) -> str:
    text = (raw or "").strip()
    if text in MODEL_ORDER:
        return text
    if text.startswith("litellm__"):
        return text.split("__", 1)[1]
    if text.startswith("ark__"):
        text = text.split("__", 1)[1]
        if text.startswith("seed-2.0-pro"):
            return "seed-2.0-pro"
        return text
    if "seed-2.0-pro" in text:
        return "seed-2.0-pro"
    if fallback in MODEL_ORDER and text not in MODEL_ORDER:
        return fallback
    if text:
        return text
    return fallback


def model_sort_key(model_name: str) -> tuple[int, str]:
    try:
        return (MODEL_ORDER.index(model_name), model_name)
    except ValueError:
        return (len(MODEL_ORDER), model_name)


def algorithm_sort_key(algorithm_name: str) -> tuple[int, str]:
    try:
        return (ALGORITHM_ORDER.index(algorithm_name), algorithm_name)
    except ValueError:
        return (len(ALGORITHM_ORDER), algorithm_name)


def parse_run_dir_timestamp(run_name: str) -> float | None:
    match = _RUN_NAME_TS_RE.search(run_name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S").timestamp()
    except Exception:
        return None


def _has_algorithm_outputs(algo_dir: Path, algorithm_name: str) -> bool:
    if algorithm_name == "openevolve":
        return (algo_dir / "history" / "index.jsonl").is_file() or (algo_dir / "evolution_trace.jsonl").is_file()
    if algorithm_name == "shinkaevolve":
        return any(algo_dir.glob("gen_*/results/metrics.json"))
    if algorithm_name == "abmcts":
        return (algo_dir / "trace.jsonl").is_file() or (algo_dir / "baseline" / "metrics.json").is_file()
    return False


def scan_batch_task_runs(batch_dir: Path) -> list[BatchTaskRun]:
    runs: list[BatchTaskRun] = []
    for batch_run_dir in sorted(p for p in batch_dir.iterdir() if p.is_dir()):
        batch_run_name = batch_run_dir.name
        inferred_model = normalize_model_name(
            _RUN_NAME_MODEL_RE.search(batch_run_name).group(1) if _RUN_NAME_MODEL_RE.search(batch_run_name) else None,
            fallback=batch_run_name,
        )
        for task_dir in sorted(p for p in batch_run_dir.iterdir() if p.is_dir()):
            algorithm_roots = [p for p in sorted(task_dir.iterdir()) if p.is_dir() and p.name in _SUPPORTED_ALGORITHMS]
            for provider_root in algorithm_roots:
                algorithm_name = provider_root.name
                for provider_dir in sorted(p for p in provider_root.iterdir() if p.is_dir()):
                    algo_dir = provider_dir / algorithm_name
                    if not _has_algorithm_outputs(algo_dir, algorithm_name):
                        continue
                    history_index_path = algo_dir / "history" / "index.jsonl"
                    launcher_result_path = provider_dir / "launcher_result.json"
                    launcher_run_path = provider_dir / "launcher_run.json"
                    launcher_result = _read_json(launcher_result_path)
                    launcher_run = _read_json(launcher_run_path)
                    model_name = normalize_model_name(
                        (launcher_result or {}).get("model")
                        or (launcher_run or {}).get("model")
                        or (launcher_result or {}).get("llm")
                        or (launcher_run or {}).get("llm"),
                        fallback=inferred_model,
                    )
                    runs.append(
                        BatchTaskRun(
                            batch_run_name=batch_run_name,
                            batch_run_dir=batch_run_dir,
                            task_name=task_dir.name,
                            algorithm_name=algorithm_name,
                            provider_name=provider_dir.name,
                            model_name=model_name,
                            output_dir=provider_dir,
                            algo_dir=algo_dir,
                            history_index_path=history_index_path,
                            launcher_run_path=launcher_run_path,
                            launcher_result_path=launcher_result_path,
                            best_info_path=algo_dir / "best" / "best_program_info.json",
                        )
                    )
    return runs


def load_history_entries(history_index_path: Path) -> list[HistoryEntry]:
    entries: list[HistoryEntry] = []
    history_dir = history_index_path.parent
    meta_cache: dict[Path, dict[str, Any] | None] = {}
    for record in _iter_jsonl(history_index_path):
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        iteration = _as_int(record.get("iteration"))
        program_id = str(record.get("id", "")).strip()
        if iteration is None or not program_id:
            continue
        timestamp = _as_float(record.get("timestamp"))
        if timestamp is None:
            program_dir = history_dir / f"iter_{iteration:06d}__{program_id}"
            if not program_dir.is_dir():
                matches = sorted(history_dir.glob(f"iter_*__{program_id}"))
                program_dir = matches[0] if matches else None
            if program_dir is not None:
                meta_path = program_dir / "meta.json"
                if meta_path not in meta_cache:
                    meta_obj = _read_json(meta_path)
                    meta_cache[meta_path] = meta_obj if isinstance(meta_obj, dict) else None
                timestamp = _as_float((meta_cache.get(meta_path) or {}).get("timestamp"))
        entries.append(
            HistoryEntry(
                iteration=iteration,
                program_id=program_id,
                parent_id=str(record.get("parent_id")).strip() if record.get("parent_id") not in (None, "") else None,
                generation=_as_int(record.get("generation")),
                timestamp=timestamp,
                score=_normalize_score(metrics.get("combined_score", metrics.get("score"))),
                valid=_as_float(metrics.get("valid")),
                runtime_s=_as_float(metrics.get("runtime_s")),
            )
        )
    entries.sort(key=lambda item: (item.iteration, item.timestamp if item.timestamp is not None else float("inf"), item.program_id))
    return entries


def _path_timestamp(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    try:
        return path.stat().st_mtime
    except Exception:
        return None


def _load_shinkaevolve_entries(algo_dir: Path) -> list[HistoryEntry]:
    entries: list[HistoryEntry] = []
    previous_program_id: str | None = None
    for metrics_path in sorted(algo_dir.glob("gen_*/results/metrics.json")):
        gen_dir = metrics_path.parent.parent
        match = _GEN_DIR_RE.match(gen_dir.name)
        if not match:
            continue
        generation = _as_int(match.group(1))
        metrics = _read_json(metrics_path)
        if generation is None or not isinstance(metrics, dict):
            continue
        program_id = gen_dir.name
        score = _normalize_score(metrics.get("combined_score", metrics.get("score")))
        entries.append(
            HistoryEntry(
                iteration=generation,
                program_id=program_id,
                parent_id=previous_program_id,
                generation=generation,
                timestamp=_path_timestamp(metrics_path),
                score=score,
                valid=_as_float(metrics.get("valid")),
                runtime_s=_as_float(metrics.get("runtime_s")),
            )
        )
        previous_program_id = program_id
    entries.sort(key=lambda item: (item.iteration, item.timestamp if item.timestamp is not None else float("inf"), item.program_id))
    return entries


def _load_openevolve_trace_entries(algo_dir: Path) -> list[HistoryEntry]:
    entries: list[HistoryEntry] = []
    baseline_added = False
    for record in _iter_jsonl(algo_dir / "evolution_trace.jsonl"):
        iteration = _as_int(record.get("iteration"))
        timestamp = _as_float(record.get("timestamp"))
        if iteration is None:
            continue

        parent_id = str(record.get("parent_id", "")).strip()
        child_id = str(record.get("child_id", "")).strip()
        parent_metrics = record.get("parent_metrics")
        child_metrics = record.get("child_metrics")

        if not baseline_added and parent_id and isinstance(parent_metrics, dict):
            entries.append(
                HistoryEntry(
                    iteration=0,
                    program_id=parent_id,
                    parent_id=None,
                    generation=0,
                    timestamp=timestamp,
                    score=_normalize_score(parent_metrics.get("combined_score", parent_metrics.get("score"))),
                    valid=_as_float(parent_metrics.get("valid")),
                    runtime_s=_as_float(parent_metrics.get("runtime_s")),
                )
            )
            baseline_added = True

        if not child_id or not isinstance(child_metrics, dict):
            continue
        entries.append(
            HistoryEntry(
                iteration=iteration,
                program_id=child_id,
                parent_id=parent_id or None,
                generation=iteration,
                timestamp=timestamp,
                score=_normalize_score(child_metrics.get("combined_score", child_metrics.get("score"))),
                valid=_as_float(child_metrics.get("valid")),
                runtime_s=_as_float(child_metrics.get("runtime_s")),
            )
        )

    entries.sort(key=lambda item: (item.iteration, item.timestamp if item.timestamp is not None else float("inf"), item.program_id))
    return entries


def _load_abmcts_entries(algo_dir: Path) -> list[HistoryEntry]:
    entries: list[HistoryEntry] = []
    baseline_metrics_path = algo_dir / "baseline" / "metrics.json"
    baseline_metrics = _read_json(baseline_metrics_path)
    if isinstance(baseline_metrics, dict):
        entries.append(
            HistoryEntry(
                iteration=0,
                program_id="baseline",
                parent_id=None,
                generation=0,
                timestamp=_path_timestamp(baseline_metrics_path),
                score=_normalize_score(baseline_metrics.get("combined_score", baseline_metrics.get("score"))),
                valid=_as_float(baseline_metrics.get("valid")),
                runtime_s=_as_float(baseline_metrics.get("runtime_s")),
            )
        )

    for record in _iter_jsonl(algo_dir / "trace.jsonl"):
        step = _as_int(record.get("step"))
        if step is None:
            continue
        node_id = _as_int(record.get("node_id"))
        metrics_path = algo_dir / "tree" / f"node_{node_id:06d}" / "metrics.json" if node_id is not None else None
        metrics = _read_json(metrics_path) if metrics_path is not None and metrics_path.is_file() else None
        score = _normalize_score(((metrics or {}).get("combined_score", (metrics or {}).get("score"))))
        if score is None:
            score = _normalize_score(record.get("combined_score", record.get("score")))
        parent_node_id = _as_int(record.get("parent_id"))
        entries.append(
            HistoryEntry(
                iteration=step,
                program_id=f"node_{node_id:06d}" if node_id is not None else f"step_{step:06d}",
                parent_id=f"node_{parent_node_id:06d}" if parent_node_id is not None and parent_node_id >= 0 else "baseline",
                generation=step,
                timestamp=_path_timestamp(metrics_path) or _path_timestamp(algo_dir / "trace.jsonl"),
                score=score,
                valid=_as_float((metrics or {}).get("valid")),
                runtime_s=_as_float((metrics or {}).get("runtime_s")),
            )
        )

    entries.sort(key=lambda item: (item.iteration, item.timestamp if item.timestamp is not None else float("inf"), item.program_id))
    return entries


def load_run_entries(run: BatchTaskRun) -> list[HistoryEntry]:
    if run.algorithm_name == "openevolve":
        if run.history_index_path.is_file():
            return load_history_entries(run.history_index_path)
        return _load_openevolve_trace_entries(run.algo_dir)
    if run.algorithm_name == "shinkaevolve":
        return _load_shinkaevolve_entries(run.algo_dir)
    if run.algorithm_name == "abmcts":
        return _load_abmcts_entries(run.algo_dir)
    return load_history_entries(run.history_index_path)


def _score_sort_key(entry: HistoryEntry) -> tuple[float, float, int]:
    score = entry.score if entry.score is not None else float("-inf")
    ts = entry.timestamp if entry.timestamp is not None else float("inf")
    return (score, -ts, -entry.iteration)


def select_best_entry(entries: list[HistoryEntry]) -> HistoryEntry | None:
    if not entries:
        return None
    max_score = max((entry.score for entry in entries if entry.score is not None), default=None)
    if max_score is None:
        return min(entries, key=lambda entry: (entry.iteration, entry.timestamp if entry.timestamp is not None else float("inf"), entry.program_id))
    best_candidates = [entry for entry in entries if entry.score == max_score]
    return min(
        best_candidates,
        key=lambda entry: (
            entry.timestamp if entry.timestamp is not None else float("inf"),
            entry.iteration,
            entry.program_id,
        ),
    )


def group_entries_by_iteration(entries: list[HistoryEntry]) -> dict[int, list[HistoryEntry]]:
    grouped: dict[int, list[HistoryEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.iteration, []).append(entry)
    return grouped


def chronological_entries(entries: list[HistoryEntry]) -> list[HistoryEntry]:
    return sorted(
        entries,
        key=lambda entry: (
            entry.timestamp is None,
            entry.timestamp if entry.timestamp is not None else float("inf"),
            entry.iteration,
            entry.generation if entry.generation is not None else -1,
            entry.program_id,
        ),
    )


def select_root_entry(entries: list[HistoryEntry]) -> HistoryEntry | None:
    roots = [entry for entry in entries if entry.parent_id is None]
    if not roots:
        return None
    return min(
        roots,
        key=lambda entry: (
            entry.timestamp is None,
            entry.timestamp if entry.timestamp is not None else float("inf"),
            entry.iteration,
            entry.program_id,
        ),
    )


def iteration_best_entries(entries: list[HistoryEntry]) -> list[HistoryEntry]:
    grouped = group_entries_by_iteration(entries)
    return [select_best_entry(grouped[iteration]) for iteration in sorted(grouped) if grouped[iteration]]


def cumulative_best_points(entries: list[HistoryEntry]) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    best_so_far: float | None = None
    for entry in iteration_best_entries(entries):
        if entry is None or entry.score is None:
            continue
        if best_so_far is None or entry.score > best_so_far:
            best_so_far = entry.score
        points.append((entry.iteration, best_so_far))
    return points


def iteration_max_points(entries: list[HistoryEntry]) -> list[tuple[int, float]]:
    points: list[tuple[int, float]] = []
    for entry in iteration_best_entries(entries):
        if entry is None or entry.score is None:
            continue
        points.append((entry.iteration, entry.score))
    return points


def _load_launcher_times(run: BatchTaskRun) -> tuple[float | None, float | None]:
    launcher_run = _read_json(run.launcher_run_path)
    launcher_result = _read_json(run.launcher_result_path)
    start_epoch = _iso_to_epoch((launcher_run or {}).get("start_time"))
    elapsed_s = _as_float((launcher_result or {}).get("elapsed_s"))
    if start_epoch is not None and elapsed_s is not None:
        return start_epoch, start_epoch + elapsed_s
    return start_epoch, None


def dedupe_latest_runs(runs: list[BatchTaskRun]) -> list[BatchTaskRun]:
    latest: dict[tuple[str, str, str], BatchTaskRun] = {}
    for run in runs:
        key = (run.task_name, run.algorithm_name, run.model_name)
        current = latest.get(key)
        if current is None:
            latest[key] = run
            continue

        current_start, _ = _load_launcher_times(current)
        run_start, _ = _load_launcher_times(run)
        current_rank = (
            current_start if current_start is not None else parse_run_dir_timestamp(current.batch_run_name) or float("-inf"),
            current.batch_run_name,
        )
        run_rank = (
            run_start if run_start is not None else parse_run_dir_timestamp(run.batch_run_name) or float("-inf"),
            run.batch_run_name,
        )
        if run_rank > current_rank:
            latest[key] = run
    return sorted(
        latest.values(),
        key=lambda item: (
            item.task_name,
            algorithm_sort_key(item.algorithm_name),
            model_sort_key(item.model_name),
            item.batch_run_name,
        ),
    )


def _compute_elapsed_s(
    run: BatchTaskRun,
    entries: list[HistoryEntry],
    best_info: dict[str, Any] | None,
) -> tuple[float | None, str]:
    launcher_result = _read_json(run.launcher_result_path)
    launcher_elapsed = _as_float((launcher_result or {}).get("elapsed_s"))
    if launcher_elapsed is not None:
        return launcher_elapsed, "launcher_result.elapsed_s"

    start_epoch, launcher_end_epoch = _load_launcher_times(run)
    if start_epoch is not None and launcher_end_epoch is not None:
        return launcher_end_epoch - start_epoch, "launcher_run+launcher_result"

    observed_timestamps = [entry.timestamp for entry in entries if entry.timestamp is not None]
    best_timestamp = _as_float((best_info or {}).get("timestamp"))
    if best_timestamp is not None:
        observed_timestamps.append(best_timestamp)
    latest_observed = max(observed_timestamps) if observed_timestamps else None

    if start_epoch is not None and latest_observed is not None:
        return max(0.0, latest_observed - start_epoch), "launcher_run_to_latest_observed"

    if observed_timestamps:
        return max(observed_timestamps) - min(observed_timestamps), "history_timestamp_span"

    return None, ""


def summarize_run(run: BatchTaskRun) -> dict[str, Any]:
    entries = load_run_entries(run)
    launcher_run = _read_json(run.launcher_run_path)
    launcher_result = _read_json(run.launcher_result_path)
    best_info = _read_json(run.best_info_path)
    best_info_timestamp = _as_float((best_info or {}).get("timestamp"))

    iter0_entries = [entry for entry in entries if entry.iteration == 0]
    iter0_best = select_best_entry(iter0_entries)
    best_entry = select_best_entry(entries)
    latest_iteration = max((entry.iteration for entry in entries), default=None)
    iterations_seen = len({entry.iteration for entry in entries})
    valid_entries = sum(1 for entry in entries if (entry.valid or 0.0) > 0.0)
    start_epoch, _launcher_end_epoch = _load_launcher_times(run)
    elapsed_s, elapsed_source = _compute_elapsed_s(run, entries, best_info)
    observed_epochs = [entry.timestamp for entry in entries if entry.timestamp is not None]
    if best_info_timestamp is not None:
        observed_epochs.append(best_info_timestamp)
    latest_observed_epoch = max(observed_epochs, default=None)
    best_file_metrics = (best_info or {}).get("metrics") or {}
    best_file_score = _normalize_score(best_file_metrics.get("combined_score", best_file_metrics.get("score")))
    best_file_iteration = _as_int((best_info or {}).get("iteration"))
    best_file_generation = _as_int((best_info or {}).get("generation"))
    best_found_epoch = best_entry.timestamp if best_entry is not None else None
    if best_found_epoch is None:
        best_found_epoch = best_info_timestamp
    time_to_best_s = None
    if start_epoch is not None and best_found_epoch is not None:
        time_to_best_s = max(0.0, best_found_epoch - start_epoch)

    iter0_score = iter0_best.score if iter0_best is not None else None
    best_score = best_entry.score if best_entry is not None else None
    score_gain_abs = None
    score_gain_pct = None
    if iter0_score is not None and best_score is not None:
        score_gain_abs = best_score - iter0_score
        if iter0_score != 0:
            score_gain_pct = score_gain_abs / iter0_score * 100.0

    return {
        "task_name": run.task_name,
        "algorithm_name": run.algorithm_name,
        "model_name": run.model_name,
        "provider_name": run.provider_name,
        "batch_run_name": run.batch_run_name,
        "batch_run_dir": str(run.batch_run_dir),
        "output_dir": str(run.output_dir),
        "history_index_path": str(run.history_index_path),
        "best_info_path": str(run.best_info_path) if run.best_info_path.is_file() else "",
        "launcher_run_path": str(run.launcher_run_path) if run.launcher_run_path.is_file() else "",
        "launcher_result_path": str(run.launcher_result_path) if run.launcher_result_path.is_file() else "",
        "launcher_start_time": (launcher_run or {}).get("start_time", ""),
        "launcher_returncode": _as_int((launcher_result or {}).get("returncode")),
        "launcher_completed": 1 if run.launcher_result_path.is_file() else 0,
        "history_entries": len(entries),
        "valid_entries": valid_entries,
        "iterations_seen": iterations_seen,
        "latest_iteration_seen": latest_iteration,
        "iter0_candidates": len(iter0_entries),
        "iter0_best_id": iter0_best.program_id if iter0_best is not None else "",
        "iter0_best_generation": iter0_best.generation if iter0_best is not None else None,
        "iter0_best_score": iter0_score,
        "iter0_best_valid": iter0_best.valid if iter0_best is not None else None,
        "iter0_best_runtime_s": iter0_best.runtime_s if iter0_best is not None else None,
        "best_id": best_entry.program_id if best_entry is not None else "",
        "best_iteration": best_entry.iteration if best_entry is not None else None,
        "best_generation": best_entry.generation if best_entry is not None else None,
        "best_score": best_score,
        "best_valid": best_entry.valid if best_entry is not None else None,
        "best_runtime_s": best_entry.runtime_s if best_entry is not None else None,
        "best_timestamp_iso": _epoch_to_iso(best_found_epoch),
        "best_file_present": 1 if run.best_info_path.is_file() else 0,
        "best_file_iteration": best_file_iteration,
        "best_file_generation": best_file_generation,
        "best_file_score": best_file_score,
        "best_is_iter0": 1 if best_entry is not None and best_entry.iteration == 0 else 0,
        "score_gain_abs": score_gain_abs,
        "score_gain_pct": score_gain_pct,
        "elapsed_s": elapsed_s,
        "elapsed_h": elapsed_s / 3600.0 if elapsed_s is not None else None,
        "elapsed_source": elapsed_source,
        "time_to_best_s": time_to_best_s,
        "time_to_best_h": time_to_best_s / 3600.0 if time_to_best_s is not None else None,
        "latest_observed_time_iso": _epoch_to_iso(latest_observed_epoch),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
