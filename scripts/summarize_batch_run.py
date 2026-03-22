#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(text)
    except Exception:
        return None


def _first_not_none(*values: Any) -> Any | None:
    for value in values:
        if value is not None:
            return value
    return None


_STEP_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:^|[\\/])gen_(\d+)(?:[\\/]|$)"),
    re.compile(r"(?:^|[\\/])iter_(\d+)(?:[\\/]|$)"),
    re.compile(r"(?:^|[\\/])iteration_(\d+)(?:[\\/]|$)"),
    re.compile(r"(?:^|[\\/])step_(\d+)(?:[\\/]|$)"),
]
_ABMCTS_NODE_PATTERN: re.Pattern[str] = re.compile(r"(?:^|[\\/])node_(\d+)(?:[\\/]|$)")
_OPENEVOLVE_HISTORY_PATTERN: re.Pattern[str] = re.compile(r"^iter_(\d+)__(.+)$")


def _parse_step_from_path(path_str: str) -> int | None:
    raw = str(path_str or "")
    if not raw:
        return None
    for pat in _STEP_PATTERNS:
        matches = list(pat.finditer(raw))
        if not matches:
            continue
        try:
            return int(matches[-1].group(1))
        except Exception:
            continue
    return None


def _relpath(path: Path | None, root: Path, *, absolute: bool) -> str:
    if path is None:
        return ""
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    if absolute:
        return str(resolved)
    try:
        return str(resolved.relative_to(root.resolve()))
    except Exception:
        return str(resolved)


def _float_eq(a: float | None, b: float | None, *, atol: float = 1e-12, rtol: float = 1e-9) -> bool:
    if a is None or b is None:
        return False
    diff = abs(float(a) - float(b))
    limit = max(atol, rtol * max(abs(float(a)), abs(float(b))))
    return diff <= limit


def _find_program_file(*dirs: Path) -> Path | None:
    seen: set[Path] = set()
    for directory in dirs:
        candidates = [
            directory / "main.py",
            directory / "program.py",
            directory / "best_program.py",
        ]
        candidates.extend(sorted(directory.glob("main.*")))
        candidates.extend(sorted(directory.glob("program.*")))
        candidates.extend(sorted(directory.glob("best_program.*")))
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file():
                return candidate
    return None


def _scan_openevolve_history(algo_root: Path) -> list[tuple[int, str, Path]]:
    records: list[tuple[int, str, Path]] = []
    for metrics_path in algo_root.glob("history/iter_*__*/metrics.json"):
        program_dir = metrics_path.parent
        match = _OPENEVOLVE_HISTORY_PATTERN.match(program_dir.name)
        if not match:
            continue
        try:
            iteration = int(match.group(1))
        except Exception:
            continue
        records.append((iteration, match.group(2), metrics_path))
    records.sort(key=lambda item: (item[0], item[1], str(item[2])))
    return records


def _find_openevolve_baseline_metrics(algo_root: Path) -> tuple[int | None, Path | None]:
    history = _scan_openevolve_history(algo_root)
    if not history:
        return None, None

    baseline_candidates: list[tuple[int, Path]] = []
    for iteration, _program_id, metrics_path in history:
        meta = _read_json(metrics_path.parent / "meta.json")
        parent_id = meta.get("parent_id") if isinstance(meta, dict) else None
        if parent_id in (None, ""):
            baseline_candidates.append((iteration, metrics_path))

    if baseline_candidates:
        baseline_candidates.sort(key=lambda item: (item[0], str(item[1])))
        return baseline_candidates[0]

    iteration, _program_id, metrics_path = history[0]
    return iteration, metrics_path


def _find_openevolve_history_dir(
    algo_root: Path,
    *,
    iteration: int | None,
    program_id: str | None,
) -> Path | None:
    history_root = algo_root / "history"
    if not history_root.is_dir():
        return None

    if iteration is not None and program_id:
        direct = history_root / f"iter_{iteration:06d}__{program_id}"
        if direct.is_dir():
            return direct

    if program_id:
        matches = sorted(history_root.glob(f"iter_*__{program_id}"))
        if matches:
            return matches[0]

    if iteration is not None:
        matches = sorted(history_root.glob(f"iter_{iteration:06d}__*"))
        if matches:
            return matches[0]

    return None


def _extract_baseline_metrics(output_dir: Path, algorithm: str) -> dict[str, Any]:
    algo_root = output_dir / algorithm
    if algorithm == "abmcts":
        metrics_path = algo_root / "baseline" / "metrics.json"
        metrics = _read_json(metrics_path)
        if isinstance(metrics, dict):
            return {
                "step": 0,
                "score": _as_float(metrics.get("combined_score", metrics.get("score"))),
                "valid": _as_float(metrics.get("valid")),
                "runtime_s": _as_float(metrics.get("runtime_s")),
                "metrics_path": metrics_path,
            }
        return {"step": None, "score": None, "valid": None, "runtime_s": None, "metrics_path": None}

    if algorithm == "openevolve":
        step, metrics_path = _find_openevolve_baseline_metrics(algo_root)
        if metrics_path is None:
            return {"step": None, "score": None, "valid": None, "runtime_s": None, "metrics_path": None}

        metrics = _read_json(metrics_path)
        if not isinstance(metrics, dict):
            return {
                "step": step,
                "score": None,
                "valid": None,
                "runtime_s": None,
                "metrics_path": metrics_path,
            }

        return {
            "step": step,
            "score": _as_float(metrics.get("combined_score", metrics.get("score"))),
            "valid": _as_float(metrics.get("valid")),
            "runtime_s": _as_float(metrics.get("runtime_s")),
            "metrics_path": metrics_path,
        }

    if algorithm == "shinkaevolve":
        metrics_path = algo_root / "gen_0" / "results" / "metrics.json"
        step = 0
        if not metrics_path.is_file():
            return {"step": None, "score": None, "valid": None, "runtime_s": None, "metrics_path": None}

        metrics = _read_json(metrics_path)
        if not isinstance(metrics, dict):
            return {"step": step, "score": None, "valid": None, "runtime_s": None, "metrics_path": metrics_path}

        return {
            "step": step,
            "score": _as_float(metrics.get("combined_score", metrics.get("score"))),
            "valid": _as_float(metrics.get("valid")),
            "runtime_s": _as_float(metrics.get("runtime_s")),
            "metrics_path": metrics_path,
        }

    return {"step": None, "score": None, "valid": None, "runtime_s": None, "metrics_path": None}


def _load_abmcts_trace_steps(trace_path: Path) -> dict[int, int]:
    mapping: dict[int, int] = {}
    if not trace_path.is_file():
        return mapping
    try:
        lines = trace_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return mapping

    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        node_id = obj.get("node_id")
        step = obj.get("step")
        if not isinstance(node_id, int) or not isinstance(step, int):
            continue
        mapping[node_id] = step
    return mapping


def _infer_abmcts_best_from_tree(algo_root: Path, target_score: float | None) -> tuple[int | None, Path | None]:
    if target_score is None:
        return None, None

    trace_steps = _load_abmcts_trace_steps(algo_root / "trace.jsonl")
    best_step: int | None = None
    best_dir: Path | None = None

    for metrics_path in (algo_root / "tree").rglob("metrics.json"):
        metrics = _read_json(metrics_path)
        if not isinstance(metrics, dict):
            continue
        score = _as_float(metrics.get("combined_score", metrics.get("score")))
        if not _float_eq(score, target_score):
            continue
        node_match = _ABMCTS_NODE_PATTERN.search(str(metrics_path))
        node_step: int | None = None
        if node_match:
            try:
                node_step = trace_steps.get(int(node_match.group(1)))
            except Exception:
                node_step = None
        if node_step is None:
            continue
        if best_step is None or node_step < best_step:
            best_step = node_step
            best_dir = metrics_path.parent

    return best_step, best_dir


def _scan_best_metrics(algo_root: Path) -> tuple[dict[str, Any] | None, Path | None]:
    best_correct_score: float | None = None
    best_correct_metrics: dict[str, Any] | None = None
    best_correct_path: Path | None = None

    best_any_score: float | None = None
    best_any_metrics: dict[str, Any] | None = None
    best_any_path: Path | None = None

    for metrics_path in algo_root.rglob("metrics.json"):
        metrics_raw = _read_json(metrics_path)
        if not isinstance(metrics_raw, dict):
            continue
        score = _as_float(metrics_raw.get("combined_score", metrics_raw.get("score")))
        if score is None:
            continue

        correct_raw = _read_json(metrics_path.parent / "correct.json")
        correct = bool(correct_raw.get("correct")) if isinstance(correct_raw, dict) else True

        if best_any_score is None or score > best_any_score:
            best_any_score = score
            best_any_metrics = metrics_raw
            best_any_path = metrics_path

        if correct and (best_correct_score is None or score > best_correct_score):
            best_correct_score = score
            best_correct_metrics = metrics_raw
            best_correct_path = metrics_path

    if best_correct_metrics is not None:
        return best_correct_metrics, best_correct_path
    return best_any_metrics, best_any_path


def _iter_shinkaevolve_gen_metrics(algo_root: Path) -> list[tuple[int, Path, dict[str, Any], bool]]:
    records: list[tuple[int, Path, dict[str, Any], bool]] = []
    for metrics_path in algo_root.glob("gen_*/results/metrics.json"):
        gen_dir = metrics_path.parent.parent
        gen_match = re.match(r"gen_(\d+)$", gen_dir.name)
        if not gen_match:
            continue
        try:
            gen_num = int(gen_match.group(1))
        except Exception:
            continue

        metrics_raw = _read_json(metrics_path)
        if not isinstance(metrics_raw, dict):
            continue

        correct_raw = _read_json(metrics_path.parent / "correct.json")
        correct = bool(correct_raw.get("correct")) if isinstance(correct_raw, dict) else True
        records.append((gen_num, metrics_path, metrics_raw, correct))

    records.sort(key=lambda item: (item[0], str(item[1])))
    return records


def _scan_shinkaevolve_best_gen(
    algo_root: Path,
) -> tuple[dict[str, Any] | None, int | None, Path | None]:
    best_correct_score: float | None = None
    best_correct_metrics: dict[str, Any] | None = None
    best_correct_gen: int | None = None
    best_correct_path: Path | None = None

    best_any_score: float | None = None
    best_any_metrics: dict[str, Any] | None = None
    best_any_gen: int | None = None
    best_any_path: Path | None = None

    for gen_num, metrics_path, metrics_raw, correct in _iter_shinkaevolve_gen_metrics(algo_root):
        score = _as_float(metrics_raw.get("combined_score", metrics_raw.get("score")))
        if score is None:
            continue

        if best_any_score is None or score > best_any_score:
            best_any_score = score
            best_any_metrics = metrics_raw
            best_any_gen = gen_num
            best_any_path = metrics_path

        if correct and (best_correct_score is None or score > best_correct_score):
            best_correct_score = score
            best_correct_metrics = metrics_raw
            best_correct_gen = gen_num
            best_correct_path = metrics_path

    if best_correct_metrics is not None:
        return best_correct_metrics, best_correct_gen, best_correct_path
    return best_any_metrics, best_any_gen, best_any_path


def _find_shinkaevolve_first_matching_gen(
    algo_root: Path,
    target_score: float | None,
) -> tuple[int | None, Path | None, Path | None]:
    if target_score is None:
        return None, None, None

    first_correct: tuple[int, Path, Path | None] | None = None
    first_any: tuple[int, Path, Path | None] | None = None

    for gen_num, metrics_path, metrics_raw, correct in _iter_shinkaevolve_gen_metrics(algo_root):
        score = _as_float(metrics_raw.get("combined_score", metrics_raw.get("score")))
        if not _float_eq(score, target_score):
            continue

        results_dir = metrics_path.parent
        program_path = _find_program_file(results_dir.parent, results_dir)

        if first_any is None:
            first_any = (gen_num, results_dir, program_path)
        if correct and first_correct is None:
            first_correct = (gen_num, results_dir, program_path)

    chosen = first_correct if first_correct is not None else first_any
    if chosen is None:
        return None, None, None
    return chosen


def _extract_best_info(output_dir: Path, algorithm: str) -> dict[str, Any]:
    algo_root = output_dir / algorithm
    info_path = algo_root / "best" / "best_program_info.json"
    info = _read_json(info_path)
    if isinstance(info, dict) and isinstance(info.get("metrics"), dict):
        metrics = info["metrics"]
        score = _as_float(metrics.get("combined_score", metrics.get("score")))
        program_path_raw = str(info.get("program_path") or "")
        results_dir_raw = str(info.get("results_dir") or "")
        step = _first_not_none(
            _as_int(info.get("best_step")),
            _as_int(info.get("iteration")),
            _as_int(info.get("generation")),
            _parse_step_from_path(program_path_raw),
            _parse_step_from_path(results_dir_raw),
        )
        program_path = Path(program_path_raw) if program_path_raw else None
        results_dir = Path(results_dir_raw) if results_dir_raw else None

        if algorithm == "shinkaevolve" and algo_root.is_dir():
            inferred_step, inferred_results, inferred_program = _find_shinkaevolve_first_matching_gen(
                algo_root,
                score,
            )
            if inferred_step is not None:
                step = inferred_step
                if inferred_results is not None:
                    results_dir = inferred_results
                if inferred_program is not None:
                    program_path = inferred_program
            elif step is None:
                _, best_gen, _ = _scan_shinkaevolve_best_gen(algo_root)
                if best_gen is not None:
                    step = best_gen
                    inferred_results = algo_root / f"gen_{best_gen}" / "results"
                    if program_path is None:
                        program_path = _find_program_file(
                            algo_root / f"gen_{best_gen}",
                            inferred_results,
                        )
                    if results_dir is None and inferred_results.is_dir():
                        results_dir = inferred_results
        elif step is None and algorithm == "abmcts" and algo_root.is_dir():
            baseline_metrics = _read_json(algo_root / "baseline" / "metrics.json")
            baseline_score = None
            if isinstance(baseline_metrics, dict):
                baseline_score = _as_float(
                    baseline_metrics.get("combined_score", baseline_metrics.get("score"))
                )

            if _float_eq(score, baseline_score):
                step = 0
                if results_dir is None and (algo_root / "baseline").is_dir():
                    results_dir = algo_root / "baseline"
            else:
                inferred_step, inferred_results = _infer_abmcts_best_from_tree(algo_root, score)
                if inferred_step is not None:
                    step = inferred_step
                if results_dir is None and inferred_results is not None:
                    results_dir = inferred_results
        elif algorithm == "openevolve" and algo_root.is_dir():
            program_id = str(info.get("id") or "").strip() or None
            history_dir = _find_openevolve_history_dir(
                algo_root,
                iteration=step,
                program_id=program_id,
            )
            if results_dir is None and history_dir is not None:
                results_dir = history_dir
            if program_path is None:
                search_dirs: list[Path] = [algo_root / "best"]
                if history_dir is not None:
                    search_dirs.append(history_dir)
                program_path = _find_program_file(*search_dirs)

        if results_dir is not None:
            resolved_metrics = _read_json(results_dir / "metrics.json")
            if isinstance(resolved_metrics, dict):
                metrics = resolved_metrics
                score = _as_float(metrics.get("combined_score", metrics.get("score")))

        return {
            "step": step,
            "score": score,
            "valid": _as_float(metrics.get("valid")),
            "runtime_s": _as_float(metrics.get("runtime_s")),
            "info_path": info_path if info_path.is_file() else None,
            "program_path": program_path,
            "results_dir": results_dir,
        }

    best_metrics, best_metrics_path = _scan_best_metrics(algo_root) if algo_root.is_dir() else (None, None)
    if not isinstance(best_metrics, dict) or best_metrics_path is None:
        return {
            "step": None,
            "score": None,
            "valid": None,
            "runtime_s": None,
            "info_path": info_path if info_path.is_file() else None,
            "program_path": None,
            "results_dir": None,
        }

    step = _parse_step_from_path(str(best_metrics_path))
    results_dir = best_metrics_path.parent
    program_path = _find_program_file(results_dir, results_dir.parent)

    return {
        "step": step,
        "score": _as_float(best_metrics.get("combined_score", best_metrics.get("score"))),
        "valid": _as_float(best_metrics.get("valid")),
        "runtime_s": _as_float(best_metrics.get("runtime_s")),
        "info_path": info_path if info_path.is_file() else None,
        "program_path": program_path,
        "results_dir": results_dir,
    }


def _load_summary_records(summary_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with summary_path.open("r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                print(f"warning: failed to parse JSON at {summary_path}:{line_num}", file=sys.stderr)
                continue
            if not isinstance(obj, dict):
                continue
            records.append(obj)
    return records


def _resolve_batch_root(path: Path) -> tuple[Path, Path]:
    p = path.expanduser()
    if p.is_file():
        return p.parent.resolve(), p.resolve()
    batch_root = p.resolve()
    summary_path = batch_root / "summary.jsonl"
    return batch_root, summary_path


def _scan_launcher_results(batch_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in batch_root.rglob("launcher_result.json"):
        obj = _read_json(path)
        if not isinstance(obj, dict):
            continue
        output_dir = obj.get("output_dir", None)
        if not output_dir:
            continue
        records.append(obj)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a summary.csv for a Frontier Eval batch run directory (runs/batch/<batch_id>)."
        )
    )
    parser.add_argument(
        "batch_path",
        type=str,
        help="Batch root directory (or summary.jsonl path).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: <batch_root>/summary.csv).",
    )
    parser.add_argument(
        "--absolute-paths",
        action="store_true",
        help="Write absolute paths in CSV (default: relative to batch_root when possible).",
    )

    args = parser.parse_args(argv)

    batch_root, summary_path = _resolve_batch_root(Path(args.batch_path))
    out_path = Path(args.output).expanduser().resolve() if args.output else (batch_root / "summary.csv")

    records: list[dict[str, Any]]
    if summary_path.is_file():
        records = _load_summary_records(summary_path)
    else:
        records = _scan_launcher_results(batch_root)
        if records:
            print(
                f"warning: summary.jsonl not found; using launcher_result.json scan under {batch_root}",
                file=sys.stderr,
            )
        else:
            print(f"error: summary.jsonl not found and no launcher_result.json under {batch_root}", file=sys.stderr)
            return 2

    records.sort(key=lambda r: (str(r.get("task") or ""), str(r.get("algorithm") or ""), str(r.get("llm") or "")))

    fieldnames = [
        "task",
        "algorithm",
        "llm",
        "run_dir",
        "returncode",
        "elapsed_s",
        "baseline_step",
        "baseline_score",
        "baseline_valid",
        "baseline_runtime_s",
        "baseline_metrics_path",
        "best_step",
        "best_score",
        "best_valid",
        "best_runtime_s",
        "best_info_path",
        "best_program_path",
        "best_results_dir",
        "delta_score",
        "delta_pct_vs_baseline",
        "notes",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in records:
            task = str(rec.get("task") or "")
            algorithm = str(rec.get("algorithm") or "")
            llm = rec.get("llm", "")
            llm_str = "" if llm is None else str(llm)

            output_dir_raw = rec.get("output_dir")
            output_dir = Path(str(output_dir_raw)).expanduser() if output_dir_raw else None

            notes: list[str] = []
            if output_dir is None:
                notes.append("missing_output_dir")
                output_dir = batch_root

            baseline = _extract_baseline_metrics(output_dir, algorithm=algorithm)
            best = _extract_best_info(output_dir, algorithm=algorithm)

            baseline_score = baseline.get("score")
            best_score = best.get("score")
            baseline_valid = _as_float(baseline.get("valid"))
            best_valid = _as_float(best.get("valid"))
            delta_score: float | None = None
            if isinstance(baseline_score, (int, float)) and isinstance(best_score, (int, float)):
                delta_score = float(best_score) - float(baseline_score)
            delta_pct_vs_baseline: float | None = None
            if (
                delta_score is not None
                and isinstance(baseline_score, (int, float))
                and abs(float(baseline_score)) > 0.0
            ):
                delta_pct_vs_baseline = (delta_score / abs(float(baseline_score))) * 100.0

            if baseline.get("score") is None:
                notes.append("missing_baseline_score")
            if best.get("score") is None:
                notes.append("missing_best_score")
            if baseline_valid == 0.0:
                notes.append("baseline_valid_zero")
            if best_valid == 0.0:
                notes.append("best_valid_zero")
            if (
                delta_score is not None
                and isinstance(baseline_score, (int, float))
                and float(baseline_score) == 0.0
            ):
                notes.append("delta_pct_unavailable_zero_baseline")
            if best.get("step") is None:
                notes.append("missing_best_step")
            returncode = rec.get("returncode", "")
            if returncode not in ("", None, 0, "0"):
                notes.append("returncode_nonzero")

            row = {
                "task": task,
                "algorithm": algorithm,
                "llm": llm_str,
                "run_dir": _relpath(output_dir, batch_root, absolute=bool(args.absolute_paths)),
                "returncode": returncode,
                "elapsed_s": rec.get("elapsed_s", ""),
                "baseline_step": baseline.get("step", ""),
                "baseline_score": baseline.get("score", ""),
                "baseline_valid": baseline.get("valid", ""),
                "baseline_runtime_s": baseline.get("runtime_s", ""),
                "baseline_metrics_path": _relpath(
                    baseline.get("metrics_path"), batch_root, absolute=bool(args.absolute_paths)
                ),
                "best_step": best.get("step", ""),
                "best_score": best.get("score", ""),
                "best_valid": best.get("valid", ""),
                "best_runtime_s": best.get("runtime_s", ""),
                "best_info_path": _relpath(best.get("info_path"), batch_root, absolute=bool(args.absolute_paths)),
                "best_program_path": _relpath(
                    best.get("program_path"), batch_root, absolute=bool(args.absolute_paths)
                ),
                "best_results_dir": _relpath(best.get("results_dir"), batch_root, absolute=bool(args.absolute_paths)),
                "delta_score": "" if delta_score is None else delta_score,
                "delta_pct_vs_baseline": "" if delta_pct_vs_baseline is None else delta_pct_vs_baseline,
                "notes": ";".join(notes),
            }
            writer.writerow(row)

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
