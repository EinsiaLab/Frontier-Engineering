#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_ITER_DIR_RE = re.compile(r"^iter_(\d+)__")
_GEN_DIR_RE = re.compile(r"^gen_(\d+)$")
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class Point:
    step: int
    score: float


@dataclass(frozen=True)
class RunTrajectory:
    task_name: str
    task_label: str
    batch_name: str
    algorithm: str
    llm: str
    run_dir: Path
    raw_points: tuple[Point, ...]
    best_points: tuple[Point, ...]


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


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        try:
            result = float(str(value).strip())
        except Exception:
            return None
    if not math.isfinite(result):
        return None
    return result


def _extract_score(metrics: Any) -> float | None:
    if not isinstance(metrics, dict):
        return None
    return _as_float(metrics.get("combined_score", metrics.get("score")))


def _aggregate_step_scores(step_scores: dict[int, list[float]], mode: str) -> list[Point]:
    points: list[Point] = []
    for step in sorted(step_scores):
        scores = [score for score in step_scores[step] if score is not None]
        if not scores:
            continue
        if mode == "last":
            agg = scores[-1]
        elif mode == "mean":
            agg = sum(scores) / len(scores)
        else:
            agg = max(scores)
        points.append(Point(step=step, score=agg))
    return points


def _to_cumulative_best(points: list[Point]) -> list[Point]:
    best_points: list[Point] = []
    best_so_far: float | None = None
    for point in sorted(points, key=lambda item: item.step):
        if best_so_far is None or point.score > best_so_far:
            best_so_far = point.score
        best_points.append(Point(step=point.step, score=best_so_far))
    return best_points


def _extract_abmcts_trajectory(run_dir: Path, *, aggregate_mode: str) -> tuple[list[Point], list[Point]]:
    algo_root = run_dir / "abmcts"
    raw_by_step: dict[int, list[float]] = defaultdict(list)
    best_by_step: dict[int, list[float]] = defaultdict(list)

    baseline_metrics = _read_json(algo_root / "baseline" / "metrics.json")
    baseline_score = _extract_score(baseline_metrics)
    if baseline_score is not None:
        raw_by_step[0].append(baseline_score)
        best_by_step[0].append(baseline_score)

    for record in _iter_jsonl(algo_root / "trace.jsonl"):
        step = _as_int(record.get("step"))
        if step is None:
            continue

        raw_score = _as_float(record.get("combined_score", record.get("score")))
        if raw_score is not None:
            raw_by_step[step].append(raw_score)

        best_score = _as_float(record.get("best_combined_score"))
        if best_score is not None:
            best_by_step[step].append(best_score)

    raw_points = _aggregate_step_scores(raw_by_step, aggregate_mode)
    best_points = _aggregate_step_scores(best_by_step, "max")
    if not best_points:
        best_points = _to_cumulative_best(raw_points)
    return raw_points, best_points


def _extract_openevolve_history(run_dir: Path) -> list[Point]:
    algo_root = run_dir / "openevolve"
    by_iteration: dict[int, list[float]] = defaultdict(list)

    for metrics_path in sorted(algo_root.glob("history/iter_*__*/metrics.json")):
        match = _ITER_DIR_RE.match(metrics_path.parent.name)
        if not match:
            continue
        step = _as_int(match.group(1))
        score = _extract_score(_read_json(metrics_path))
        if step is None or score is None:
            continue
        by_iteration[step].append(score)

    return _aggregate_step_scores(by_iteration, "max")


def _extract_openevolve_trace(run_dir: Path) -> list[Point]:
    algo_root = run_dir / "openevolve"
    by_iteration: dict[int, list[float]] = defaultdict(list)

    for record in _iter_jsonl(algo_root / "evolution_trace.jsonl"):
        iteration = _as_int(record.get("iteration"))
        if iteration is None:
            continue

        if iteration == 1:
            baseline_score = _extract_score(record.get("parent_metrics"))
            if baseline_score is not None:
                by_iteration[0].append(baseline_score)

        child_score = _extract_score(record.get("child_metrics"))
        if child_score is not None:
            by_iteration[iteration].append(child_score)

    return _aggregate_step_scores(by_iteration, "max")


def _extract_openevolve_trajectory(run_dir: Path) -> tuple[list[Point], list[Point]]:
    raw_points = _extract_openevolve_history(run_dir)
    if not raw_points:
        raw_points = _extract_openevolve_trace(run_dir)
    best_points = _to_cumulative_best(raw_points)
    return raw_points, best_points


def _extract_shinkaevolve_trajectory(run_dir: Path) -> tuple[list[Point], list[Point]]:
    algo_root = run_dir / "shinkaevolve"
    by_generation: dict[int, list[float]] = defaultdict(list)

    for metrics_path in sorted(algo_root.glob("gen_*/results/metrics.json")):
        match = _GEN_DIR_RE.match(metrics_path.parent.parent.name)
        if not match:
            continue
        step = _as_int(match.group(1))
        score = _extract_score(_read_json(metrics_path))
        if step is None or score is None:
            continue
        by_generation[step].append(score)

    raw_points = _aggregate_step_scores(by_generation, "max")
    best_points = _to_cumulative_best(raw_points)
    return raw_points, best_points


def _extract_trajectory(run_dir: Path, algorithm: str, *, abmcts_aggregate: str) -> tuple[list[Point], list[Point]]:
    if algorithm == "abmcts":
        return _extract_abmcts_trajectory(run_dir, aggregate_mode=abmcts_aggregate)
    if algorithm == "openevolve":
        return _extract_openevolve_trajectory(run_dir)
    if algorithm == "shinkaevolve":
        return _extract_shinkaevolve_trajectory(run_dir)
    return [], []


def _normalize_batch_root(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        return root.parent
    return root


def _task_matches(task_name: str, task_label: str, patterns: list[str]) -> bool:
    if not patterns:
        return True
    task_name_lower = task_name.lower()
    task_label_lower = task_label.lower()
    return any(pattern.lower() in task_name_lower or pattern.lower() in task_label_lower for pattern in patterns)


def _discover_trajectories(
    batch_root: Path,
    *,
    task_patterns: list[str],
    abmcts_aggregate: str,
) -> list[RunTrajectory]:
    trajectories: list[RunTrajectory] = []
    seen_run_dirs: set[Path] = set()

    for launcher_run in sorted(batch_root.rglob("launcher_run.json")):
        run_dir = launcher_run.parent.resolve()
        if run_dir in seen_run_dirs:
            continue
        seen_run_dirs.add(run_dir)

        record = _read_json(launcher_run)
        if not isinstance(record, dict):
            continue

        algorithm = str(record.get("algorithm") or run_dir.parent.name)
        llm = str(record.get("llm") or run_dir.name)
        task_name = str(record.get("task") or launcher_run.parents[2].name)
        task_label = launcher_run.parents[2].name

        if not _task_matches(task_name, task_label, task_patterns):
            continue

        raw_points, best_points = _extract_trajectory(
            run_dir,
            algorithm,
            abmcts_aggregate=abmcts_aggregate,
        )
        if not raw_points and not best_points:
            continue

        trajectories.append(
            RunTrajectory(
                task_name=task_name,
                task_label=task_label,
                batch_name=batch_root.name,
                algorithm=algorithm,
                llm=llm,
                run_dir=run_dir,
                raw_points=tuple(raw_points),
                best_points=tuple(best_points),
            )
        )

    return trajectories


def _sanitize_filename(name: str) -> str:
    sanitized = _SAFE_FILENAME_RE.sub("__", name.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "task"


def _series_label(trajectory: RunTrajectory, *, show_batch_name: bool) -> str:
    base = f"{trajectory.algorithm}/{trajectory.llm}"
    if show_batch_name:
        return f"{trajectory.batch_name} | {base}"
    return base


def _plot_task(
    task_name: str,
    trajectories: list[RunTrajectory],
    *,
    output_path: Path,
    plot_mode: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6), dpi=dpi)
    show_batch_name = len({trajectory.batch_name for trajectory in trajectories}) > 1

    for trajectory in sorted(
        trajectories,
        key=lambda item: (item.batch_name, item.algorithm, item.llm, str(item.run_dir)),
    ):
        label = _series_label(trajectory, show_batch_name=show_batch_name)

        if plot_mode in {"raw", "both"} and trajectory.raw_points:
            steps = [point.step for point in trajectory.raw_points]
            scores = [point.score for point in trajectory.raw_points]
            (line,) = ax.plot(
                steps,
                scores,
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=label,
            )
            color = line.get_color()
        else:
            color = None

        if plot_mode in {"best", "both"} and trajectory.best_points:
            steps = [point.step for point in trajectory.best_points]
            scores = [point.score for point in trajectory.best_points]
            best_label = label if plot_mode == "best" else f"{label} (best)"
            ax.plot(
                steps,
                scores,
                linestyle="--" if plot_mode == "both" else "-",
                linewidth=2.0,
                alpha=0.85,
                color=color,
                label=best_label,
            )

    ax.set_title(task_name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            frameon=False,
        )

    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _choose_output_paths(task_names: list[str], output_dir: Path, image_format: str) -> dict[str, Path]:
    used_stems: set[str] = set()
    mapping: dict[str, Path] = {}

    for task_name in sorted(task_names):
        stem = _sanitize_filename(task_name)
        candidate = stem
        suffix = 2
        while candidate in used_stems:
            candidate = f"{stem}_{suffix}"
            suffix += 1
        used_stems.add(candidate)
        mapping[task_name] = output_dir / f"{candidate}.{image_format}"

    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one score trajectory image per task from one or more Frontier Eval batch roots."
        )
    )
    parser.add_argument(
        "batch_paths",
        nargs="+",
        help="One or more runs/batch/<batch_id> directories.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/task_score_trajectories",
        help="Directory where images will be written.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=("raw", "best", "both"),
        default="raw",
        help="Plot raw per-step scores, cumulative best, or both.",
    )
    parser.add_argument(
        "--task-pattern",
        action="append",
        default=[],
        help="Only include tasks whose task name or task folder contains this substring. Repeatable.",
    )
    parser.add_argument(
        "--abmcts-aggregate",
        choices=("max", "last", "mean"),
        default="max",
        help=(
            "How to collapse multiple AB-MCTS evaluations that share the same step. "
            "Default: max."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("png", "jpg", "svg", "pdf"),
        default="png",
        help="Image format for the generated plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure DPI for raster outputs.",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).expanduser().resolve()
    batch_roots = [_normalize_batch_root(path) for path in args.batch_paths]

    all_trajectories: list[RunTrajectory] = []
    for batch_root in batch_roots:
        if not batch_root.is_dir():
            print(f"warning: batch root does not exist: {batch_root}", file=sys.stderr)
            continue
        all_trajectories.extend(
            _discover_trajectories(
                batch_root,
                task_patterns=args.task_pattern,
                abmcts_aggregate=args.abmcts_aggregate,
            )
        )

    if not all_trajectories:
        print("No task trajectories found.", file=sys.stderr)
        return 1

    by_task: dict[str, list[RunTrajectory]] = defaultdict(list)
    for trajectory in all_trajectories:
        by_task[trajectory.task_name].append(trajectory)

    output_paths = _choose_output_paths(list(by_task), output_dir, args.format)

    for task_name, trajectories in sorted(by_task.items()):
        _plot_task(
            task_name,
            trajectories,
            output_path=output_paths[task_name],
            plot_mode=args.plot_mode,
            dpi=args.dpi,
        )

    print(f"Wrote {len(by_task)} task plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
