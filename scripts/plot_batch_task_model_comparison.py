#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
from typing import Any

from batch_model_analysis_lib import (
    ALGORITHM_COLORS,
    algorithm_sort_key,
    MODEL_COLORS,
    chronological_entries,
    dedupe_latest_runs,
    format_number,
    format_pct,
    load_run_entries,
    model_sort_key,
    safe_filename,
    scan_batch_task_runs,
    select_best_entry,
    select_root_entry,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one SVG per task comparing model or algorithm trajectories from runs/batch. "
            "The default x-axis is chronological discovery order, with score on the main panel "
            "and generation/step on a lower strip."
        ),
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=Path("runs/batch"),
        help="Batch run root directory.",
    )
    parser.add_argument(
        "--sub-dir",
        type=str,
        default="init",
        help="Batch run sub directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/batch_analysis/task_model_plots"),
        help="Directory to store generated plots.",
    )
    parser.add_argument(
        "--overview-dir",
        type=Path,
        default=Path("runs/batch_analysis/task_model_plot_pages"),
        help="Directory to store 9-up overview SVG pages.",
    )
    parser.add_argument(
        "--mode",
        choices=("cumulative-best", "candidate-score", "iteration-max"),
        default="cumulative-best",
        help=(
            "Line style on the chronological timeline. "
            "`iteration-max` is kept as a backward-compatible alias of `candidate-score`."
        ),
    )
    parser.add_argument(
        "--timeline",
        choices=("discovery-order", "elapsed-minutes"),
        default="discovery-order",
        help=(
            "Chronological x-axis. `discovery-order` uses the rank of each discovered program "
            "after sorting by meta.json timestamp. `elapsed-minutes` uses minutes since the first "
            "program in that run."
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task name filter. Can be repeated.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model name filter. Can be repeated.",
    )
    parser.add_argument(
        "--algorithm",
        action="append",
        default=[],
        help="Algorithm name filter. Can be repeated.",
    )
    parser.add_argument(
        "--compare-by",
        choices=("auto", "model", "algorithm"),
        default="auto",
        help=(
            "Which series dimension to compare inside each task plot. "
            "`auto` uses models when only one algorithm is present, otherwise algorithms."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only plot the first N tasks after filtering. 0 means no limit.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=9,
        help="Number of task charts per overview page.",
    )
    return parser.parse_args()


def _matches_filters(
    task_name: str,
    model_name: str,
    algorithm_name: str,
    task_filters: list[str],
    model_filters: list[str],
    algorithm_filters: list[str],
) -> bool:
    if task_filters and task_name not in task_filters:
        return False
    if model_filters and model_name not in model_filters:
        return False
    if algorithm_filters and algorithm_name not in algorithm_filters:
        return False
    return True


def _normalize_mode(mode: str) -> str:
    if mode == "iteration-max":
        return "candidate-score"
    return mode


def _resolve_compare_by(compare_by: str, runs) -> str:
    if compare_by != "auto":
        return compare_by
    algorithms = {run.algorithm_name for run in runs}
    if len(algorithms) <= 1:
        return "model"
    return "algorithm"


def _facet_by(compare_by: str, runs) -> str | None:
    if compare_by == "model":
        algorithms = {run.algorithm_name for run in runs}
        return "algorithm" if len(algorithms) > 1 else None
    models = {run.model_name for run in runs}
    return "model" if len(models) > 1 else None


def _series_value(run, compare_by: str) -> str:
    return run.model_name if compare_by == "model" else run.algorithm_name


def _series_label(run, compare_by: str) -> str:
    return _series_value(run, compare_by)


def _series_color(run, compare_by: str) -> str:
    if compare_by == "model":
        return MODEL_COLORS.get(run.model_name, "#444444")
    return ALGORITHM_COLORS.get(run.algorithm_name, "#444444")


def _series_sort(run, compare_by: str) -> tuple[int, str]:
    if compare_by == "model":
        return model_sort_key(run.model_name)
    return algorithm_sort_key(run.algorithm_name)


def _series_heading(compare_by: str) -> str:
    return "Models" if compare_by == "model" else "Algorithms"


def _facet_value(run, facet_by: str | None) -> str | None:
    if facet_by == "model":
        return run.model_name
    if facet_by == "algorithm":
        return run.algorithm_name
    return None


def _panel_title(task_name: str, facet_value: str | None) -> str:
    if facet_value:
        return f"{task_name} [{facet_value}]"
    return task_name


def _facet_sort_key(facet_value: str | None, facet_by: str | None) -> tuple[int, str]:
    if facet_value is None:
        return (-1, "")
    if facet_by == "model":
        return model_sort_key(facet_value)
    if facet_by == "algorithm":
        return algorithm_sort_key(facet_value)
    return (0, facet_value)


def _nice_ticks(vmin: float, vmax: float, count: int = 6) -> list[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return [0.0, 1.0]
    if vmin == vmax:
        return [vmin]
    span = vmax - vmin
    raw_step = span / max(count - 1, 1)
    power = 10 ** math.floor(math.log10(abs(raw_step))) if raw_step != 0 else 1.0
    scaled = raw_step / power
    if scaled <= 1:
        nice = 1
    elif scaled <= 2:
        nice = 2
    elif scaled <= 5:
        nice = 5
    else:
        nice = 10
    step = nice * power
    start = math.ceil(vmin / step) * step
    end = math.floor(vmax / step) * step
    if start > end:
        return [vmin, vmax]
    ticks: list[float] = []
    value = start
    for _ in range(1000):
        if value > end + step * 1e-9:
            break
        ticks.append(value)
        value += step
    if not ticks:
        return [vmin, vmax]
    if vmin < ticks[0] - step * 0.5:
        ticks.insert(0, vmin)
    if vmax > ticks[-1] + step * 0.5:
        ticks.append(vmax)
    return ticks


def _integer_ticks(vmin: int, vmax: int, count: int = 6) -> list[int]:
    if vmax <= vmin:
        return [vmin]
    span = vmax - vmin
    step = max(1, math.ceil(span / max(count - 1, 1)))
    ticks = list(range(vmin, vmax + 1, step))
    if ticks[-1] != vmax:
        ticks.append(vmax)
    return ticks


def _format_tick(value: float, *, integer: bool = False) -> str:
    if integer or abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return format_number(value, digits=6)


def _timeline_axis(entries, timeline: str) -> tuple[list[float], str, bool, str]:
    if timeline == "elapsed-minutes" and entries and all(entry.timestamp is not None for entry in entries):
        start = min(entry.timestamp for entry in entries if entry.timestamp is not None)
        return (
            [max(0.0, (entry.timestamp - start) / 60.0) for entry in entries if entry.timestamp is not None],
            "Elapsed Minutes",
            False,
            "chronological by meta.json timestamp",
        )
    return (
        [float(idx) for idx, _entry in enumerate(entries)],
        "Discovery Order",
        True,
        "chronological by meta.json timestamp",
    )


def _build_series_item(
    run,
    *,
    mode: str,
    timeline: str,
    series_label: str,
    series_color: str,
) -> dict[str, object] | None:
    entries = load_run_entries(run)
    scored_entries = [entry for entry in chronological_entries(entries) if entry.score is not None]
    if not scored_entries:
        return None

    x_values, x_label, x_is_integer, timeline_desc = _timeline_axis(scored_entries, timeline)
    best_entry = select_best_entry(entries)
    root_entry = select_root_entry(entries)
    iter0_entries = [entry for entry in entries if entry.iteration == 0]
    iter0_best = select_best_entry(iter0_entries)

    root_score = root_entry.score if root_entry is not None else None
    best_score = best_entry.score if best_entry is not None else None
    gain_pct = None
    if root_score not in (None, 0.0) and best_score is not None:
        gain_pct = (best_score - root_score) / root_score * 100.0

    best_so_far: float | None = None
    scatter_points: list[dict[str, object]] = []
    line_points: list[tuple[float, float]] = []
    final_best_program_id = best_entry.program_id if best_entry is not None else ""
    root_program_id = root_entry.program_id if root_entry is not None else ""
    for order, (entry, x_value) in enumerate(zip(scored_entries, x_values, strict=False)):
        score = float(entry.score)
        is_improvement = best_so_far is None or score > best_so_far
        if is_improvement:
            best_so_far = score
        line_score = best_so_far if mode == "cumulative-best" else score
        hover = (
            f"{run.algorithm_name}/{run.model_name}"
            f" | order={order}"
            f" | score={format_number(score)}"
            f" | iter={entry.iteration}"
            f" | gen={entry.generation if entry.generation is not None else '?'}"
        )
        if entry.timestamp is not None:
            hover += f" | ts={format_number(entry.timestamp, digits=10)}"
        scatter_points.append(
            {
                "x": x_value,
                "score": score,
                "iteration": entry.iteration,
                "generation": max(int(entry.generation or 0), 0),
                "program_id": entry.program_id,
                "is_iter0": entry.iteration == 0,
                "is_root": entry.program_id == root_program_id,
                "is_improvement": is_improvement,
                "is_global_best": entry.program_id == final_best_program_id,
                "hover": hover,
            }
        )
        line_points.append((x_value, line_score))

    best_marker = ""
    if best_entry is not None:
        best_marker = f"best@i{best_entry.iteration}/g{best_entry.generation if best_entry.generation is not None else '?'}"

    summary = (
        f"root={format_number(root_score)} | "
        f"iter0-best={format_number(iter0_best.score if iter0_best is not None else None)} | "
        f"best={format_number(best_score)} | "
        f"Δroot={format_pct(gain_pct)} | "
        f"{best_marker}"
    )

    return {
        "series_key": series_label,
        "color": series_color,
        "entries": entries,
        "scatter_points": scatter_points,
        "line_points": line_points,
        "x_label": x_label,
        "x_is_integer": x_is_integer,
        "timeline_desc": timeline_desc,
        "label": series_label,
        "summary": summary,
        "root_score": root_score,
        "best_score": best_score,
        "bootstrap_count": len(iter0_entries),
    }


def _panel_svg_parts(
    task_name: str,
    series: list[dict[str, object]],
    *,
    mode: str,
    timeline: str,
    width: int,
    height: int,
    origin_x: int = 0,
    origin_y: int = 0,
    title_font_size: int = 24,
    subtitle_font_size: int = 14,
    label_font_size: int = 14,
    tick_font_size: int = 12,
    include_subtitle: bool = True,
    include_local_legend: bool = True,
    series_heading: str = "Models",
    margin_left: int = 90,
    margin_right: int = 430,
    margin_top: int = 76,
    margin_bottom: int = 86,
) -> list[str]:
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    score_gap = 28
    generation_h = max(52, int(plot_h * 0.2))
    score_h = plot_h - score_gap - generation_h
    if score_h < 80:
        score_h = max(80, plot_h - score_gap - 52)
        generation_h = max(44, plot_h - score_gap - score_h)

    score_top = origin_y + margin_top
    score_bottom = score_top + score_h
    generation_top = score_bottom + score_gap
    generation_bottom = generation_top + generation_h

    all_scatter_points = [point for item in series for point in item["scatter_points"]]  # type: ignore[index]
    all_line_points = [point for item in series for point in item["line_points"]]  # type: ignore[index]
    all_x_values = [float(point["x"]) for point in all_scatter_points]  # type: ignore[index]
    if not all_x_values:
        all_x_values = [point[0] for point in all_line_points]
    x_min = min(all_x_values) if all_x_values else 0.0
    x_max = max(all_x_values) if all_x_values else 1.0
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    all_scores = [float(point["score"]) for point in all_scatter_points]  # type: ignore[index]
    y_min = min(all_scores) if all_scores else 0.0
    y_max = max(all_scores) if all_scores else 1.0
    if y_min == y_max:
        pad = max(abs(y_min) * 0.05, 1.0)
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    max_generation = max((int(point["generation"]) for point in all_scatter_points), default=0)  # type: ignore[index]
    generation_ticks = _integer_ticks(0, max_generation, count=5)
    generation_min = -0.5
    generation_max = max(max_generation + 0.5, 0.5)

    def sx(value: float) -> float:
        return origin_x + margin_left + (value - x_min) / (x_max - x_min) * plot_w

    def sy_score(value: float) -> float:
        return score_top + score_h - (value - y_min) / (y_max - y_min) * score_h

    def sy_generation(value: float) -> float:
        return generation_top + generation_h - (value - generation_min) / (generation_max - generation_min) * generation_h

    x_is_integer = bool(series[0]["x_is_integer"]) if series else True
    x_label = str(series[0]["x_label"]) if series else "Discovery Order"
    x_ticks = (
        [float(value) for value in _integer_ticks(int(math.floor(x_min)), int(math.ceil(x_max)), count=7)]
        if x_is_integer
        else _nice_ticks(x_min, x_max, count=7)
    )
    y_ticks = _nice_ticks(y_min, y_max, count=6)

    mode_text = "best-so-far line" if mode == "cumulative-best" else "candidate-score line"
    subtitle = f"x = {x_label.lower()} ({series[0]['timeline_desc']}) | upper = score | lower = generation/step | hollow dot = iter 0 | {mode_text}" if series else ""

    parts: list[str] = []
    parts.append(f'<rect x="{origin_x}" y="{origin_y}" width="{width}" height="{height}" fill="white"/>')
    parts.append(
        f'<text x="{origin_x + margin_left}" y="{origin_y + 36}" font-size="{title_font_size}" font-family="Arial, sans-serif" font-weight="bold">{html.escape(task_name)}</text>'
    )
    if include_subtitle:
        parts.append(
            f'<text x="{origin_x + margin_left}" y="{origin_y + 60}" font-size="{subtitle_font_size}" font-family="Arial, sans-serif" fill="#555">{html.escape(subtitle)}</text>'
        )

    for tick in x_ticks:
        x = sx(float(tick))
        parts.append(
            f'<line x1="{x:.2f}" y1="{score_top:.2f}" x2="{x:.2f}" y2="{generation_bottom:.2f}" stroke="#f0f0f0" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{generation_bottom + 24:.2f}" text-anchor="middle" font-size="{tick_font_size}" font-family="Arial, sans-serif" fill="#444">{html.escape(_format_tick(float(tick), integer=x_is_integer))}</text>'
        )

    for tick in y_ticks:
        y = sy_score(float(tick))
        parts.append(
            f'<line x1="{origin_x + margin_left}" y1="{y:.2f}" x2="{origin_x + margin_left + plot_w}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{origin_x + margin_left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="{tick_font_size}" font-family="Arial, sans-serif" fill="#444">{html.escape(format_number(float(tick), digits=6))}</text>'
        )

    for tick in generation_ticks:
        y = sy_generation(float(tick))
        parts.append(
            f'<line x1="{origin_x + margin_left}" y1="{y:.2f}" x2="{origin_x + margin_left + plot_w}" y2="{y:.2f}" stroke="#f6f6f6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{origin_x + margin_left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="{tick_font_size}" font-family="Arial, sans-serif" fill="#444">{tick}</text>'
        )

    parts.append(
        f'<rect x="{origin_x + margin_left}" y="{score_top}" width="{plot_w}" height="{score_h}" fill="none" stroke="#222" stroke-width="1.5"/>'
    )
    parts.append(
        f'<rect x="{origin_x + margin_left}" y="{generation_top}" width="{plot_w}" height="{generation_h}" fill="none" stroke="#222" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{origin_x + margin_left + plot_w / 2:.2f}" y="{origin_y + height - 18}" text-anchor="middle" font-size="{label_font_size}" font-family="Arial, sans-serif">{html.escape(x_label)}</text>'
    )
    parts.append(
        f'<text x="{origin_x + 24}" y="{score_top + score_h / 2:.2f}" transform="rotate(-90 {origin_x + 24} {score_top + score_h / 2:.2f})" text-anchor="middle" font-size="{label_font_size}" font-family="Arial, sans-serif">Combined Score</text>'
    )
    parts.append(
        f'<text x="{origin_x + 24}" y="{generation_top + generation_h / 2:.2f}" transform="rotate(-90 {origin_x + 24} {generation_top + generation_h / 2:.2f})" text-anchor="middle" font-size="{label_font_size}" font-family="Arial, sans-serif">Generation / Step</text>'
    )

    for item in series:
        color = str(item["color"])
        line_points = item["line_points"]  # type: ignore[index]
        scatter_points = item["scatter_points"]  # type: ignore[index]
        if line_points:
            polyline = " ".join(f"{sx(float(x)):.2f},{sy_score(float(y)):.2f}" for x, y in line_points)
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}"/>'
            )
        best_label_drawn = False
        for point in scatter_points:
            x = sx(float(point["x"]))  # type: ignore[index]
            y_score = sy_score(float(point["score"]))  # type: ignore[index]
            y_generation = sy_generation(float(point["generation"]))  # type: ignore[index]
            fill = "white" if point["is_iter0"] else color  # type: ignore[index]
            stroke = color
            radius = 4.2 if point["is_improvement"] else 3.0  # type: ignore[index]
            if point["is_global_best"]:  # type: ignore[index]
                parts.append(
                    f'<circle cx="{x:.2f}" cy="{y_score:.2f}" r="{radius + 2.2:.2f}" fill="none" stroke="{color}" stroke-width="1.2" opacity="0.9"/>'
                )
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y_score:.2f}" r="{radius:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"><title>{html.escape(str(point["hover"]))}</title></circle>'
            )
            parts.append(
                f'<circle cx="{x:.2f}" cy="{y_generation:.2f}" r="2.7" fill="{fill}" stroke="{stroke}" stroke-width="1.0"><title>{html.escape(str(point["hover"]))}</title></circle>'
            )
            if point["is_root"]:  # type: ignore[index]
                parts.append(f'<circle cx="{x:.2f}" cy="{y_score:.2f}" r="1.4" fill="#111"/>')
                parts.append(f'<circle cx="{x:.2f}" cy="{y_generation:.2f}" r="1.2" fill="#111"/>')
            if include_local_legend and point["is_global_best"] and not best_label_drawn:  # type: ignore[index]
                label = f"i{point['iteration']} g{point['generation']}"
                label_x = min(x + 8.0, origin_x + margin_left + plot_w - 6.0)
                anchor = "start"
                if label_x > origin_x + margin_left + plot_w - 60:
                    label_x = x - 8.0
                    anchor = "end"
                parts.append(
                    f'<text x="{label_x:.2f}" y="{y_score - 8:.2f}" text-anchor="{anchor}" font-size="11" font-family="Arial, sans-serif" fill="{color}">{html.escape(label)}</text>'
                )
                best_label_drawn = True

    if include_local_legend:
        legend_x = origin_x + margin_left + plot_w + 26
        legend_y = origin_y + margin_top + 8
        row_h = 72
        encoding_lines = [
            f"x = {x_label.lower()}",
            "upper = score, lower = generation/step",
            "hollow dot = iteration 0",
            "black center = root baseline",
            "outer ring = final best",
        ]
        legend_h = 34 + row_h * len(series) + 30 + 20 * (len(encoding_lines) + 1)
        legend_w = margin_right - 36
        parts.append(
            f'<rect x="{legend_x - 16}" y="{legend_y - 22}" width="{legend_w}" height="{legend_h}" fill="#fafafa" stroke="#dddddd" rx="8"/>'
        )
        parts.append(
            f'<text x="{legend_x}" y="{legend_y}" font-size="16" font-family="Arial, sans-serif" font-weight="bold">{html.escape(series_heading)}</text>'
        )
        for idx, item in enumerate(series):
            y = legend_y + 26 + idx * row_h
            color = str(item["color"])
            parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
            parts.append(f'<circle cx="{legend_x + 14}" cy="{y}" r="4" fill="{color}" stroke="white" stroke-width="1"/>')
            parts.append(
                f'<text x="{legend_x + 40}" y="{y + 4}" font-size="14" font-family="Arial, sans-serif" font-weight="bold">{html.escape(str(item["label"]))}</text>'
            )
            parts.append(
                f'<text x="{legend_x + 40}" y="{y + 22}" font-size="11" font-family="Arial, sans-serif" fill="#555">{html.escape(str(item["summary"]))}</text>'
            )
            parts.append(
                f'<text x="{legend_x + 40}" y="{y + 38}" font-size="11" font-family="Arial, sans-serif" fill="#777">iter0 points={int(item["bootstrap_count"])} | root={format_number(item["root_score"] if isinstance(item["root_score"], float | int) else None)} | best={format_number(item["best_score"] if isinstance(item["best_score"], float | int) else None)}</text>'
            )

        encoding_y = legend_y + 26 + row_h * len(series) + 12
        parts.append(
            f'<text x="{legend_x}" y="{encoding_y}" font-size="14" font-family="Arial, sans-serif" font-weight="bold">Encoding</text>'
        )
        for idx, line in enumerate(encoding_lines, start=1):
            parts.append(
                f'<text x="{legend_x}" y="{encoding_y + idx * 20}" font-size="11" font-family="Arial, sans-serif" fill="#555">{html.escape(line)}</text>'
            )

    return parts


def _build_svg(
    task_name: str,
    series: list[dict[str, object]],
    *,
    mode: str,
    timeline: str,
    series_heading: str,
) -> str:
    width = 1460
    height = 860
    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.extend(
        _panel_svg_parts(
            task_name,
            series,
            mode=mode,
            timeline=timeline,
            width=width,
            height=height,
            series_heading=series_heading,
        )
    )
    parts.append("</svg>")
    return "\n".join(parts)


def _build_overview_svg(
    page_title: str,
    panels: list[dict[str, Any]],
    *,
    mode: str,
    timeline: str,
    page_size: int,
    series_heading: str,
    legend_items: list[tuple[str, str]],
) -> str:
    page_width = 1800
    page_height = 2400
    page_margin_x = 36
    page_margin_top = 110
    page_margin_bottom = 36
    gap_x = 24
    gap_y = 24
    cols = 3
    rows = max(1, math.ceil(len(panels) / cols))
    header_h = 102
    panel_width = int((page_width - 2 * page_margin_x - gap_x * (cols - 1)) / cols)
    panel_height = int((page_height - page_margin_top - page_margin_bottom - header_h - gap_y * (rows - 1)) / rows)

    mode_text = "best-so-far line" if mode == "cumulative-best" else "candidate-score line"

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{page_width}" height="{page_height}" viewBox="0 0 {page_width} {page_height}">')
    parts.append(f'<rect x="0" y="0" width="{page_width}" height="{page_height}" fill="white"/>')
    parts.append(
        f'<text x="{page_margin_x}" y="34" font-size="28" font-family="Arial, sans-serif" font-weight="bold">{html.escape(page_title)}</text>'
    )
    parts.append(
        f'<text x="{page_margin_x}" y="60" font-size="14" font-family="Arial, sans-serif" fill="#555">Alphabetical order | up to {page_size} tasks per page | x = {html.escape(timeline)} | {html.escape(mode_text)}</text>'
    )
    parts.append(
        f'<text x="{page_margin_x}" y="82" font-size="13" font-family="Arial, sans-serif" fill="#666">Each panel: upper = score, lower = generation/step, hollow dots = iter 0, black center = root baseline, outer ring = final best</text>'
    )

    legend_x = page_width - 760
    legend_y = 34
    parts.append(
        f'<text x="{legend_x}" y="{legend_y}" font-size="15" font-family="Arial, sans-serif" font-weight="bold">{html.escape(series_heading)} Legend</text>'
    )
    for idx, (label, color) in enumerate(legend_items):
        x = legend_x + (idx % 2) * 250
        y = legend_y + 24 + (idx // 2) * 24
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 26}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<circle cx="{x + 13}" cy="{y}" r="4" fill="{color}" stroke="white" stroke-width="1"/>')
        parts.append(
            f'<text x="{x + 36}" y="{y + 4}" font-size="12" font-family="Arial, sans-serif">{html.escape(label)}</text>'
        )

    for idx, panel in enumerate(panels):
        row = idx // cols
        col = idx % cols
        origin_x = page_margin_x + col * (panel_width + gap_x)
        origin_y = page_margin_top + header_h + row * (panel_height + gap_y)
        parts.extend(
            _panel_svg_parts(
                panel["task_name"],
                panel["series"],
                mode=mode,
                timeline=timeline,
                width=panel_width,
                height=panel_height,
                origin_x=origin_x,
                origin_y=origin_y,
                title_font_size=16,
                subtitle_font_size=10,
                label_font_size=11,
                tick_font_size=9,
                include_subtitle=False,
                include_local_legend=False,
                series_heading=series_heading,
                margin_left=52,
                margin_right=20,
                margin_top=34,
                margin_bottom=44,
            )
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _write_overview_pages(
    overview_panels: list[dict[str, Any]],
    overview_dir: Path,
    *,
    mode: str,
    timeline: str,
    page_size: int,
    page_prefix: str,
    series_heading: str,
    legend_items: list[tuple[str, str]],
) -> int:
    if not overview_panels:
        return 0
    overview_dir.mkdir(parents=True, exist_ok=True)
    pages = _chunked(overview_panels, page_size)
    for page_index, panels in enumerate(pages, start=1):
        first_task = panels[0]["task_name"]
        last_task = panels[-1]["task_name"]
        page_title = f"{page_prefix} Page {page_index}: {first_task} ... {last_task}"
        svg = _build_overview_svg(
            page_title,
            panels,
            mode=mode,
            timeline=timeline,
            page_size=page_size,
            series_heading=series_heading,
            legend_items=legend_items,
        )
        out_path = overview_dir / f"{page_index:03d}__{safe_filename(first_task)}__to__{safe_filename(last_task)}.svg"
        out_path.write_text(svg, encoding="utf-8")
        print(f"[page] {out_path} ({len(panels)} tasks)")
    return len(pages)


def _plot_task(
    task_name: str,
    output_name: str,
    runs,
    output_dir: Path,
    *,
    mode: str,
    timeline: str,
    compare_by: str,
) -> tuple[Path, list[dict[str, object]]]:
    series: list[dict[str, object]] = []
    for run in sorted(runs, key=lambda item: _series_sort(item, compare_by)):
        item = _build_series_item(
            run,
            mode=mode,
            timeline=timeline,
            series_label=_series_label(run, compare_by),
            series_color=_series_color(run, compare_by),
        )
        if item is None:
            continue
        series.append(item)

    if not series:
        raise RuntimeError(f"No plottable trajectories found for task {task_name}")

    svg = _build_svg(task_name, series, mode=mode, timeline=timeline, series_heading=_series_heading(compare_by))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{safe_filename(output_name)}.svg"
    out_path.write_text(svg, encoding="utf-8")
    return out_path, series


def main() -> int:
    args = _parse_args()
    mode = _normalize_mode(args.mode)
    runs = dedupe_latest_runs(scan_batch_task_runs(args.batch_dir / args.sub_dir))
    compare_by = _resolve_compare_by(args.compare_by, runs)
    facet_by = _facet_by(compare_by, runs)

    grouped: dict[tuple[str | None, str], list] = {}
    for run in runs:
        if not _matches_filters(run.task_name, run.model_name, run.algorithm_name, args.task, args.model, args.algorithm):
            continue
        grouped.setdefault((_facet_value(run, facet_by), run.task_name), []).append(run)

    panel_keys = sorted(grouped, key=lambda item: (_facet_sort_key(item[0], facet_by), item[1]))
    if args.limit > 0:
        panel_keys = panel_keys[: args.limit]

    generated = 0
    skipped = 0
    overview_panels: dict[str | None, list[dict[str, Any]]] = {}
    for facet_value, task_name in panel_keys:
        panel_runs = grouped[(facet_value, task_name)]
        panel_title = _panel_title(task_name, facet_value)
        output_dir = args.output_dir / args.sub_dir
        if facet_value is not None:
            output_dir = output_dir / safe_filename(facet_value)
        try:
            out_path, series = _plot_task(
                panel_title,
                task_name,
                panel_runs,
                output_dir,
                mode=mode,
                timeline=args.timeline,
                compare_by=compare_by,
            )
        except RuntimeError as exc:
            print(f"[skip] {panel_title}: {exc}")
            skipped += 1
            continue

        series_labels = ", ".join(
            _series_value(run, compare_by) for run in sorted(panel_runs, key=lambda item: _series_sort(item, compare_by))
        )
        print(f"[plot] {panel_title} -> {out_path} ({series_labels})")
        overview_panels.setdefault(facet_value, []).append({"task_name": panel_title, "series": series})
        generated += 1

    if generated == 0:
        print("No tasks matched the requested filters.")
        return 1

    page_count = 0
    for facet_value in sorted(overview_panels, key=lambda item: _facet_sort_key(item, facet_by)):
        facet_panels = overview_panels[facet_value]
        facet_overview_dir = args.overview_dir / args.sub_dir
        if facet_value is not None:
            facet_overview_dir = facet_overview_dir / safe_filename(facet_value)
        legend_map: dict[str, str] = {}
        for panel in facet_panels:
            for series in panel["series"]:
                label = str(series["label"])
                legend_map.setdefault(label, str(series["color"]))
        legend_items = sorted(
            legend_map.items(),
            key=lambda item: model_sort_key(item[0]) if compare_by == "model" else algorithm_sort_key(item[0]),
        )
        page_prefix = f"Task {_series_heading(compare_by)} Comparison"
        if facet_value is not None:
            page_prefix += f" [{facet_value}]"
        page_count += _write_overview_pages(
            facet_panels,
            facet_overview_dir,
            mode=mode,
            timeline=args.timeline,
            page_size=args.page_size,
            page_prefix=page_prefix,
            series_heading=_series_heading(compare_by),
            legend_items=legend_items,
        )

    print(
        f"Generated {generated} plot(s) in {args.output_dir / args.sub_dir}; "
        f"generated {page_count} overview page(s) in {args.overview_dir / args.sub_dir}; "
        f"skipped {skipped} task(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
