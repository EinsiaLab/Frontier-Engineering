#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from batch_model_analysis_lib import (
    dedupe_latest_runs,
    model_sort_key,
    scan_batch_task_runs,
    summarize_run,
    write_csv,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one CSV row per task/model from runs/batch history/index.jsonl files.",
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=Path("runs/batch"),
        help="Batch run root directory.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("runs/batch_analysis/task_model_summary.csv"),
        help="Output CSV path.",
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
        "--all-runs",
        action="store_true",
        help="Export all matching runs instead of only the latest run per task/model.",
    )
    return parser.parse_args()


def _matches_filters(task_name: str, model_name: str, task_filters: list[str], model_filters: list[str]) -> bool:
    if task_filters and task_name not in task_filters:
        return False
    if model_filters and model_name not in model_filters:
        return False
    return True


def main() -> int:
    args = _parse_args()
    runs = scan_batch_task_runs(args.batch_dir)
    if not args.all_runs:
        runs = dedupe_latest_runs(runs)

    filtered = [
        run
        for run in runs
        if _matches_filters(run.task_name, run.model_name, args.task, args.model)
    ]
    filtered.sort(key=lambda item: (item.task_name, model_sort_key(item.model_name), item.batch_run_name))

    rows = [summarize_run(run) for run in filtered]
    if not rows:
        print("No runs matched the requested filters.")
        return 1

    write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} row(s) to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
