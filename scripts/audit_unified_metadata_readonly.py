#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_PATTERNS = (
    "verification",
    "frontier_eval/run_eval.sh",
)
RECOMMENDED_PATTERNS = (
    "verification/evaluate.py",
    "verification/evaluator.py",
    "frontier_eval/parse_mdriver_result.py",
    "frontier_eval/evaluate_submission.py",
)


def _read_nonempty_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _contains_pattern(entries: list[str], pattern: str) -> bool:
    for entry in entries:
        if entry == ".":
            return True
        if pattern in entry:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit unified task frontier_eval metadata readonly coverage."
    )
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="Repository root path.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat missing recommended coverage as failure.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    benchmarks_root = repo_root / "benchmarks"
    if not benchmarks_root.is_dir():
        print(f"[error] benchmarks dir not found: {benchmarks_root}", file=sys.stderr)
        return 2

    task_meta_dirs = sorted(benchmarks_root.glob("**/frontier_eval"))
    weak: list[tuple[str, list[str], list[str]]] = []
    missing_meta: list[str] = []

    for meta_dir in task_meta_dirs:
        benchmark_dir = meta_dir.parent
        rel_task = benchmark_dir.relative_to(benchmarks_root).as_posix()
        eval_command_file = meta_dir / "eval_command.txt"
        if not eval_command_file.is_file():
            continue
        readonly_file = meta_dir / "readonly_files.txt"
        entries = _read_nonempty_lines(readonly_file)
        if not entries:
            missing_meta.append(rel_task)
            continue

        missing_required = [p for p in REQUIRED_PATTERNS if not _contains_pattern(entries, p)]
        missing_recommended = [p for p in RECOMMENDED_PATTERNS if not _contains_pattern(entries, p)]
        if missing_required or (args.strict and missing_recommended):
            weak.append((rel_task, missing_required, missing_recommended))

    if missing_meta:
        print("[warn] missing or empty readonly_files.txt:")
        for task in missing_meta:
            print(f"  - {task}")

    if weak:
        print("[warn] readonly coverage issues:")
        for task, req, rec in weak:
            print(f"  - {task}")
            if req:
                print(f"      missing required: {', '.join(req)}")
            if rec:
                print(f"      missing recommended: {', '.join(rec)}")

    if missing_meta or weak:
        return 1

    print("[ok] readonly coverage checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
