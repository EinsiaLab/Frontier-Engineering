#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Re-evaluate stored baseline/best snapshots and overwrite metrics in-place."
    )
    p.add_argument("--batch-root", type=Path, required=True)
    p.add_argument("--algorithm", type=str, required=True, choices=["shinkaevolve", "openevolve", "abmcts"])
    p.add_argument("--mode", type=str, default="baseline", choices=["baseline", "best"])
    p.add_argument("--llm", type=str, default="openai__qwen3_coder_next")
    p.add_argument("--tasks", type=str, required=True, help="Comma-separated task directory names.")
    p.add_argument("--no-backup", action="store_true")
    return p.parse_args()


def _iter0_dir(history_dir: Path) -> Path:
    matches = sorted(history_dir.glob("iter_000000__*"))
    if not matches:
        raise FileNotFoundError(f"Missing iter_000000__* under {history_dir}")
    return matches[0]


def _find_program_file(*dirs: Path) -> Path | None:
    names = (
        "best_program.py",
        "best_program.cpp",
        "best_program.c",
        "best_program.rs",
        "best_program.java",
        "best_program.sh",
        "main.py",
        "main.cpp",
        "main.c",
        "main.rs",
        "main.java",
        "main.sh",
        "program.py",
        "program.cpp",
        "program.c",
        "program.rs",
        "program.java",
        "program.sh",
    )
    for d in dirs:
        if not d.is_dir():
            continue
        for name in names:
            p = d / name
            if p.is_file():
                return p
    return None


def _resolve_paths(batch_root: Path, task_dir_name: str, algorithm: str, llm: str, mode: str) -> tuple[Path, Path]:
    run_root = batch_root / task_dir_name / algorithm / llm
    if algorithm == "shinkaevolve" and mode == "baseline":
        algo_root = run_root / "shinkaevolve"
        program_path = algo_root / "gen_0" / "main.py"
        results_dir = algo_root / "gen_0" / "results"
        return program_path, results_dir
    if algorithm == "shinkaevolve" and mode == "best":
        algo_root = run_root / "shinkaevolve"
        best_dir = algo_root / "best"
        program_path = _find_program_file(best_dir)
        results_dir = best_dir / "results"
        info_path = best_dir / "best_program_info.json"
        if info_path.is_file():
            import json

            try:
                info = json.loads(info_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                info = {}
            if isinstance(info, dict):
                raw_program = str(info.get("program_path") or "").strip()
                raw_results = str(info.get("results_dir") or "").strip()
                if raw_program:
                    candidate = Path(raw_program)
                    if candidate.is_file():
                        program_path = candidate
                if raw_results:
                    candidate = Path(raw_results)
                    if candidate.is_dir():
                        results_dir = candidate
        if program_path is None:
            raise FileNotFoundError(f"Missing best program under {best_dir}")
        return program_path, results_dir
    if algorithm == "openevolve" and mode == "baseline":
        algo_root = run_root / "openevolve"
        iter0 = _iter0_dir(algo_root / "history")
        program_path = iter0 / "program.py"
        results_dir = iter0
        return program_path, results_dir
    if algorithm == "openevolve" and mode == "best":
        import json

        algo_root = run_root / "openevolve"
        best_dir = algo_root / "best"
        info_path = best_dir / "best_program_info.json"
        program_path = _find_program_file(best_dir)
        results_dir = best_dir
        if info_path.is_file():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                info = {}
            if isinstance(info, dict):
                raw_program = str(info.get("program_path") or "").strip()
                raw_results = str(info.get("results_dir") or "").strip()
                if raw_program:
                    candidate = Path(raw_program)
                    if candidate.is_file():
                        program_path = candidate
                if raw_results:
                    candidate = Path(raw_results)
                    if candidate.is_dir():
                        results_dir = candidate
                if raw_results == "":
                    iteration = info.get("iteration")
                    program_id = str(info.get("id") or "").strip()
                    if isinstance(iteration, int):
                        direct = algo_root / "history" / f"iter_{iteration:06d}__{program_id}"
                        if direct.is_dir():
                            results_dir = direct
                            if program_path is None:
                                candidate = direct / "program.py"
                                if candidate.is_file():
                                    program_path = candidate
        if program_path is None:
            raise FileNotFoundError(f"Missing best program under {best_dir}")
        return program_path, results_dir
    if algorithm == "abmcts" and mode == "baseline":
        algo_root = run_root / "abmcts"
        program_path = algo_root / "baseline" / "program.py"
        results_dir = algo_root / "baseline"
        return program_path, results_dir
    if algorithm == "abmcts" and mode == "best":
        algo_root = run_root / "abmcts"
        best_dir = algo_root / "best"
        program_path = _find_program_file(best_dir)
        if program_path is None:
            raise FileNotFoundError(f"Missing best program under {best_dir}")
        return program_path, best_dir
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _backup_existing(results_dir: Path) -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = results_dir / f"_baseline_repair_backup_{stamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for name in (
        "metrics.json",
        "correct.json",
        "artifacts.json",
        "text_feedback.txt",
        "stdout_bridge.txt",
        "stderr_bridge.txt",
        "context_manifest.json",
        "job_log.out",
        "job_log.err",
    ):
        src = results_dir / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)


def main() -> int:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    os.environ.setdefault("FRONTIER_ENGINEERING_ROOT", str(repo_root))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from frontier_eval.algorithms.shinkaevolve.shinkaevolve_entrypoint import main as eval_main

    batch_root = args.batch_root.expanduser().resolve()
    task_dir_names = [x.strip() for x in args.tasks.split(",") if x.strip()]
    failures: list[str] = []

    for task_dir_name in task_dir_names:
        task_name = task_dir_name.replace("/", "_")
        program_path, results_dir = _resolve_paths(
            batch_root, task_dir_name, args.algorithm, args.llm, args.mode
        )

        if not program_path.is_file():
            failures.append(f"{task_dir_name}: missing program snapshot {program_path}")
            continue

        results_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_backup:
            _backup_existing(results_dir)

        print(f"[repair] {task_dir_name} -> {results_dir}")
        rc = eval_main(str(program_path), str(results_dir), task_name=task_name)
        if rc != 0:
            failures.append(f"{task_dir_name}: evaluator returned {rc}")

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
