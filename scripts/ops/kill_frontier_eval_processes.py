#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcInfo:
    pid: int
    ppid: int
    pgid: int
    sid: int
    stat: str
    etime: str
    comm: str
    args: str


def _load_processes() -> list[ProcInfo]:
    cmd = ["ps", "-eo", "pid=,ppid=,pgid=,sid=,stat=,etime=,comm=,args=", "-ww"]
    out = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")
    procs: list[ProcInfo] = []
    for raw_line in out.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(None, 7)
        if len(parts) < 8:
            continue
        try:
            procs.append(
                ProcInfo(
                    pid=int(parts[0]),
                    ppid=int(parts[1]),
                    pgid=int(parts[2]),
                    sid=int(parts[3]),
                    stat=parts[4],
                    etime=parts[5],
                    comm=parts[6],
                    args=parts[7],
                )
            )
        except ValueError:
            continue
    return procs


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _build_match_tokens(
    *,
    patterns: list[str],
    matrices: list[str],
    batch_roots: list[str],
) -> list[str]:
    tokens: list[str] = []
    tokens.extend(patterns)

    for raw in matrices:
        path = Path(raw).expanduser()
        tokens.append(raw)
        tokens.append(path.name)
        tokens.append(path.stem)
        try:
            tokens.append(str(path.resolve()))
        except Exception:
            pass

    for raw in batch_roots:
        path = Path(raw).expanduser()
        tokens.append(raw)
        tokens.append(path.name)
        try:
            tokens.append(str(path.resolve()))
        except Exception:
            pass

    return _dedupe_keep_order(tokens)


def _looks_frontier_related(proc: ProcInfo) -> bool:
    args = proc.args
    return (
        "frontier_eval.batch" in args
        or "-m frontier_eval" in args
        or "run.output_dir=" in args
    )


def _collect_matches(
    *,
    processes: list[ProcInfo],
    tokens: list[str],
    include_all: bool,
) -> tuple[list[ProcInfo], dict[int, str]]:
    matches: list[ProcInfo] = []
    reasons: dict[int, str] = {}
    self_pid = os.getpid()

    for proc in processes:
        if proc.pid == self_pid:
            continue
        if not _looks_frontier_related(proc):
            continue

        if include_all:
            matches.append(proc)
            reasons[proc.pid] = "all-frontier-eval"
            continue

        matched_token = next((token for token in tokens if token in proc.args), None)
        if matched_token is None:
            continue
        matches.append(proc)
        reasons[proc.pid] = matched_token

    return matches, reasons


def _collect_descendants(
    *,
    processes: list[ProcInfo],
    roots: list[ProcInfo],
) -> tuple[dict[int, ProcInfo], dict[int, int]]:
    procs_by_pid = {proc.pid: proc for proc in processes}
    children_by_ppid: dict[int, list[int]] = defaultdict(list)
    for proc in processes:
        children_by_ppid[proc.ppid].append(proc.pid)

    selected: dict[int, ProcInfo] = {proc.pid: proc for proc in roots}
    depth: dict[int, int] = {proc.pid: 0 for proc in roots}
    queue = deque(proc.pid for proc in roots)

    while queue:
        pid = queue.popleft()
        parent_depth = depth[pid]
        for child_pid in children_by_ppid.get(pid, []):
            if child_pid in selected:
                continue
            child = procs_by_pid[child_pid]
            selected[child_pid] = child
            depth[child_pid] = parent_depth + 1
            queue.append(child_pid)

    return selected, depth


def _truncate(text: str, width: int = 140) -> str:
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def _print_processes(
    *,
    selected: dict[int, ProcInfo],
    roots: list[ProcInfo],
    reasons: dict[int, str],
    depth: dict[int, int],
) -> None:
    root_ids = {proc.pid for proc in roots}
    ordered = sorted(selected.values(), key=lambda proc: (depth.get(proc.pid, 0), proc.pid))
    print(f"Matched processes: {len(ordered)}")
    print(
        f"{'PID':>8} {'PPID':>8} {'PGID':>8} {'SID':>8} {'TYPE':>7} {'STAT':>6} {'ELAPSED':>10}  COMMAND"
    )
    for proc in ordered:
        proc_type = "root" if proc.pid in root_ids else f"child+{depth.get(proc.pid, 0)}"
        suffix = ""
        if proc.pid in reasons:
            suffix = f"  [match={reasons[proc.pid]}]"
        print(
            f"{proc.pid:>8} {proc.ppid:>8} {proc.pgid:>8} {proc.sid:>8} "
            f"{proc_type:>7} {proc.stat:>6} {proc.etime:>10}  {_truncate(proc.args)}{suffix}"
        )


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _kill_pid(pid: int, sig: signal.Signals) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return


def _kill_pgid(pgid: int, sig: signal.Signals) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return


def _terminate_selected(
    *,
    selected: dict[int, ProcInfo],
    roots: list[ProcInfo],
    depth: dict[int, int],
    force_after_s: float,
) -> None:
    root_ids = {proc.pid for proc in roots}
    leader_groups = {
        proc.pgid
        for proc in roots
        if proc.pid in root_ids and proc.pid == proc.pgid and proc.pid == proc.sid
    }

    covered_by_group = {
        proc.pid for proc in selected.values() if proc.pgid in leader_groups and proc.pid != proc.pgid
    }

    pid_targets = [
        proc.pid
        for proc in sorted(selected.values(), key=lambda proc: (depth.get(proc.pid, 0), proc.pid), reverse=True)
        if proc.pid not in covered_by_group
    ]

    for pgid in sorted(leader_groups):
        _kill_pgid(pgid, signal.SIGTERM)
    for pid in pid_targets:
        _kill_pid(pid, signal.SIGTERM)

    deadline = time.time() + force_after_s
    while time.time() < deadline:
        remaining = [pid for pid in selected if _pid_exists(pid)]
        if not remaining:
            return
        time.sleep(0.2)

    remaining = [selected[pid] for pid in selected if _pid_exists(pid)]
    remaining_groups = {
        proc.pgid for proc in remaining if proc.pgid in leader_groups and proc.pid == proc.pgid
    }
    remaining_pids = [
        proc.pid
        for proc in sorted(remaining, key=lambda proc: (depth.get(proc.pid, 0), proc.pid), reverse=True)
        if proc.pgid not in remaining_groups or proc.pid == proc.pgid
    ]

    for pgid in sorted(remaining_groups):
        _kill_pgid(pgid, signal.SIGKILL)
    for pid in remaining_pids:
        _kill_pid(pid, signal.SIGKILL)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List and kill Frontier Eval batch/frontier_eval related processes."
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Substring to match in process args. Repeatable.",
    )
    parser.add_argument(
        "--matrix",
        action="append",
        default=[],
        help="Matrix YAML path to match. Repeatable.",
    )
    parser.add_argument(
        "--batch-root",
        action="append",
        default=[],
        help="Batch root path or directory name to match. Repeatable.",
    )
    parser.add_argument(
        "--all-frontier-eval",
        action="store_true",
        help="Match all frontier_eval.batch / python -m frontier_eval processes.",
    )
    parser.add_argument(
        "--no-descendants",
        action="store_true",
        help="Do not include child processes of matched roots.",
    )
    parser.add_argument(
        "--force-after",
        type=float,
        default=5.0,
        help="Seconds to wait between SIGTERM and SIGKILL. Default: 5.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually kill matched processes. Without this flag the script only lists.",
    )

    args = parser.parse_args()

    tokens = _build_match_tokens(
        patterns=[str(x) for x in args.pattern],
        matrices=[str(x) for x in args.matrix],
        batch_roots=[str(x) for x in args.batch_root],
    )
    if not args.all_frontier_eval and not tokens:
        parser.error("provide at least one of --pattern/--matrix/--batch-root, or use --all-frontier-eval")

    processes = _load_processes()
    roots, reasons = _collect_matches(
        processes=processes,
        tokens=tokens,
        include_all=bool(args.all_frontier_eval),
    )
    if not roots:
        print("No matching Frontier Eval processes found.")
        return 0

    if args.no_descendants:
        selected = {proc.pid: proc for proc in roots}
        depth = {proc.pid: 0 for proc in roots}
    else:
        selected, depth = _collect_descendants(processes=processes, roots=roots)

    _print_processes(selected=selected, roots=roots, reasons=reasons, depth=depth)

    if not args.yes:
        print("\nDry-run only. Re-run with --yes to kill the matched processes.")
        return 0

    _terminate_selected(
        selected=selected,
        roots=roots,
        depth=depth,
        force_after_s=float(args.force_after),
    )

    time.sleep(0.5)
    remaining = [pid for pid in selected if _pid_exists(pid)]
    if remaining:
        print(f"\nSome processes are still alive: {remaining}", file=sys.stderr)
        return 1

    print("\nAll matched processes have been terminated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
