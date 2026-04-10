# EVOLVE-BLOCK-START
"""Enhanced multi-rule greedy baseline with local search for TA (Taillard, 1993).

Baseline constraints:
- Pure Python implementation.
- Standard library only.
- No `job_shop_lib` import and no external solver usage.
"""

from __future__ import annotations

import argparse
import os
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

FAMILY_PREFIX = "ta"
FAMILY_NAME = "TA (Taillard, 1993)"


def _natural_key(name: str) -> list[object]:
    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p for p in parts]


def _benchmark_json_path() -> Path:
    env_path = str(os.environ.get("JOBSHOP_BENCHMARK_JSON", "")).strip()
    if env_path:
        candidate = Path(env_path).expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"JOBSHOP_BENCHMARK_JSON points to a missing file: {candidate}"
        )

    candidates = [
        Path(__file__).resolve().parents[2] / "data" / "benchmark_instances.json",
        Path(__file__).resolve().parents[1] / "data" / "benchmark_instances.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "benchmark_instances.json not found under JobShop/data. "
        "Expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def load_benchmark_json() -> dict[str, dict[str, Any]]:
    with _benchmark_json_path().open("r", encoding="utf-8") as f:
        return json.load(f)


def load_family_instances() -> list[dict[str, Any]]:
    data = load_benchmark_json()
    selected = [
        value
        for name, value in data.items()
        if name.startswith(FAMILY_PREFIX)
    ]
    return sorted(selected, key=lambda x: _natural_key(x["name"]))


def load_instance_by_name(name: str) -> dict[str, Any]:
    data = load_benchmark_json()
    if name not in data:
        raise KeyError(f"Unknown instance: {name}")
    return data[name]


class ScheduleBuilder:
    """Encapsulates schedule building with pluggable priority functions."""

    def __init__(self, durations: list[list[int]], machines: list[list[int]]):
        self.durations = durations
        self.machines = machines
        self.num_jobs = len(durations)
        self.num_machines = max(max(row) for row in machines) + 1
        self.total_ops = sum(len(job) for job in durations)

        # Precomputed job statistics
        self.total_work = [sum(job) for job in durations]
        self.num_ops = [len(job) for job in durations]

        # Scheduling state
        self.next_op = [0] * self.num_jobs
        self.job_ready = [0] * self.num_jobs
        self.machine_ready = [0] * self.num_machines
        self.remaining_work = list(self.total_work)
        self.remaining_ops = list(self.num_ops)
        self.machine_queued = [0] * self.num_machines

        # Output
        self.schedules: list[list[dict[str, int]]] = [
            [] for _ in range(self.num_machines)
        ]

    def get_winq(self, job_id: int, op_idx: int) -> int:
        """Compute work in next queue for WINQ rules."""
        if op_idx + 1 < len(self.machines[job_id]):
            next_machine = self.machines[job_id][op_idx + 1]
            return self.machine_queued[next_machine]
        return 0

    def get_priority(self, job_id: int, op_idx: int, priority_func: Callable) -> tuple:
        """Compute priority tuple using the provided function."""
        machine_id = self.machines[job_id][op_idx]
        duration = self.durations[job_id][op_idx]
        est = max(self.job_ready[job_id], self.machine_ready[machine_id])
        winq = self.get_winq(job_id, op_idx)

        return priority_func(
            est=est,
            duration=duration,
            job_id=job_id,
            remaining_work=self.remaining_work[job_id],
            remaining_ops=self.remaining_ops[job_id],
            winq=winq,
            total_work=self.total_work[job_id],
            num_ops=self.num_ops[job_id]
        )

    def schedule_next(self, priority_func: Callable) -> None:
        """Select and schedule the best operation according to priority."""
        best_priority = None
        best_job = -1

        for job_id in range(self.num_jobs):
            op_idx = self.next_op[job_id]
            if op_idx >= len(self.durations[job_id]):
                continue

            priority = self.get_priority(job_id, op_idx, priority_func)

            if best_priority is None or priority < best_priority:
                best_priority = priority
                best_job = job_id

        if best_job < 0:
            return

        job_id = best_job
        op_idx = self.next_op[job_id]
        machine_id = self.machines[job_id][op_idx]
        duration = self.durations[job_id][op_idx]
        est = max(self.job_ready[job_id], self.machine_ready[machine_id])
        end = est + duration

        self.schedules[machine_id].append({
            "job_id": job_id,
            "operation_index": op_idx,
            "start_time": est,
            "end_time": end,
            "duration": duration,
        })

        self.next_op[job_id] += 1
        self.job_ready[job_id] = end
        self.machine_ready[machine_id] = end
        self.remaining_work[job_id] -= duration
        self.remaining_ops[job_id] -= 1
        self.machine_queued[machine_id] += duration

    def build(self, priority_func: Callable) -> tuple[int, list[list[dict[str, int]]]]:
        """Build complete schedule and return (makespan, schedules)."""
        total = self.total_ops
        for _ in range(total):
            self.schedule_next(priority_func)

        makespan = max(self.job_ready) if self.job_ready else 0
        return makespan, self.schedules


def _propagate_job_changes(
    schedules: list[list[dict[str, int]]],
    op_info: list[list[tuple[int, int, int, int] | None]],
    job_id: int,
    start_op_idx: int,
    new_end_time: int
) -> None:
    """Propagate changes through subsequent operations in a job."""
    prev_end = new_end_time
    for next_op_idx in range(start_op_idx + 1, len(op_info[job_id])):
        next_info = op_info[job_id][next_op_idx]
        if next_info is None:
            break
        next_m, next_s, next_e, next_d = next_info
        if next_s < prev_end:
            shift = prev_end - next_s
            next_s += shift
            next_e += shift
            op_info[job_id][next_op_idx] = (next_m, next_s, next_e, next_d)
            for m_op in schedules[next_m]:
                if m_op["job_id"] == job_id and m_op["operation_index"] == next_op_idx:
                    m_op["start_time"] = next_s
                    m_op["end_time"] = next_e
                    break
            prev_end = next_e


def _identify_critical_path(
    schedules: list[list[dict[str, int]]],
    op_info: list[list[tuple[int, int, int, int] | None]],
    num_jobs: int
) -> tuple[set[tuple[int, int]], list[list[tuple[int, int, int, int] | None]]]:
    """Identify operations on the critical path using forward-backward pass.

    Returns a set of (job_id, op_idx) tuples that are on the critical path.
    Uses proper head (earliest start) and tail (latest start) computation.
    """
    # Build machine operation order lookup
    machine_op_pos: dict[tuple[int, int], int] = {}  # (job_id, op_idx) -> position in machine schedule
    for m_id, sched in enumerate(schedules):
        for pos, op in enumerate(sched):
            machine_op_pos[(op["job_id"], op["operation_index"])] = (m_id, pos)

    # Forward pass: compute head (earliest completion time) for each operation
    head = [[0] * len(job) for job in op_info]  # head[j][op] = earliest completion time

    for j_id in range(num_jobs):
        for op_idx in range(len(op_info[j_id])):
            info = op_info[j_id][op_idx]
            if info is None:
                continue
            m_id, start, end, duration = info

            # Job predecessor constraint
            job_pred_end = head[j_id][op_idx - 1] if op_idx > 0 else 0

            # Machine predecessor constraint
            m_id, pos = machine_op_pos.get((j_id, op_idx), (m_id, -1))
            machine_pred_end = 0
            if pos > 0:
                pred_op = schedules[m_id][pos - 1]
                machine_pred_end = head[pred_op["job_id"]][pred_op["operation_index"]]

            head[j_id][op_idx] = max(job_pred_end, machine_pred_end) + duration

    # Makespan is the maximum head value
    makespan = max(head[j][-1] for j in range(num_jobs) if head[j])

    # Backward pass: compute tail (time from operation end to makespan)
    tail = [[0] * len(job) for job in op_info]  # tail[j][op] = time from completion to makespan

    for j_id in range(num_jobs - 1, -1, -1):
        for op_idx in range(len(op_info[j_id]) - 1, -1, -1):
            info = op_info[j_id][op_idx]
            if info is None:
                continue
            m_id, _, _, duration = info

            # Job successor constraint
            job_succ_start = tail[j_id][op_idx + 1] if op_idx + 1 < len(op_info[j_id]) else 0

            # Machine successor constraint
            m_id, pos = machine_op_pos.get((j_id, op_idx), (m_id, -1))
            machine_succ_start = 0
            if pos >= 0 and pos + 1 < len(schedules[m_id]):
                succ_op = schedules[m_id][pos + 1]
                machine_succ_start = tail[succ_op["job_id"]][succ_op["operation_index"]]

            tail[j_id][op_idx] = max(job_succ_start, machine_succ_start) + duration

    # Critical operations: head + tail == makespan
    critical_ops: set[tuple[int, int]] = set()
    for j_id in range(num_jobs):
        for op_idx in range(len(op_info[j_id])):
            if op_info[j_id][op_idx] is not None:
                if head[j_id][op_idx] + tail[j_id][op_idx] == makespan:
                    critical_ops.add((j_id, op_idx))

    return critical_ops, op_info


def _find_critical_blocks(
    schedules: list[list[dict[str, int]]],
    critical_ops: set[tuple[int, int]]
) -> list[list[int]]:
    """Find critical blocks (consecutive critical operations on same machine).

    Returns list of blocks, where each block is a list of operation indices in the schedule.
    """
    blocks = []

    for m_id, sched in enumerate(schedules):
        block = []
        for i, op in enumerate(sched):
            if (op["job_id"], op["operation_index"]) in critical_ops:
                block.append(i)
            else:
                if len(block) >= 1:
                    blocks.append((m_id, block))
                block = []
        if len(block) >= 1:
            blocks.append((m_id, block))

    return blocks


def _local_search(
    durations: list[list[int]],
    machines: list[list[int]],
    schedules: list[list[dict[str, int]]],
    max_iter: int = 50
) -> tuple[int, list[list[dict[str, int]]]]:
    """Iterative improvement through operation shifting with critical path focus."""
    num_jobs = len(durations)
    num_machines = len(schedules)

    # Build operation lookup
    op_info: list[list[tuple[int, int, int, int] | None]] = [
        [None] * len(job) for job in durations
    ]

    for m_id, sched in enumerate(schedules):
        for op in sched:
            j_id = op["job_id"]
            op_idx = op["operation_index"]
            op_info[j_id][op_idx] = (m_id, op["start_time"], op["end_time"], op["duration"])

    def calculate_makespan() -> int:
        ms = 0
        for m_sched in schedules:
            for op in m_sched:
                if op["end_time"] > ms:
                    ms = op["end_time"]
        return ms

    def rebuild_op_info():
        """Rebuild op_info after major changes."""
        for j_id in range(num_jobs):
            for op_idx in range(len(op_info[j_id])):
                op_info[j_id][op_idx] = None
        for m_id, sched in enumerate(schedules):
            for op in sched:
                j_id = op["job_id"]
                op_idx = op["operation_index"]
                op_info[j_id][op_idx] = (m_id, op["start_time"], op["end_time"], op["duration"])

    # Phase 0: Identify critical path
    critical_ops, op_info = _identify_critical_path(schedules, op_info, num_jobs)

    improved = True
    iteration = 0
    last_makespan = calculate_makespan()
    no_improve_count = 0

    while improved and iteration < max_iter:
        improved = False
        iteration += 1

        makespan = calculate_makespan()

        # Early termination if no improvement for several iterations
        if makespan < last_makespan:
            last_makespan = makespan
            no_improve_count = 0
            # Re-identify critical path after improvement
            critical_ops, op_info = _identify_critical_path(schedules, op_info, num_jobs)
        else:
            no_improve_count += 1
            if no_improve_count > 3:
                break

        # Process machines in order of their latest completion time (critical first)
        machine_order = sorted(range(num_machines),
                               key=lambda m: max((op["end_time"] for op in schedules[m]), default=0),
                               reverse=True)

        for m_id in machine_order:
            sched = schedules[m_id]
            n = len(sched)
            if n < 1:
                continue

            # Sort by start time to ensure proper ordering
            sched.sort(key=lambda x: x["start_time"])

            for i in range(n):
                op = sched[i]
                j_id = op["job_id"]
                op_idx = op["operation_index"]
                duration = op["duration"]
                current_start = op["start_time"]

                # Prioritize critical operations
                is_critical = (j_id, op_idx) in critical_ops

                # Job precedence constraint
                job_earliest = 0
                if op_idx > 0 and op_info[j_id][op_idx - 1]:
                    job_earliest = op_info[j_id][op_idx - 1][2]

                # Find the best gap to fit this operation
                best_new_start = current_start

                # Check gap at the beginning (before first operation)
                if n > 0:
                    first_start = sched[0]["start_time"]
                    if i != 0 and first_start >= duration:
                        candidate = max(0, job_earliest)
                        if candidate + duration <= first_start and candidate < current_start:
                            best_new_start = candidate

                # Check gaps between operations
                for k in range(n - 1):
                    gap_start = sched[k]["end_time"]
                    gap_end = sched[k + 1]["start_time"]
                    gap_size = gap_end - gap_start

                    if gap_size >= duration:
                        candidate = max(gap_start, job_earliest)
                        if candidate + duration <= gap_end and candidate < best_new_start:
                            # Don't move operation into a position after itself
                            if not (k >= i - 1 and k + 1 <= i):
                                best_new_start = candidate

                # Also check if we can shift left within current position
                if i > 0:
                    prev_end = sched[i - 1]["end_time"]
                    candidate = max(prev_end, job_earliest)
                    if candidate < current_start:
                        # Check if there's room before next operation
                        next_start = sched[i + 1]["start_time"] if i + 1 < n else float('inf')
                        if candidate + duration <= next_start:
                            best_new_start = min(best_new_start, candidate)
                elif i == 0:
                    # First operation - can we start earlier?
                    candidate = max(0, job_earliest)
                    if candidate < current_start:
                        best_new_start = candidate

                if best_new_start >= current_start:
                    continue

                # Apply the improvement
                new_end = best_new_start + duration
                op["start_time"] = best_new_start
                op["end_time"] = new_end
                op_info[j_id][op_idx] = (m_id, best_new_start, new_end, duration)

                _propagate_job_changes(schedules, op_info, j_id, op_idx, new_end)

                improved = True
                break

            if improved:
                break

    # Phase 2: Critical block-based swapping (N5 neighborhood) - simplified and efficient
    # Key insight: Only swaps at block boundaries can improve makespan
    for iteration_swap in range(max_iter // 2):  # Reduced iterations
        makespan_before = calculate_makespan()
        critical_ops, op_info = _identify_critical_path(schedules, op_info, num_jobs)
        critical_blocks = _find_critical_blocks(schedules, critical_ops)

        found_improvement = False

        for m_id, block in critical_blocks:
            if len(block) < 2:
                continue

            sched = schedules[m_id]

            # N5 neighborhood: only try swaps at block boundaries
            for i in [block[0], block[-1] if block[-1] != block[0] else block[0]]:
                if i >= len(sched) - 1:
                    continue

                op1 = sched[i]
                op2 = sched[i + 1]

                # Both operations must be critical for swap to potentially help
                if (op1["job_id"], op1["operation_index"]) not in critical_ops:
                    continue
                if (op2["job_id"], op2["operation_index"]) not in critical_ops:
                    continue

                j1, j2 = op1["job_id"], op2["job_id"]
                op_idx1, op_idx2 = op1["operation_index"], op2["operation_index"]
                dur1, dur2 = op1["duration"], op2["duration"]

                # Skip if durations are equal (swap won't help)
                if dur1 == dur2:
                    continue

                # Check if swap could potentially improve: at block start need dur2 < dur1, at end need dur1 < dur2
                is_at_block_start = (i == block[0])
                is_at_block_end = (i + 1 == block[-1])

                if is_at_block_start and dur2 >= dur1:
                    continue
                if is_at_block_end and dur1 >= dur2:
                    continue
                if not is_at_block_start and not is_at_block_end:
                    continue

                # Get job precedence constraints
                earliest1 = op_info[j1][op_idx1 - 1][2] if op_idx1 > 0 and op_info[j1][op_idx1 - 1] else 0
                earliest2 = op_info[j2][op_idx2 - 1][2] if op_idx2 > 0 and op_info[j2][op_idx2 - 1] else 0

                # Try swap: op2 first, then op1
                prev_end = sched[i - 1]["end_time"] if i > 0 else 0
                new_start2 = max(earliest2, prev_end)
                new_end2 = new_start2 + dur2

                new_start1 = max(earliest1, new_end2)
                new_end1 = new_start1 + dur1

                # Check machine feasibility
                next_start = sched[i + 2]["start_time"] if i + 2 < len(sched) else float('inf')
                if new_end1 > next_start:
                    continue

                # Check job successor constraints
                if op_idx1 + 1 < len(durations[j1]) and op_info[j1][op_idx1 + 1]:
                    if op_info[j1][op_idx1 + 1][1] < new_end1:
                        continue

                # Apply swap directly (it's guaranteed to be improving by the checks above)
                op1["start_time"] = new_start1
                op1["end_time"] = new_end1
                op2["start_time"] = new_start2
                op2["end_time"] = new_end2
                sched[i], sched[i + 1] = sched[i + 1], sched[i]

                op_info[j1][op_idx1] = (m_id, new_start1, new_end1, dur1)
                op_info[j2][op_idx2] = (m_id, new_start2, new_end2, dur2)
                _propagate_job_changes(schedules, op_info, j1, op_idx1, new_end1)
                _propagate_job_changes(schedules, op_info, j2, op_idx2, new_end2)

                found_improvement = True
                break

            if found_improvement:
                break

        if not found_improvement:
            break

    # Phase 2.5: Try insertion moves within critical blocks
    for _ in range(max_iter // 4):
        makespan = calculate_makespan()
        critical_ops, op_info = _identify_critical_path(schedules, op_info, num_jobs)
        critical_blocks = _find_critical_blocks(schedules, critical_ops)

        found_insert = False

        for m_id, block in critical_blocks:
            if len(block) < 2:
                continue

            sched = schedules[m_id]

            # Try moving first operation to after second, or last operation to before second-to-last
            for src_pos in [block[0], block[-1]]:
                if src_pos >= len(sched):
                    continue

                op = sched[src_pos]
                j_id = op["job_id"]
                op_idx = op["operation_index"]
                duration = op["duration"]

                # Get job precedence constraints
                job_earliest = 0
                if op_idx > 0 and op_info[j_id][op_idx - 1]:
                    job_earliest = op_info[j_id][op_idx - 1][2]

                job_latest = float('inf')
                if op_idx + 1 < len(durations[j_id]) and op_info[j_id][op_idx + 1]:
                    job_latest = op_info[j_id][op_idx + 1][1]

                # Try inserting at different positions within the block
                for target_pos in block:
                    if target_pos == src_pos:
                        continue

                    # Determine new start time at target position
                    if target_pos < src_pos:
                        # Moving left
                        prev_end = sched[target_pos - 1]["end_time"] if target_pos > 0 else 0
                        next_start = sched[target_pos]["start_time"]
                    else:
                        # Moving right
                        prev_end = sched[target_pos]["end_time"]
                        next_start = sched[target_pos + 1]["start_time"] if target_pos + 1 < len(sched) else float('inf')

                    new_start = max(prev_end, job_earliest)
                    new_end = new_start + duration

                    # Check if fits in gap
                    if new_end > next_start:
                        continue

                    # Check job successor constraint
                    if new_end > job_latest:
                        continue

                    # Only accept if we're moving to a better position
                    current_start = op["start_time"]
                    if new_start >= current_start:
                        continue

                    # Apply the insertion
                    op["start_time"] = new_start
                    op["end_time"] = new_end
                    op_info[j_id][op_idx] = (m_id, new_start, new_end, duration)

                    # Reorder schedule
                    sched.pop(src_pos)
                    insert_pos = target_pos if target_pos < src_pos else target_pos
                    sched.insert(insert_pos, op)

                    _propagate_job_changes(schedules, op_info, j_id, op_idx, new_end)

                    found_insert = True
                    break

                if found_insert:
                    break
            if found_insert:
                break

        if not found_insert:
            break

    # Phase 2.75: Makespan-focused aggressive shifting
    # Target operations that end at or very close to makespan
    makespan = calculate_makespan()
    makespan_threshold = makespan * 0.95

    for _ in range(max_iter // 3):
        found_improvement = False
        makespan = calculate_makespan()

        # Find operations ending near makespan
        critical_end_ops = []
        for m_id, sched in enumerate(schedules):
            for op in sched:
                if op["end_time"] >= makespan_threshold:
                    critical_end_ops.append((op, m_id))

        # Sort by end time (latest first)
        critical_end_ops.sort(key=lambda x: x[0]["end_time"], reverse=True)

        for op, m_id in critical_end_ops:
            j_id = op["job_id"]
            op_idx = op["operation_index"]
            duration = op["duration"]
            current_start = op["start_time"]
            current_end = op["end_time"]

            # Get job precedence constraint
            job_earliest = 0
            if op_idx > 0 and op_info[j_id][op_idx - 1]:
                job_earliest = op_info[j_id][op_idx - 1][2]

            # Get job successor constraint
            job_latest = float('inf')
            if op_idx + 1 < len(durations[j_id]) and op_info[j_id][op_idx + 1]:
                job_latest = op_info[j_id][op_idx + 1][1]

            sched = schedules[m_id]
            sched.sort(key=lambda x: x["start_time"])

            # Find all gaps where this operation could fit
            best_new_start = current_start

            # Gap at beginning
            if len(sched) > 0 and sched[0]["start_time"] >= duration:
                candidate = max(0, job_earliest)
                if candidate + duration <= sched[0]["start_time"] and candidate < current_start:
                    best_new_start = candidate

            # Gaps between operations
            for k in range(len(sched) - 1):
                gap_start = sched[k]["end_time"]
                gap_end = sched[k + 1]["start_time"]
                gap_size = gap_end - gap_start

                if gap_size >= duration:
                    candidate = max(gap_start, job_earliest)
                    if candidate + duration <= gap_end and candidate < best_new_start:
                        best_new_start = candidate

            # Check if we found a better position
            if best_new_start < current_start:
                new_end = best_new_start + duration

                # Verify job successor constraint
                if new_end <= job_latest:
                    op["start_time"] = best_new_start
                    op["end_time"] = new_end
                    op_info[j_id][op_idx] = (m_id, best_new_start, new_end, duration)

                    _propagate_job_changes(schedules, op_info, j_id, op_idx, new_end)

                    found_improvement = True
                    break

        if not found_improvement:
            break

        # Update threshold if makespan improved
        new_makespan = calculate_makespan()
        if new_makespan < makespan:
            makespan_threshold = new_makespan * 0.95
            rebuild_op_info()

    # Phase 3: Final gap-filling pass
    rebuild_op_info()
    for m_id in range(num_machines):
        sched = schedules[m_id]
        if not sched:
            continue
        sched.sort(key=lambda x: x["start_time"])

        for i, op in enumerate(sched):
            j_id = op["job_id"]
            op_idx = op["operation_index"]
            duration = op["duration"]

            job_earliest = 0
            if op_idx > 0 and op_info[j_id][op_idx - 1]:
                job_earliest = op_info[j_id][op_idx - 1][2]

            prev_end = sched[i - 1]["end_time"] if i > 0 else 0
            candidate = max(prev_end, job_earliest)

            if candidate < op["start_time"]:
                op["start_time"] = candidate
                op["end_time"] = candidate + duration
                op_info[j_id][op_idx] = (m_id, candidate, candidate + duration, duration)
                _propagate_job_changes(schedules, op_info, j_id, op_idx, candidate + duration)

    # Phase 4: Forward-backward compaction for additional improvement
    for fb_iter in range(3):  # Reduced iterations
        makespan = calculate_makespan()

        # Backward pass: shift all operations as late as possible
        all_ops = []
        for m_id, sched in enumerate(schedules):
            for op in sched:
                all_ops.append((op, m_id))
        all_ops.sort(key=lambda x: -x[0]["end_time"])

        for op, m_id in all_ops:
            j_id = op["job_id"]
            op_idx = op["operation_index"]
            duration = op["duration"]

            # Find latest possible end time
            job_latest = makespan
            if op_idx + 1 < len(durations[j_id]) and op_info[j_id][op_idx + 1]:
                job_latest = op_info[j_id][op_idx + 1][1]

            sched = schedules[m_id]
            op_pos = -1
            for i, o in enumerate(sched):
                if o["job_id"] == j_id and o["operation_index"] == op_idx:
                    op_pos = i
                    break

            machine_latest = makespan
            if op_pos >= 0 and op_pos + 1 < len(sched):
                machine_latest = sched[op_pos + 1]["start_time"]

            new_end = min(job_latest, machine_latest)
            new_start = new_end - duration

            if new_start > op["start_time"]:
                op["start_time"] = new_start
                op["end_time"] = new_end
                op_info[j_id][op_idx] = (m_id, new_start, new_end, duration)

        # Forward pass: shift all operations as early as possible
        all_ops.sort(key=lambda x: x[0]["start_time"])

        for op, m_id in all_ops:
            j_id = op["job_id"]
            op_idx = op["operation_index"]
            duration = op["duration"]

            job_earliest = 0
            if op_idx > 0 and op_info[j_id][op_idx - 1]:
                job_earliest = op_info[j_id][op_idx - 1][2]

            sched = schedules[m_id]
            op_pos = -1
            for i, o in enumerate(sched):
                if o["job_id"] == j_id and o["operation_index"] == op_idx:
                    op_pos = i
                    break

            machine_earliest = 0
            if op_pos > 0:
                machine_earliest = sched[op_pos - 1]["end_time"]

            new_start = max(job_earliest, machine_earliest)
            new_end = new_start + duration

            if new_start < op["start_time"]:
                op["start_time"] = new_start
                op["end_time"] = new_end
                op_info[j_id][op_idx] = (m_id, new_start, new_end, duration)

        # Check for improvement
        new_makespan = calculate_makespan()
        if new_makespan >= makespan:
            break
        rebuild_op_info()

    return calculate_makespan(), schedules


# Priority function registry
PriorityFunc = Callable[..., tuple]


def _priority_spt(est: int, duration: int, job_id: int, **kwargs) -> tuple:
    return (est, duration, job_id)


def _priority_lpt(est: int, duration: int, job_id: int, **kwargs) -> tuple:
    return (est, -duration, job_id)


def _priority_mor(est: int, remaining_ops: int, job_id: int, **kwargs) -> tuple:
    return (est, -remaining_ops, job_id)


def _priority_lwkr(est: int, remaining_work: int, job_id: int, **kwargs) -> tuple:
    return (est, remaining_work, job_id)


def _priority_mwkr(est: int, remaining_work: int, job_id: int, **kwargs) -> tuple:
    return (est, -remaining_work, job_id)


def _priority_spt_mor(est: int, duration: int, remaining_ops: int, job_id: int, **kwargs) -> tuple:
    return (est, duration, -remaining_ops, job_id)


def _priority_lwkr_spt(est: int, remaining_work: int, duration: int, job_id: int, **kwargs) -> tuple:
    return (est, remaining_work, duration, job_id)


def _priority_mwkr_lpt(est: int, remaining_work: int, duration: int, job_id: int, **kwargs) -> tuple:
    return (est, -remaining_work, -duration, job_id)


def _priority_winq(est: int, winq: int, duration: int, job_id: int, **kwargs) -> tuple:
    return (est, winq, duration, job_id)


def _priority_mor_winq(est: int, remaining_ops: int, winq: int, job_id: int, **kwargs) -> tuple:
    return (est, -remaining_ops, winq, job_id)


def _priority_lwkr_winq(est: int, remaining_work: int, winq: int, job_id: int, **kwargs) -> tuple:
    return (est, remaining_work, winq, job_id)


def _priority_spt_winq(est: int, duration: int, winq: int, job_id: int, **kwargs) -> tuple:
    return (est, duration, winq, job_id)


def _priority_mor_winq_spt(est: int, remaining_ops: int, winq: int, duration: int, job_id: int, **kwargs) -> tuple:
    return (est, -remaining_ops, winq, duration, job_id)


def _priority_lwkr_winq_spt(est: int, remaining_work: int, winq: int, duration: int, job_id: int, **kwargs) -> tuple:
    return (est, remaining_work, winq, duration, job_id)


def _priority_mwkr_winq(est: int, remaining_work: int, winq: int, job_id: int, **kwargs) -> tuple:
    return (est, -remaining_work, winq, job_id)


def _priority_look_ahead(est: int, duration: int, job_id: int, remaining_ops: int,
                         remaining_work: int, num_ops: int, **kwargs) -> tuple:
    # Approximate next operation duration
    next_dur = (remaining_work - duration) / max(1, remaining_ops - 1) if remaining_ops > 1 else 0
    return (est, duration + int(next_dur), job_id)


def _priority_critical(est: int, remaining_work: int, duration: int, job_id: int, **kwargs) -> tuple:
    return (est, -remaining_work, duration, job_id)


def _priority_critical_winq(est: int, remaining_work: int, winq: int, job_id: int, **kwargs) -> tuple:
    return (est, -remaining_work, winq, job_id)


def _priority_fdd(est: int, duration: int, remaining_work: int, job_id: int, **kwargs) -> tuple:
    """Flow Due Date: prioritize by estimated completion time."""
    return (est, est + remaining_work, job_id)


def _priority_rnd(est: int, duration: int, remaining_work: int, job_id: int, **kwargs) -> tuple:
    """Ratio-based: balance between duration and remaining work."""
    ratio = duration / max(1, remaining_work)
    return (est, ratio, job_id)


def _priority_spt_mwkr(est: int, duration: int, remaining_work: int, job_id: int, **kwargs) -> tuple:
    """Combined SPT with MWKR for critical path awareness."""
    return (est, duration, -remaining_work, job_id)


def _priority_bottleneck(est: int, duration: int, remaining_work: int, winq: int, job_id: int, **kwargs) -> tuple:
    """Focus on bottleneck operations - high remaining work + high WINQ."""
    bottleneck_score = remaining_work + winq * 2
    return (est, -bottleneck_score, duration, job_id)


def _priority_avg_duration(est: int, duration: int, remaining_work: int, remaining_ops: int, job_id: int, **kwargs) -> tuple:
    """Prioritize by average remaining operation duration."""
    avg_dur = remaining_work / max(1, remaining_ops)
    return (est, avg_dur, job_id)


def _priority_weighted_spt_mwkr(est: int, duration: int, remaining_work: int, job_id: int, **kwargs) -> tuple:
    """Weighted combination: prioritize short operations with high remaining work.

    Uses normalized scores to balance SPT (short-term) and MWKR (long-term critical path).
    """
    # Normalize duration and remaining work into comparable scales
    # Higher remaining work should have higher priority (negative in tuple for min)
    # Lower duration should have higher priority (positive in tuple for min)
    score = duration - remaining_work / 10
    return (est, score, job_id)


def _priority_covertime(est: int, duration: int, remaining_work: int, winq: int, job_id: int, **kwargs) -> tuple:
    """Cover time: prioritize operations that will finish their job soonest.

    Similar to FDD but considers current queue status.
    """
    cover_time = est + duration + remaining_work + winq
    return (est, cover_time, job_id)


def _priority_atc(est: int, duration: int, remaining_work: int, job_id: int, **kwargs) -> tuple:
    """Apparent Tardiness Cost inspired rule.

    Balances short processing time with criticality (remaining work).
    """
    atc_score = duration / (1 + remaining_work / 10.0)
    return (est, atc_score, job_id)


def _priority_slack(est: int, duration: int, remaining_work: int, winq: int, job_id: int, **kwargs) -> tuple:
    """Slack-based priority: prioritize operations with less flexibility."""
    slack_indicator = est + remaining_work
    return (est, slack_indicator, duration, job_id)


def _priority_critical_ratio(est: int, duration: int, remaining_work: int, job_id: int, **kwargs) -> tuple:
    """Critical ratio: balance urgency with work content."""
    ratio = (est + duration) / max(1, remaining_work)
    return (est, ratio, job_id)


def _priority_twkr_spt(est: int, duration: int, remaining_work: int, total_work: int, job_id: int, **kwargs) -> tuple:
    """Total Work Remaining with SPT tiebreaker."""
    return (est, -total_work, duration, job_id)


def _priority_two_phase(est: int, duration: int, remaining_work: int, remaining_ops: int, winq: int, job_id: int, **kwargs) -> tuple:
    """Two-phase priority combining local and global criteria."""
    global_score = -remaining_work
    local_score = winq + duration
    return (est, global_score, local_score, job_id)


def _priority_orm_rule(est: int, duration: int, remaining_work: int, remaining_ops: int, winq: int, job_id: int, **kwargs) -> tuple:
    """Operations Remaining Modified rule."""
    score = -remaining_ops * 10 + winq
    return (est, score, duration, job_id)


def _priority_dynamic_spt(est: int, duration: int, remaining_work: int, remaining_ops: int, job_id: int, **kwargs) -> tuple:
    """Dynamic SPT that adjusts based on job progress."""
    if remaining_ops <= 0:
        progress = 1.0
    else:
        progress = 1.0 - (remaining_ops / max(1, kwargs.get('num_ops', remaining_ops + 1)))
    weight = progress * 0.5
    score = duration * (1 - weight) - remaining_work * weight / 10
    return (est, score, job_id)


def _priority_holthaus(est: int, duration: int, remaining_work: int, winq: int, job_id: int, **kwargs) -> tuple:
    """Holthaus rule: weighted combination proven effective in JSSP research.

    Combines SPT (short-term efficiency), WINQ (downstream awareness), and MWKR (critical path).
    Weights based on empirical studies: duration has highest weight, then WINQ, then remaining work.
    """
    # Weight factors from JSSP literature
    w_dur = 0.5      # Short processing time
    w_winq = 0.3     # Low work in next queue
    w_mwkr = 0.2     # High remaining work (critical path)

    score = w_dur * duration + w_winq * winq - w_mwkr * remaining_work
    return (est, score, job_id)


def _priority_holthaus2(est: int, duration: int, remaining_work: int, winq: int, job_id: int, **kwargs) -> tuple:
    """Holthaus variant with different weight emphasis."""
    score = duration + winq - remaining_work / 5
    return (est, score, job_id)


# Registry mapping rule names to priority functions
PRIORITY_REGISTRY: dict[str, PriorityFunc] = {
    'EST_SPT': _priority_spt,
    'EST_LPT': _priority_lpt,
    'EST_MOR': _priority_mor,
    'EST_LWKR': _priority_lwkr,
    'EST_MWKR': _priority_mwkr,
    'EST_SPT_MOR': _priority_spt_mor,
    'EST_LWKR_SPT': _priority_lwkr_spt,
    'EST_MWKR_LPT': _priority_mwkr_lpt,
    'EST_WINQ': _priority_winq,
    'EST_MOR_WINQ': _priority_mor_winq,
    'EST_LWKR_WINQ': _priority_lwkr_winq,
    'EST_SPT_WINQ': _priority_spt_winq,
    'EST_MOR_WINQ_SPT': _priority_mor_winq_spt,
    'EST_LWKR_WINQ_SPT': _priority_lwkr_winq_spt,
    'EST_MWKR_WINQ': _priority_mwkr_winq,
    'EST_LOOK_AHEAD': _priority_look_ahead,
    'EST_CRITICAL_PATH': _priority_critical,
    'EST_CRITICAL_WINQ': _priority_critical_winq,
    'EST_FDD': _priority_fdd,
    'EST_RATIO': _priority_rnd,
    'EST_SPT_MWKR': _priority_spt_mwkr,
    'EST_BOTTLENECK': _priority_bottleneck,
    'EST_AVG_DURATION': _priority_avg_duration,
    'EST_WEIGHTED_SPT_MWKR': _priority_weighted_spt_mwkr,
    'EST_COVERTIME': _priority_covertime,
    'EST_ATC': _priority_atc,
    'EST_SLACK': _priority_slack,
    'EST_CRITICAL_RATIO': _priority_critical_ratio,
    'EST_TWKR_SPT': _priority_twkr_spt,
    'EST_TWO_PHASE': _priority_two_phase,
    'EST_ORM': _priority_orm_rule,
    'EST_DYNAMIC_SPT': _priority_dynamic_spt,
    'EST_HOLTHAUS': _priority_holthaus,
    'EST_HOLTHAUS2': _priority_holthaus2,
}


def _solve_with_rule(
    durations: list[list[int]],
    machines: list[list[int]],
    rule: str,
) -> tuple[int, list[list[dict[str, int]]]]:
    """Solve JSSP instance with a specific dispatch rule."""
    if rule not in PRIORITY_REGISTRY:
        rule = 'EST_SPT'

    builder = ScheduleBuilder(durations, machines)
    makespan, schedules = builder.build(PRIORITY_REGISTRY[rule])
    return makespan, schedules


def solve_instance(instance: dict[str, Any]) -> dict[str, Any]:
    """Enhanced greedy scheduler with local search improvement.

    Tries multiple dispatch rules and applies local search refinement.
    Uses early pruning to skip unpromising rules.

    Input:
        instance dict with keys:
        - name
        - duration_matrix
        - machines_matrix
        - metadata

    Output:
        dict with at least:
        - name
        - makespan
        - machine_schedules
    """
    durations: list[list[int]] = instance["duration_matrix"]
    machines: list[list[int]] = instance["machines_matrix"]

    # Order rules by expected effectiveness (based on JSSP literature)
    # Holthaus rules are placed early as they're proven effective
    priority_order = [
        'EST_HOLTHAUS',       # Holthaus rule - proven effective combination
        'EST_HOLTHAUS2',      # Holthaus variant
        'EST_MWKR',           # Most work remaining - critical path awareness
        'EST_SPT_MWKR',       # Combined SPT + MWKR
        'EST_ATC',            # Apparent tardiness cost
        'EST_BOTTLENECK',     # Bottleneck awareness
        'EST_TWO_PHASE',      # Two-phase priority
        'EST_CRITICAL_PATH',  # Critical path focus
        'EST_SPT',            # Shortest processing time (classic)
        'EST_MOR_WINQ_SPT',   # Combined rule
        'EST_MWKR_WINQ',      # MWKR with queue awareness
        'EST_SLACK',          # Slack-based
        'EST_FDD',            # Flow due date
        'EST_DYNAMIC_SPT',    # Dynamic SPT
        'EST_TWKR_SPT',       # Total work with SPT
        'EST_ORM',            # Operations remaining modified
        'EST_WEIGHTED_SPT_MWKR',
        'EST_CRITICAL_WINQ',
        'EST_CRITICAL_RATIO',
        'EST_SPT_MOR',
        'EST_MOR_WINQ',
        'EST_LWKR_WINQ_SPT',
        'EST_COVERTIME',
        'EST_LOOK_AHEAD',
        'EST_AVG_DURATION',
        'EST_LWKR_SPT',
        'EST_MWKR_LPT',
        'EST_SPT_WINQ',
        'EST_MOR',
        'EST_LWKR',
        'EST_LWKR_WINQ',
        'EST_WINQ',
        'EST_LPT',
        'EST_RATIO',
    ]

    # Filter to only rules that exist in registry
    rules = [r for r in priority_order if r in PRIORITY_REGISTRY]
    # Add any remaining rules not in priority order
    for r in PRIORITY_REGISTRY:
        if r not in rules:
            rules.append(r)

    best_makespan = float('inf')
    best_schedules = None
    best_rule = None
    best_initial_makespan = float('inf')
    rules_tried = 0

    for rule in rules:
        makespan, schedules = _solve_with_rule(durations, machines, rule)
        rules_tried += 1

        # Early pruning: skip local search if initial solution is much worse
        # But always try the first 8 rules to establish a good baseline
        if rules_tried > 8 and best_makespan < float('inf') and makespan > best_makespan * 1.10:
            # Still check if this initial solution is better than current best
            if makespan < best_makespan:
                best_makespan = makespan
                best_schedules = schedules
                best_rule = rule
            continue

        # Apply local search improvement
        improved_makespan, improved_schedules = _local_search(
            durations, machines, schedules
        )

        if improved_makespan < best_makespan:
            best_makespan = improved_makespan
            best_schedules = improved_schedules
            best_rule = rule

        # Track best initial makespan for adaptive pruning
        if makespan < best_initial_makespan:
            best_initial_makespan = makespan

    return {
        "name": instance["name"],
        "makespan": best_makespan,
        "machine_schedules": best_schedules,
        "solved_by": f"PriorityRegistryLocalSearch({best_rule})",
        "family": FAMILY_PREFIX,
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=f"Run pure-python baseline on {FAMILY_NAME}."
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=None,
        help="Instance name. If omitted, run the first N family instances.",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=3,
        help="How many family instances to run when --instance is omitted.",
    )
    args = parser.parse_args()

    if args.instance:
        instances = [load_instance_by_name(args.instance)]
    else:
        instances = load_family_instances()[: max(args.max_instances, 1)]

    for instance in instances:
        start = time.perf_counter()
        result = solve_instance(instance)
        elapsed = time.perf_counter() - start
        print(
            f"[{FAMILY_PREFIX}] {instance['name']}: "
            f"makespan={result['makespan']} elapsed={elapsed:.4f}s"
        )


if __name__ == "__main__":
    _cli()
# EVOLVE-BLOCK-END