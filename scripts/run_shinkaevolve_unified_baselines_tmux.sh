#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

DEFAULT_PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$DEFAULT_PYTHON_BIN" ]]; then
    DEFAULT_PYTHON_BIN="python3"
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
EXPECTED_COUNT="${EXPECTED_COUNT:-49}"
EVALUATOR_TIMEOUT_S="${EVALUATOR_TIMEOUT_S:-300}"
RESUME="${RESUME:-1}"
BENCHMARK_FILTER="${BENCHMARK_FILTER:-}"
BATCH_ROOT="${BATCH_ROOT:-$REPO_ROOT/runs/batch/shinkaevolve_unified_baselines__$(date -u +%Y%m%d_%H%M%S)__${RANDOM}${RANDOM}}"
SESSION_NAME="${SESSION_NAME:-shinka_baselines_$(date -u +%Y%m%d_%H%M%S)}"

WORKER_MODE=0
FOREGROUND_MODE=0
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage:
  scripts/run_shinkaevolve_unified_baselines_tmux.sh
  scripts/run_shinkaevolve_unified_baselines_tmux.sh --foreground
  scripts/run_shinkaevolve_unified_baselines_tmux.sh --dry-run

Behavior:
  - Discovers all unified benchmarks under `benchmarks/**/frontier_eval`
    that define both `initial_program.txt` and `eval_command.txt`.
  - Runs each benchmark with `algorithm=shinkaevolve` and `algorithm.max_generations=0`.
  - By default, launches the batch inside a detached tmux session.

Useful env vars:
  PYTHON_BIN=.venv/bin/python
  BATCH_ROOT=runs/batch/my_batch
  SESSION_NAME=my_tmux_session
  EXPECTED_COUNT=49
  EVALUATOR_TIMEOUT_S=300
  RESUME=1
  BENCHMARK_FILTER=MallocLab
EOF
}

safe_slug() {
    printf '%s' "$1" | sed -E 's#[^A-Za-z0-9._-]+#_#g; s#^[-._]+##; s#[-._]+$##'
}

discover_benchmarks() {
    "$PYTHON_BIN" - "$REPO_ROOT" "$EXPECTED_COUNT" "$BENCHMARK_FILTER" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(sys.argv[1]).resolve()
expected_count = int(sys.argv[2])
benchmark_filter = str(sys.argv[3] or "").strip().lower()

required = ("initial_program.txt", "eval_command.txt")
benchmarks: list[str] = []

for meta_dir in sorted(repo_root.glob("benchmarks/**/frontier_eval")):
    if not meta_dir.is_dir():
        continue
    if any(not (meta_dir / name).is_file() for name in required):
        continue
    benchmark_dir = meta_dir.parent.resolve()
    benchmark_id = benchmark_dir.relative_to((repo_root / "benchmarks").resolve()).as_posix()
    if benchmark_filter and benchmark_filter not in benchmark_id.lower():
        continue
    benchmarks.append(benchmark_id)

if not benchmarks:
    raise SystemExit("No runnable unified benchmarks found.")

if not benchmark_filter and expected_count > 0 and len(benchmarks) != expected_count:
    raise SystemExit(
        f"Expected {expected_count} runnable unified benchmarks, found {len(benchmarks)}."
    )

for benchmark in benchmarks:
    print(benchmark)
PY
}

benchmark_runtime_overrides() {
    local benchmark="$1"

    case "$benchmark" in
    ReactionOptimisation/*)
        printf '%s\n' "task.runtime.conda_env=summit"
        ;;
    SustainableDataCenterControl/*)
        printf '%s\n' "task.runtime.conda_env=sustaindc"
        ;;
    esac
}

write_launcher_result() {
    local benchmark="$1"
    local run_dir="$2"
    local log_path="$3"
    local exit_code="$4"
    local started_at="$5"
    local finished_at="$6"
    local duration_s="$7"
    local index="$8"
    local total="$9"

    "$PYTHON_BIN" - "$benchmark" "$run_dir" "$log_path" "$exit_code" "$started_at" "$finished_at" "$duration_s" "$index" "$total" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

benchmark = sys.argv[1]
run_dir = Path(sys.argv[2]).resolve()
log_path = Path(sys.argv[3]).resolve()
exit_code = int(sys.argv[4])
started_at = sys.argv[5]
finished_at = sys.argv[6]
duration_s = float(sys.argv[7])
index = int(sys.argv[8])
total = int(sys.argv[9])

results_dir = run_dir / "shinkaevolve" / "gen_0" / "results"
metrics_path = results_dir / "metrics.json"
correct_path = results_dir / "correct.json"
artifacts_path = results_dir / "artifacts.json"
manifest_path = results_dir / "context_manifest.json"
text_feedback_path = results_dir / "text_feedback.txt"
stdout_bridge_path = results_dir / "stdout_bridge.txt"
stderr_bridge_path = results_dir / "stderr_bridge.txt"
best_info_path = run_dir / "shinkaevolve" / "best" / "best_program_info.json"


def _read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


metrics = _read_json(metrics_path)
correct = _read_json(correct_path)
manifest = _read_json(manifest_path)
best_info = _read_json(best_info_path)

baseline_correct = None
baseline_error = ""
if isinstance(correct, dict):
    if isinstance(correct.get("correct"), bool):
        baseline_correct = bool(correct["correct"])
    baseline_error = str(correct.get("error") or "")

score = None
valid = None
timeout = None
runtime_s = None
if isinstance(metrics, dict):
    score = metrics.get("combined_score")
    valid = metrics.get("valid")
    timeout = metrics.get("timeout")
    runtime_s = metrics.get("runtime_s")

if exit_code != 0:
    status = "launcher_failed"
elif baseline_correct is True:
    status = "baseline_correct"
elif baseline_correct is False:
    status = "baseline_incorrect"
else:
    status = "baseline_unknown"

record = {
    "index": index,
    "total": total,
    "benchmark": benchmark,
    "task": "unified",
    "algorithm": "shinkaevolve",
    "max_generations": 0,
    "status": status,
    "exit_code": exit_code,
    "started_at": started_at,
    "finished_at": finished_at,
    "duration_s": duration_s,
    "run_dir": str(run_dir),
    "log_path": str(log_path),
    "baseline": {
        "correct": baseline_correct,
        "error": baseline_error,
        "combined_score": score,
        "valid": valid,
        "timeout": timeout,
        "runtime_s": runtime_s,
        "metrics_path": str(metrics_path) if metrics_path.is_file() else "",
        "correct_path": str(correct_path) if correct_path.is_file() else "",
        "artifacts_path": str(artifacts_path) if artifacts_path.is_file() else "",
        "context_manifest_path": str(manifest_path) if manifest_path.is_file() else "",
        "text_feedback_path": str(text_feedback_path) if text_feedback_path.is_file() else "",
        "stdout_bridge_path": str(stdout_bridge_path) if stdout_bridge_path.is_file() else "",
        "stderr_bridge_path": str(stderr_bridge_path) if stderr_bridge_path.is_file() else "",
    },
    "best_info_path": str(best_info_path) if best_info_path.is_file() else "",
    "best_metrics": best_info.get("metrics") if isinstance(best_info, dict) else None,
    "context_manifest": manifest if isinstance(manifest, dict) else None,
}

print(json.dumps(record, ensure_ascii=False))
PY
}

run_worker() {
    mkdir -p "$BATCH_ROOT"

    mapfile -t BENCHMARKS < <(discover_benchmarks)
    local total="${#BENCHMARKS[@]}"

    printf '%s\n' "${BENCHMARKS[@]}" >"$BATCH_ROOT/benchmarks.txt"
    printf 'python_bin=%s\n' "$PYTHON_BIN" >"$BATCH_ROOT/run_config.env"
    printf 'expected_count=%s\n' "$EXPECTED_COUNT" >>"$BATCH_ROOT/run_config.env"
    printf 'evaluator_timeout_s=%s\n' "$EVALUATOR_TIMEOUT_S" >>"$BATCH_ROOT/run_config.env"
    printf 'resume=%s\n' "$RESUME" >>"$BATCH_ROOT/run_config.env"
    printf 'benchmark_filter=%s\n' "$BENCHMARK_FILTER" >>"$BATCH_ROOT/run_config.env"

    local summary_path="$BATCH_ROOT/summary.jsonl"
    local progress_path="$BATCH_ROOT/progress.log"
    local counts_path="$BATCH_ROOT/counts.txt"

    touch "$summary_path" "$progress_path"

    local launcher_failed=0
    local baseline_correct=0
    local baseline_incorrect=0
    local baseline_unknown=0
    local skipped=0

    echo "Batch root: $BATCH_ROOT" | tee -a "$progress_path"
    echo "Benchmarks: $total" | tee -a "$progress_path"
    echo "Python: $PYTHON_BIN" | tee -a "$progress_path"

    local idx=0
    for benchmark in "${BENCHMARKS[@]}"; do
        idx=$((idx + 1))

        local slug
        slug="$(safe_slug "${benchmark//\//__}")"
        local run_dir="$BATCH_ROOT/$slug"
        local log_path="$run_dir/launcher.log"
        local launcher_result_path="$run_dir/launcher_result.json"

        if [[ "$RESUME" == "1" && -f "$launcher_result_path" ]]; then
            skipped=$((skipped + 1))
            echo "[$idx/$total] SKIP $benchmark (existing launcher_result.json)" | tee -a "$progress_path"
            continue
        fi

        mkdir -p "$run_dir"

        local started_at finished_at exit_code duration_s
        started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        local start_epoch
        start_epoch="$(date +%s)"
        local extra_args=()
        mapfile -t extra_args < <(benchmark_runtime_overrides "$benchmark")

        echo "[$idx/$total] RUN  $benchmark" | tee -a "$progress_path"

        if [[ "$DRY_RUN" == "1" ]]; then
            {
                printf 'DRY RUN: %q ' "$PYTHON_BIN"
                printf '%q ' -m frontier_eval
                printf '%q ' task=unified
                printf '%q ' "task.benchmark=$benchmark"
                if ((${#extra_args[@]} > 0)); then
                    printf '%q ' "${extra_args[@]}"
                fi
                printf '%q ' algorithm=shinkaevolve
                printf '%q ' algorithm.max_generations=0
                printf '%q ' "algorithm.evaluator_timeout_s=$EVALUATOR_TIMEOUT_S"
                printf '%q ' "run.output_dir=$run_dir"
                printf '\n'
            } | tee -a "$progress_path"
            continue
        fi

        set +e
        FRONTIER_ENGINEERING_ROOT="$REPO_ROOT" \
        SHINKA_PYTHON_EXECUTABLE="$PYTHON_BIN" \
        PATH="$REPO_ROOT/.venv/bin:$PATH" \
        PYTHONUNBUFFERED=1 \
        "$PYTHON_BIN" -m frontier_eval \
            task=unified \
            "task.benchmark=$benchmark" \
            "${extra_args[@]}" \
            algorithm=shinkaevolve \
            algorithm.max_generations=0 \
            "algorithm.evaluator_timeout_s=$EVALUATOR_TIMEOUT_S" \
            "run.output_dir=$run_dir" \
            >"$log_path" 2>&1
        exit_code=$?
        set -e

        local end_epoch
        end_epoch="$(date +%s)"
        finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        duration_s="$((end_epoch - start_epoch))"

        local record
        record="$(write_launcher_result "$benchmark" "$run_dir" "$log_path" "$exit_code" "$started_at" "$finished_at" "$duration_s" "$idx" "$total")"
        printf '%s\n' "$record" >"$launcher_result_path"
        printf '%s\n' "$record" >>"$summary_path"

        local status
        status="$("$PYTHON_BIN" - "$launcher_result_path" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
obj = json.loads(path.read_text(encoding="utf-8"))
print(obj.get("status", "unknown"))
PY
)"

        case "$status" in
        baseline_correct)
            baseline_correct=$((baseline_correct + 1))
            ;;
        baseline_incorrect)
            baseline_incorrect=$((baseline_incorrect + 1))
            ;;
        baseline_unknown)
            baseline_unknown=$((baseline_unknown + 1))
            ;;
        launcher_failed)
            launcher_failed=$((launcher_failed + 1))
            ;;
        *)
            baseline_unknown=$((baseline_unknown + 1))
            ;;
        esac

        printf 'launcher_failed=%s\nbaseline_correct=%s\nbaseline_incorrect=%s\nbaseline_unknown=%s\nskipped=%s\ntotal=%s\n' \
            "$launcher_failed" "$baseline_correct" "$baseline_incorrect" "$baseline_unknown" "$skipped" "$total" \
            >"$counts_path"

        echo "[$idx/$total] DONE $benchmark :: $status (exit=$exit_code, duration=${duration_s}s)" | tee -a "$progress_path"
    done

    echo "Finished batch. launcher_failed=$launcher_failed baseline_correct=$baseline_correct baseline_incorrect=$baseline_incorrect baseline_unknown=$baseline_unknown skipped=$skipped total=$total" \
        | tee -a "$progress_path"
}

start_tmux() {
    command -v tmux >/dev/null 2>&1 || {
        echo "tmux not found in PATH" >&2
        exit 1
    }

    mkdir -p "$BATCH_ROOT"
    mapfile -t BENCHMARKS < <(discover_benchmarks)
    printf '%s\n' "${BENCHMARKS[@]}" >"$BATCH_ROOT/benchmarks.txt"

    local quoted_repo quoted_script quoted_python quoted_batch quoted_filter
    quoted_repo="$(printf '%q' "$REPO_ROOT")"
    quoted_script="$(printf '%q' "$SCRIPT_PATH")"
    quoted_python="$(printf '%q' "$PYTHON_BIN")"
    quoted_batch="$(printf '%q' "$BATCH_ROOT")"
    quoted_filter="$(printf '%q' "$BENCHMARK_FILTER")"

    local session="$SESSION_NAME"
    if tmux has-session -t "$session" 2>/dev/null; then
        session="${session}_$RANDOM"
    fi

    local cmd
    cmd="cd $quoted_repo && export PATH=$(printf '%q' "$REPO_ROOT/.venv/bin"):\$PATH && PYTHON_BIN=$quoted_python SHINKA_PYTHON_EXECUTABLE=$quoted_python BATCH_ROOT=$quoted_batch EXPECTED_COUNT=$EXPECTED_COUNT EVALUATOR_TIMEOUT_S=$EVALUATOR_TIMEOUT_S RESUME=$RESUME BENCHMARK_FILTER=$quoted_filter bash $quoted_script --worker"

    tmux new-session -d -s "$session" "$cmd"
    tmux pipe-pane -o -t "$session" "cat >> $(printf '%q' "$BATCH_ROOT/tmux.log")"

    printf 'session_name=%s\n' "$session" >"$BATCH_ROOT/tmux_session.env"
    printf 'batch_root=%s\n' "$BATCH_ROOT" >>"$BATCH_ROOT/tmux_session.env"
    printf 'attach_command=tmux attach -t %s\n' "$session" >>"$BATCH_ROOT/tmux_session.env"

    echo "Started tmux session: $session"
    echo "Batch root: $BATCH_ROOT"
    echo "Attach with: tmux attach -t $session"
}

while (($# > 0)); do
    case "$1" in
    --worker)
        WORKER_MODE=1
        ;;
    --foreground)
        FOREGROUND_MODE=1
        ;;
    --dry-run)
        DRY_RUN=1
        FOREGROUND_MODE=1
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
    shift
done

if [[ "$WORKER_MODE" == "1" || "$FOREGROUND_MODE" == "1" ]]; then
    run_worker
else
    start_tmux
fi
