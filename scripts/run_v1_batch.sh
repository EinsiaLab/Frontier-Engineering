#!/usr/bin/env bash
# Launcher for the unified v1 batch matrix (47 tasks in frontier_eval/conf/batch/v1.yaml).
#
# Prerequisites: bash init.sh (or equivalent), .venvs/frontier-eval-2, and a configured .env
# for evolution runs. Merged task envs can be prepared with scripts/setup_v1_merged_task_envs.sh.
#
# Optional before launch:
#   export CUDA_VISIBLE_DEVICES=0          # GPU-heavy tasks
#   export ENGDESIGN_EVAL_MODE=docker      # EngDesign — see benchmarks/EngDesign/README.md
#   export ENGDESIGN_DOCKER_IMAGE=...
#
# Extra CLI arguments are forwarded to: python -m frontier_eval.batch
# Examples:
#   bash scripts/run_v1_batch.sh --dry-run
#   bash scripts/run_v1_batch.sh --override algorithm.iterations=0
#   bash scripts/run_v1_batch.sh --max-parallel 2
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/lib_uv_env.sh"

DRIVER_PY="${DRIVER_PY:-}"
DRIVER_ENV="${DRIVER_ENV:-frontier-eval-2}"
V1_MATRIX="${V1_MATRIX:-frontier_eval/conf/batch/v1.yaml}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUTF8="${PYTHONUTF8:-1}"

if [[ ! -f "$V1_MATRIX" ]]; then
  echo "error: matrix not found: $ROOT/$V1_MATRIX" >&2
  exit 1
fi

if [[ -n "$DRIVER_PY" ]]; then
  DRIVER_CMD=("$DRIVER_PY")
else
  ensure_uv_in_path
  DRIVER_PY="$(uv_env_python "$ROOT" "$DRIVER_ENV")"
  if [[ ! -x "$DRIVER_PY" ]]; then
    cat >&2 <<EOF
driver python not found: $DRIVER_PY

Run one of:
  bash init.sh
  bash scripts/setup_v1_merged_task_envs.sh
EOF
    exit 127
  fi
  DRIVER_CMD=("$DRIVER_PY")
fi

prepend_uv_env_to_path "$ROOT" "$DRIVER_ENV"

echo "== Frontier-Eng v1 batch =="
echo "    repo: $ROOT"
echo "    matrix: $V1_MATRIX"
echo "    driver: ${DRIVER_PY}"
echo ""

exec "${DRIVER_CMD[@]}" -m frontier_eval.batch \
  --matrix "$V1_MATRIX" \
  "$@"
