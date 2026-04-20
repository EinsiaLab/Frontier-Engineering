#!/usr/bin/env bash
# Launcher for the unified v1 batch matrix (47 tasks in frontier_eval/conf/batch/v1.yaml).
#
# Prerequisites: bash init.sh (or equivalent), conda env frontier-eval-2, .env with OPENAI_API_KEY
# for evolution runs; merged task envs if needed (scripts/setup_v1_merged_task_envs.sh).
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

DRIVER_PY="${DRIVER_PY:-}"
DRIVER_ENV="${DRIVER_ENV:-frontier-eval-2}"
V1_MATRIX="${V1_MATRIX:-frontier_eval/conf/batch/v1.yaml}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
# Helps UTF-8 on some Windows/Python setups when using conda-run or mixed paths
export PYTHONUTF8="${PYTHONUTF8:-1}"

if [[ ! -f "$V1_MATRIX" ]]; then
  echo "error: matrix not found: $ROOT/$V1_MATRIX" >&2
  exit 1
fi

if [[ -n "$DRIVER_PY" ]]; then
  DRIVER_CMD=("$DRIVER_PY")
else
  if ! command -v conda >/dev/null 2>&1; then
    cat >&2 <<'EOF'
conda not found. Install Miniconda/Anaconda, or set DRIVER_PY to your frontier-eval-2 python, e.g.
  export DRIVER_PY="$HOME/miniconda3/envs/frontier-eval-2/bin/python"
  bash scripts/run_v1_batch.sh
EOF
    exit 127
  fi
  DRIVER_CMD=(conda run -n "$DRIVER_ENV" python)
fi

echo "== Frontier-Eng v1 batch =="
echo "    repo: $ROOT"
echo "    matrix: $V1_MATRIX"
echo "    driver: ${DRIVER_PY:-conda run -n $DRIVER_ENV python}"
echo ""

exec "${DRIVER_CMD[@]}" -m frontier_eval.batch \
  --matrix "$V1_MATRIX" \
  "$@"
