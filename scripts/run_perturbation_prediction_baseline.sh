#!/usr/bin/env bash
set -euo pipefail

ROOT="${FRONTIER_ENGINEERING_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
V2_PY="$ROOT/.venvs/frontier-v2-extra/bin/python"
TASK_DIR="$ROOT/benchmarks/SingleCellAnalysis/perturbation_prediction"
OUTPUT="${1:-$TASK_DIR/prediction.h5ad}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/run_perturbation_prediction_baseline.sh [output.h5ad]

Generates the mean-across-compounds baseline prediction and evaluates it.
Fetch the dataset first with:
  scripts/data/fetch_perturbation_prediction.sh
EOF
  exit 2
fi

if [[ ! -x "$V2_PY" ]]; then
  echo "Missing v2 environment python: $V2_PY" >&2
  echo "Run: bash $ROOT/scripts/env/setup_v2_task_envs.sh" >&2
  exit 1
fi

cd "$ROOT"

echo "[1/2] Generate baseline prediction -> $OUTPUT"
"$V2_PY" "$TASK_DIR/baseline/run_mean_across_compounds.py" --output "$OUTPUT"

echo "[2/2] Evaluate prediction"
"$V2_PY" "$TASK_DIR/verification/evaluate_perturbation_prediction.py" --prediction "$OUTPUT"
