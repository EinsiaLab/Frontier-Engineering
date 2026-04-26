#!/usr/bin/env bash
set -euo pipefail

ROOT="${FRONTIER_ENGINEERING_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
V2_PY="$ROOT/.venvs/frontier-v2-extra/bin/python"

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
Usage:
  scripts/run_v2_unified.sh <Domain/Task> [extra frontier_eval args...]

Example:
  scripts/run_v2_unified.sh CommunicationEngineering/RayleighFadingBER algorithm=openevolve algorithm.iterations=0
EOF
  exit 2
fi

BENCHMARK="$1"
shift

if [[ ! -x "$V2_PY" ]]; then
  echo "Missing v2 environment python: $V2_PY" >&2
  echo "Run: bash $ROOT/scripts/env/setup_v2_task_envs.sh" >&2
  exit 1
fi

cd "$ROOT"

export FRONTIER_EVAL_UNIFIED_RUNTIME_ENV="${FRONTIER_EVAL_UNIFIED_RUNTIME_ENV:-frontier-v2-extra}"

exec "$V2_PY" -m frontier_eval \
  task=unified \
  "task.benchmark=$BENCHMARK" \
  "$@"
