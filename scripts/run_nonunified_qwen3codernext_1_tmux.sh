#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
MATRIX_PATH="${MATRIX_PATH:-$REPO_ROOT/frontier_eval/conf/batch/nonunified_qwen3codernext_1.yaml}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
SESSION_NAME="${SESSION_NAME:-nonuni_qwen3next1_$(date -u +%Y%m%d_%H%M%S)}"
LOG_PATH="${LOG_PATH:-$REPO_ROOT/runs/batch/tmux_${SESSION_NAME}.log}"
UNSET_PROXY="${UNSET_PROXY:-1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$MATRIX_PATH" ]]; then
  echo "matrix file not found: $MATRIX_PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_PATH")"

proxy_prefix=""
if [[ "$UNSET_PROXY" == "1" ]]; then
  proxy_prefix="env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy"
fi

cmd="cd $(printf '%q' "$REPO_ROOT") && export PATH=$(printf '%q' "$REPO_ROOT/.venv/bin"):\$PATH && export SHINKA_PYTHON_EXECUTABLE=$(printf '%q' "$PYTHON_BIN") && export FRONTIER_ENGINEERING_ROOT=$(printf '%q' "$REPO_ROOT") && $proxy_prefix $(printf '%q' "$PYTHON_BIN") -m frontier_eval.batch --matrix $(printf '%q' "$MATRIX_PATH") --python $(printf '%q' "$PYTHON_BIN") --max-parallel $(printf '%q' "$MAX_PARALLEL")"
tmux new-session -d -s "$SESSION_NAME" "$cmd"
tmux pipe-pane -o -t "$SESSION_NAME" "cat >> $(printf '%q' "$LOG_PATH")"

cat <<EOF
session_name=$SESSION_NAME
matrix_path=$MATRIX_PATH
log_path=$LOG_PATH
unset_proxy=$UNSET_PROXY
attach_command=tmux attach -t $SESSION_NAME
EOF
