#!/usr/bin/env bash
# setup_uv_envs.sh — create uv virtual environments for Frontier-Engineering tasks.
#
# Usage:
#   bash scripts/setup_uv_envs.sh [--python 3.12] [--venvs-dir .venvs]
#
# Creates four environments under <venvs-dir>/:
#   fe-base          — CoFlyers, Dawn, DuckDB, EV2Gym, PyMOTO, ProtonTherapy
#   fe-jobshop       — all JobShop families (ft, la, orb, yn, abz, swv, ta)
#   fe-pyportfolioopt — PyPortfolioOpt tasks
#   fe-optics        — all 16 Optics tasks
#
# Requires: uv (https://github.com/astral-sh/uv)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="${1:-3.12}"
VENVS_DIR="${VENVS_DIR:-${ROOT}/.venvs}"
REQS_DIR="${ROOT}/scripts/requirements"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install from https://github.com/astral-sh/uv" >&2
  exit 127
fi

create_env() {
  local name="$1"
  local req="$2"
  local venv_path="${VENVS_DIR}/${name}"
  echo "[uv] creating ${name} ..."
  uv venv "${venv_path}" --python "${PYTHON_VERSION}" --quiet
  uv pip install --python "${venv_path}/bin/python" -r "${req}" --quiet
  echo "[uv] ${name} ready at ${venv_path}"
}

mkdir -p "${VENVS_DIR}"

create_env fe-base          "${REQS_DIR}/fe-base.txt"
create_env fe-jobshop       "${REQS_DIR}/fe-jobshop.txt"
create_env fe-pyportfolioopt "${REQS_DIR}/fe-pyportfolioopt.txt"
create_env fe-optics        "${REQS_DIR}/fe-optics.txt"

echo ""
echo "All uv environments ready under ${VENVS_DIR}/"
echo "Pass task.runtime.python_path=<venvs-dir>/<env>/bin/python to frontier_eval."
