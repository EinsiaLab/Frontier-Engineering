#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/env/lib_uv_env.sh"

SPECS_DIR="${SPECS_DIR:-$ROOT/scripts/env/specs}"
RUN_VALIDATION="${RUN_VALIDATION:-0}"

ensure_uv_in_path

build_from_spec() {
  local manifest="$1"
  echo "[build-v2] $(basename "$manifest")"
  python3 "$ROOT/scripts/env/ensure_uv_env.py" \
    "$manifest" \
    --root "$ROOT" \
    --envs-dir "$(uv_envs_dir "$ROOT")"
}

build_from_spec "${SPECS_DIR}/frontier-v2-extra.json"
build_from_spec "${SPECS_DIR}/frontier-v2-summit.json"
build_from_spec "${SPECS_DIR}/frontier-v2-summit-compat.json"
build_from_spec "${SPECS_DIR}/frontier-v2-optics.json"

cat <<EOF
Managed v2 task-set environments live under:
  $(uv_envs_dir "$ROOT")/frontier-v2-extra
  $(uv_envs_dir "$ROOT")/frontier-v2-summit
  $(uv_envs_dir "$ROOT")/frontier-v2-summit-compat
  $(uv_envs_dir "$ROOT")/frontier-v2-optics

Recommended reuse of existing environments without changing their specs:
  .venvs/frontier-v2-extra     -> MaterialEngineering/*, MuonTomography,
                                  PETScannerOptimization, ProtonTherapyPlanning,
                                  perturbation_prediction, CommunicationEngineering v2 tasks
  .venvs/frontier-v2-summit    -> legacy v2 summit runtime
  .venvs/frontier-v2-summit-compat -> ReactionOptimisation/dtlz2_pareto
  .venvs/frontier-v2-optics    -> Optics v2 tasks

Blocked tasks on this server profile:
  SingleCellAnalysis/denoising                 (Docker workflow in task README)
  MolecularMechanics/*                         (openff-dev special runtime, not uv-only)

This script does not modify any v1 setup script or v1 spec.
EOF

if [[ "${RUN_VALIDATION}" == "1" ]]; then
  echo ""
  echo "[note] No automatic validation is run by default for v2."
  echo "[note] Use docs/v2_task_runbook.md for task-specific smoke commands."
fi
