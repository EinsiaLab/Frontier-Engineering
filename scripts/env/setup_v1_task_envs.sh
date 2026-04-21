#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/env/lib_uv_env.sh"

SPECS_DIR="${SPECS_DIR:-$ROOT/scripts/env/specs}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
VALIDATE_GPU_DEVICES="${VALIDATE_GPU_DEVICES:-0}"

ensure_uv_in_path

build_from_spec() {
  local manifest="$1"
  echo "[build] $(basename "$manifest")"
  if ! python3 "$ROOT/scripts/env/ensure_uv_env.py" \
    "$manifest" \
    --root "$ROOT" \
    --envs-dir "$(uv_envs_dir "$ROOT")"; then
    echo "[WARN] Failed to build $(basename "$manifest"), continuing..."
  fi
}

build_from_spec "${SPECS_DIR}/frontier-eval-driver.json"
build_from_spec "${SPECS_DIR}/frontier-v1-main.json"
build_from_spec "${SPECS_DIR}/frontier-v1-summit.json"
build_from_spec "${SPECS_DIR}/frontier-v1-sustaindc.json"
build_from_spec "${SPECS_DIR}/frontier-v1-kernel.json"
build_from_spec "${SPECS_DIR}/openff-dev.json"

cat <<EOF
Managed uv environments live under:
  $(uv_envs_dir "$ROOT")

Expected environment names:
  frontier-eval-driver
  frontier-v1-main
  frontier-v1-summit
  frontier-v1-sustaindc
  frontier-v1-kernel
  openff-dev (bootstrap separately with scripts/bootstrap/install_openff_dev.sh)
EOF

if [[ "${RUN_VALIDATION}" == "1" ]]; then
  echo ""
  echo "[validate] run iteration=0 batch validation"
  GPU_DEVICES="$VALIDATE_GPU_DEVICES" \
    bash "$ROOT/scripts/batch/validate_v1_task_envs.sh"
fi
