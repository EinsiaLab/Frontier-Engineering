#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SPECS_DIR="${SPECS_DIR:-$ROOT/scripts/env_specs}"
DRIVER_ENV="${DRIVER_ENV:-frontier-eval-2}"
MAIN_ENV="${MAIN_ENV:-frontier-v1-main}"
SUMMIT_ENV="${SUMMIT_ENV:-frontier-v1-summit}"
SUSTAINDC_ENV="${SUSTAINDC_ENV:-frontier-v1-sustaindc}"
KERNEL_ENV="${KERNEL_ENV:-frontier-v1-kernel}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
VALIDATE_GPU_DEVICES="${VALIDATE_GPU_DEVICES:-0}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found" >&2
  exit 127
fi

CONDA_CMD="conda"
if command -v mamba >/dev/null 2>&1; then
  CONDA_CMD="mamba"
fi

render_spec() {
  local template_file="$1"
  local output_file="$2"
  local escaped_root escaped_python
  escaped_root="$(printf '%s' "$ROOT" | sed -e 's/[&\\]/\\&/g')"
  escaped_python="$(printf '%s' "$PYTHON_VERSION" | sed -e 's/[&\\]/\\&/g')"
  sed \
    -e "s#__ROOT__#${escaped_root}#g" \
    -e "s#__PYTHON_VERSION__#${escaped_python}#g" \
    "$template_file" >"$output_file"
}

upsert_from_spec() {
  local env_name="$1"
  local template_file="$2"
  local rendered_file
  rendered_file="$(mktemp "${TMPDIR:-/tmp}/fe_envspec_${env_name}_XXXXXX.yml")"
  render_spec "$template_file" "$rendered_file"
  if conda env list | awk '!/^#/{print $1}' | grep -qx "$env_name"; then
    echo "[update] ${env_name} from $(basename "$template_file")"
    "${CONDA_CMD}" env update -n "$env_name" -f "$rendered_file" --prune
  else
    echo "[create] ${env_name} from $(basename "$template_file")"
    "${CONDA_CMD}" env create -n "$env_name" -f "$rendered_file"
  fi
  rm -f "$rendered_file"
}

echo "[1/5] build driver env: ${DRIVER_ENV}"
upsert_from_spec "$DRIVER_ENV" "${SPECS_DIR}/frontier-eval-2.yml"

echo "[2/5] build merged main env: ${MAIN_ENV}"
upsert_from_spec "$MAIN_ENV" "${SPECS_DIR}/frontier-v1-main.yml"

echo "[3/5] build merged summit env: ${SUMMIT_ENV}"
upsert_from_spec "$SUMMIT_ENV" "${SPECS_DIR}/frontier-v1-summit.yml"

echo "[4/5] build merged sustaindc env: ${SUSTAINDC_ENV}"
upsert_from_spec "$SUSTAINDC_ENV" "${SPECS_DIR}/frontier-v1-sustaindc.yml"

echo "[5/5] build merged kernel env: ${KERNEL_ENV}"
upsert_from_spec "$KERNEL_ENV" "${SPECS_DIR}/frontier-v1-kernel.yml"

cat <<EOF
Merged task envs are ready:
  ${DRIVER_ENV}
  ${MAIN_ENV}
  ${SUMMIT_ENV}
  ${SUSTAINDC_ENV}
  ${KERNEL_ENV}
EOF

if [[ "${RUN_VALIDATION}" == "1" ]]; then
  echo ""
  echo "[validate] run iteration=0 batch validation"
  DRIVER_ENV="$DRIVER_ENV" GPU_DEVICES="$VALIDATE_GPU_DEVICES" \
    bash "$ROOT/scripts/validate_v1_merged_task_envs.sh"
fi
