#!/usr/bin/env bash
set -euo pipefail

DEN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${DEN_DIR}/../../.." && pwd)"

TOOLS_DIR="${DEN_DIR}/.tools"
TOOLS_BIN="${TOOLS_DIR}/bin"
JAVA_HOME_LOCAL="${TOOLS_DIR}/jdk-17"
CACHE_DIR="${DEN_DIR}/.cache"
DRIVER_PY="${ROOT}/.venvs/frontier-eval-driver/bin/python"

if [[ -d "${JAVA_HOME_LOCAL}" ]]; then
  export JAVA_HOME="${JAVA_HOME:-${JAVA_HOME_LOCAL}}"
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi

if [[ -d "${TOOLS_BIN}" ]]; then
  export PATH="${TOOLS_BIN}:${PATH}"
fi

export NXF_HOME="${NXF_HOME:-${CACHE_DIR}/nextflow}"
export CAPSULE_DIR="${CAPSULE_DIR:-${CACHE_DIR}/capsule}"
export VIASH_HOME="${VIASH_HOME:-${CACHE_DIR}/viash}"

if [[ -x "${DRIVER_PY}" ]]; then
  export FRONTIER_EVAL_DENOISING_PYTHON="${FRONTIER_EVAL_DENOISING_PYTHON:-${DRIVER_PY}}"
fi
