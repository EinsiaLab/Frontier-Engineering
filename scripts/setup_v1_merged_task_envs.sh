#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAIN_ENV="${MAIN_ENV:-frontier-v1-main}"
SUMMIT_ENV="${SUMMIT_ENV:-frontier-v1-summit}"
SUSTAINDC_ENV="${SUSTAINDC_ENV:-frontier-v1-sustaindc}"
KERNEL_ENV="${KERNEL_ENV:-frontier-v1-kernel}"

MAIN_CLONE_SRC="${MAIN_CLONE_SRC:-optics}"
SUMMIT_CLONE_SRC="${SUMMIT_CLONE_SRC:-summit}"
SUSTAINDC_CLONE_SRC="${SUSTAINDC_CLONE_SRC:-sustaindc}"
KERNEL_CLONE_SRC="${KERNEL_CLONE_SRC:-kernel}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found" >&2
  exit 127
fi

ensure_clone() {
  local env_name="$1"
  local clone_src="$2"
  if conda env list | awk '!/^#/{print $1}' | grep -qx "$env_name"; then
    echo "[skip] conda env already exists: $env_name"
    return 0
  fi
  echo "[clone] $clone_src -> $env_name"
  conda create -y -n "$env_name" --clone "$clone_src"
}

pip_install() {
  local env_name="$1"
  shift
  echo "[pip] $env_name :: $*"
  conda run -n "$env_name" python -m pip install "$@"
}

ensure_clone "$MAIN_ENV" "$MAIN_CLONE_SRC"
ensure_clone "$SUMMIT_ENV" "$SUMMIT_CLONE_SRC"
ensure_clone "$SUSTAINDC_ENV" "$SUSTAINDC_CLONE_SRC"
ensure_clone "$KERNEL_ENV" "$KERNEL_CLONE_SRC"

pip_install "$MAIN_ENV" -U pip
pip_install "$MAIN_ENV" anndata mqt.bench stockpyl "job-shop-lib>=1.7,<2"
pip_install "$MAIN_ENV" -r benchmarks/PyPortfolioOpt/requirements.txt
pip_install "$MAIN_ENV" -r benchmarks/Robotics/QuadrupedGaitOptimization/verification/requirements.txt
pip_install "$MAIN_ENV" -r benchmarks/Robotics/RobotArmCycleTimeOptimization/verification/requirements.txt

cat <<EOF
Merged task envs are ready:
  $MAIN_ENV
  $SUMMIT_ENV
  $SUSTAINDC_ENV
  $KERNEL_ENV
EOF
