#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${UV_ENV_ROOT:-$ROOT/.uv-envs}"
UV_BIN="${UV_BIN:-}"

resolve_uv() {
  if [[ -n "$UV_BIN" && -x "$UV_BIN" ]]; then
    return 0
  fi

  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return 0
  fi

  for candidate in "$HOME/.local/bin/uv" "/root/.local/bin/uv" "$HOME/bin/uv" "/usr/local/bin/uv"; do
    if [[ -x "$candidate" ]]; then
      UV_BIN="$candidate"
      return 0
    fi
  done

  return 1
}

need_uv() {
  if ! resolve_uv; then
    cat >&2 <<'EOF'
uv not found.

Install uv first, then rerun:
  curl -LsSf https://astral.sh/uv/install.sh | sh

If uv was installed into ~/.local/bin, make sure that directory is in PATH,
or rerun with:
  UV_BIN=$HOME/.local/bin/uv bash scripts/setup_uv_envs.sh ...
EOF
  exit 127
  fi
}

ensure_env_dir() {
  mkdir -p "$ENV_ROOT"
}

venv_python() {
  printf '%s/bin/python' "$1"
}

create_venv() {
  local env_path="$1"
  local py_ver="$2"
  if [[ -x "$(venv_python "$env_path")" ]]; then
    echo "==> reuse venv: $env_path"
    return 0
  fi
  echo "==> create venv: $env_path (python $py_ver)"
  "$UV_BIN" venv --python "$py_ver" "$env_path"
}

sync_project_driver() {
  local env_path="$1"
  echo "==> sync project driver into $env_path"
  UV_PROJECT_ENVIRONMENT="$env_path" "$UV_BIN" sync --project "$ROOT" --group driver
}

pip_install_reqs() {
  local env_path="$1"
  shift
  local py
  py="$(venv_python "$env_path")"
  for req in "$@"; do
    echo "==> install requirements: $req"
    "$UV_BIN" pip install --python "$py" -r "$ROOT/$req"
  done
}

pip_install_pkgs() {
  local env_path="$1"
  shift
  local py
  py="$(venv_python "$env_path")"
  if [[ "$#" -gt 0 ]]; then
    echo "==> install packages: $*"
    "$UV_BIN" pip install --python "$py" "$@"
  fi
}

setup_driver() {
  local env_path="$ENV_ROOT/driver"
  create_venv "$env_path" "3.12"
  sync_project_driver "$env_path"
}

setup_v1_general() {
  local env_path="$ENV_ROOT/v1-general"
  create_venv "$env_path" "3.12"
  sync_project_driver "$env_path"
  pip_install_reqs "$env_path" \
    "benchmarks/CommunicationEngineering/LDPCErrorFloor/verification/requirements.txt" \
    "benchmarks/CommunicationEngineering/PMDSimulation/verification/requirements.txt" \
    "benchmarks/CommunicationEngineering/RayleighFadingBER/verification/requirements.txt" \
    "benchmarks/EnergyStorage/BatteryFastChargingProfile/verification/requirements.txt" \
    "benchmarks/EnergyStorage/BatteryFastChargingSPMe/verification/requirements.txt" \
    "benchmarks/ParticlePhysics/MuonTomography/verification/requirements.txt" \
    "benchmarks/ParticlePhysics/ProtonTherapyPlanning/verification/requirements.txt" \
    "benchmarks/ComputerSystems/DuckDBWorkloadOptimization/verification/requirements.txt" \
    "benchmarks/SingleCellAnalysis/predict_modality/verification/requirements-predict_modality.txt" \
    "benchmarks/SingleCellAnalysis/perturbation_prediction/verification/requirements-perturbation_prediction.txt" \
    "benchmarks/StructuralOptimization/ISCSO2015/verification/requirements.txt" \
    "benchmarks/StructuralOptimization/ISCSO2023/verification/requirements.txt" \
    "benchmarks/StructuralOptimization/PyMOTOSIMPCompliance/verification/requirements.txt" \
    "benchmarks/StructuralOptimization/TopologyOptimization/verification/requirements.txt" \
    "benchmarks/WirelessChannelSimulation/HighReliableSimulation/verification/requirements.txt" \
    "benchmarks/Robotics/DynamicObstacleAvoidanceNavigation/verification/requirements.txt" \
    "benchmarks/Robotics/PIDTuning/verification/requirements.txt" \
    "benchmarks/Robotics/UAVInspectionCoverageWithWind/verification/requirements.txt" \
    "benchmarks/Robotics/CoFlyersVasarhelyiTuning/verification/requirements.txt" \
    "benchmarks/Aerodynamics/DawnAircraftDesignOptimization/verification/requirements.txt" \
    "benchmarks/PyPortfolioOpt/requirements.txt" \
    "benchmarks/JobShop/requirements.txt"
  pip_install_pkgs "$env_path" \
    "mqt.bench" \
    "stockpyl" \
    "job-shop-lib"
}

setup_v1_optics() {
  local env_path="$ENV_ROOT/v1-optics"
  create_venv "$env_path" "3.12"
  sync_project_driver "$env_path"
  pip_install_reqs "$env_path" "benchmarks/Optics/requirements.txt"
  cat <<'EOF'
NOTE: v1-optics may still require system OpenGL libraries on some machines.
If OpenCV import fails with libGL.so.1, install the missing system package first.
EOF
}

setup_v1_gpu() {
  local env_path="$ENV_ROOT/v1-gpu"
  create_venv "$env_path" "3.11"
  pip_install_reqs "$env_path" \
    "benchmarks/Aerodynamics/CarAerodynamicsSensing/verification/requirements.txt" \
    "benchmarks/Robotics/QuadrupedGaitOptimization/verification/requirements.txt" \
    "benchmarks/Robotics/RobotArmCycleTimeOptimization/verification/requirements.txt"
}

setup_v1_power() {
  local env_path="$ENV_ROOT/v1-power"
  create_venv "$env_path" "3.12"
  sync_project_driver "$env_path"
  pip_install_reqs "$env_path" "benchmarks/PowerSystems/EV2GymSmartCharging/verification/requirements.txt"
  pip_install_pkgs "$env_path" "setuptools<81"
}

setup_v1_summit() {
  local env_path="$ENV_ROOT/v1-summit"
  create_venv "$env_path" "3.9"
  pip_install_reqs "$env_path" "benchmarks/ReactionOptimisation/requirements.txt"
  pip_install_pkgs "$env_path" "setuptools<81"
}

setup_v1_sustaindc() {
  local env_path="$ENV_ROOT/v1-sustaindc"
  create_venv "$env_path" "3.10"
  pip_install_reqs "$env_path" "benchmarks/SustainableDataCenterControl/requirements.txt"
}

setup_v1_kernel() {
  local env_path="$ENV_ROOT/v1-kernel"
  create_venv "$env_path" "3.12"
  pip_install_reqs "$env_path" \
    "benchmarks/KernelEngineering/FlashAttention/verification/requirements-gpumode.txt" \
    "benchmarks/KernelEngineering/MLA/verification/requirements-gpumode.txt" \
    "benchmarks/KernelEngineering/TriMul/verification/requirements-gpumode.txt"
}

setup_v1_openff() {
  local env_path="$ENV_ROOT/v1-openff"
  create_venv "$env_path" "3.11"
  pip_install_reqs "$env_path" "benchmarks/MolecularMechanics/requirements.txt"
  cat <<'EOF'
NOTE: v1-openff is only the Python layer.
You still need binary-heavy packages such as rdkit, openmm, and ambertools from conda-forge or another system package source.
EOF
}

setup_v1_diffsim() {
  local env_path="$ENV_ROOT/v1-diffsim"
  create_venv "$env_path" "3.12"
  sync_project_driver "$env_path"
  pip_install_reqs "$env_path" "benchmarks/AdditiveManufacturing/DiffSimThermalControl/verification/requirements.txt"
}

setup_v1_singlecell_denoising() {
  local env_path="$ENV_ROOT/v1-singlecell-denoising"
  create_venv "$env_path" "3.12"
  pip_install_reqs "$env_path" "benchmarks/SingleCellAnalysis/denoising_ttt/verification/requirements-denoising.txt"
  cat <<'EOF'
NOTE: this env covers the Python package side of denoising_ttt.
The original denoising benchmark still follows the upstream build / Docker workflow documented in benchmarks/SingleCellAnalysis/denoising/README_zh-CN.md.
EOF
}

show_help() {
  cat <<EOF
Usage:
  bash scripts/setup_uv_envs.sh <target> [<target> ...]

Targets:
  driver
  v1-general
  v1-optics
  v1-gpu
  v1-power
  v1-summit
  v1-sustaindc
  v1-kernel
  v1-openff
  v1-diffsim
  v1-singlecell-denoising
  all

Environment root:
  $ENV_ROOT
EOF
}

main() {
  need_uv
  ensure_env_dir

  if [[ "$#" -eq 0 ]]; then
    show_help
    exit 1
  fi

  for target in "$@"; do
    case "$target" in
      driver) setup_driver ;;
      v1-general) setup_v1_general ;;
      v1-optics) setup_v1_optics ;;
      v1-gpu) setup_v1_gpu ;;
      v1-power) setup_v1_power ;;
      v1-summit) setup_v1_summit ;;
      v1-sustaindc) setup_v1_sustaindc ;;
      v1-kernel) setup_v1_kernel ;;
      v1-openff) setup_v1_openff ;;
      v1-diffsim) setup_v1_diffsim ;;
      v1-singlecell-denoising) setup_v1_singlecell_denoising ;;
      all)
        setup_driver
        setup_v1_general
        setup_v1_optics
        setup_v1_gpu
        setup_v1_power
        setup_v1_summit
        setup_v1_sustaindc
        setup_v1_kernel
        setup_v1_openff
        setup_v1_diffsim
        setup_v1_singlecell_denoising
        ;;
      -h|--help|help)
        show_help
        ;;
      *)
        echo "Unknown target: $target" >&2
        show_help >&2
        exit 2
        ;;
    esac
  done
}

main "$@"
