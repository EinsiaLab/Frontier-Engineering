#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DRIVER_PY="${DRIVER_PY:-}"
DRIVER_ENV="${DRIVER_ENV:-frontier-eval-2}"
V1_MATRIX="${V1_MATRIX:-frontier_eval/conf/batch/v1.yaml}"
GPU_DEVICES="${GPU_DEVICES:-0}"
RUN_BASE_DIR="${RUN_BASE_DIR:-runs/batch_validation}"

if [[ -n "$DRIVER_PY" ]]; then
  DRIVER_CMD=("$DRIVER_PY")
else
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found; set DRIVER_PY explicitly or make conda available in PATH" >&2
    exit 127
  fi
  DRIVER_CMD=(conda run -n "$DRIVER_ENV" python)
fi

run_driver() {
  "${DRIVER_CMD[@]}" "$@"
}

run_cpu_batch() {
  run_driver -m frontier_eval.batch \
    --matrix "$V1_MATRIX" \
    --exclude-tasks ParticlePhysics/MuonTomography \
      Robotics/QuadrupedGaitOptimization \
      Robotics/RobotArmCycleTimeOptimization \
      Aerodynamics/CarAerodynamicsSensing \
      KernelEngineering/FlashAttention \
      engdesign \
    --base-dir "$RUN_BASE_DIR" \
    --override algorithm.iterations=0
}

run_gpu_batch() {
  CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
    run_driver -m frontier_eval.batch \
    --matrix "$V1_MATRIX" \
    --tasks Robotics/QuadrupedGaitOptimization \
    --tasks Robotics/RobotArmCycleTimeOptimization \
    --tasks Aerodynamics/CarAerodynamicsSensing \
    --base-dir "$RUN_BASE_DIR" \
    --override algorithm.iterations=0
}

run_flash_batch() {
  CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
    run_driver -m frontier_eval.batch \
    --matrix "$V1_MATRIX" \
    --tasks KernelEngineering/FlashAttention \
    --base-dir "$RUN_BASE_DIR" \
    --override algorithm.iterations=0
}

run_kernel_task() {
  local benchmark="$1"
  CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
    run_driver -m frontier_eval \
    task=unified \
    task.benchmark="$benchmark" \
    task.runtime.conda_env=frontier-v1-kernel \
    algorithm=openevolve \
    algorithm.iterations=0
}

run_cpu_batch
run_gpu_batch
run_flash_batch
run_kernel_task KernelEngineering/MLA
run_kernel_task KernelEngineering/TriMul
