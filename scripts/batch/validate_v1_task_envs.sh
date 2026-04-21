#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source "$ROOT/scripts/env/lib_uv_env.sh"

DRIVER_PY="${DRIVER_PY:-}"
DRIVER_ENV="${DRIVER_ENV:-frontier-eval-driver}"
V1_MATRIX="${V1_MATRIX:-frontier_eval/conf/batch/v1.yaml}"
GPU_DEVICES="${GPU_DEVICES:-0}"
RUN_BASE_DIR="${RUN_BASE_DIR:-runs/batch_validation}"

if [[ -n "$DRIVER_PY" ]]; then
  DRIVER_CMD=("$DRIVER_PY")
else
  ensure_uv_in_path
  DRIVER_PY="$(uv_env_python "$ROOT" "$DRIVER_ENV")"
  if [[ ! -x "$DRIVER_PY" ]]; then
    echo "driver python not found: $DRIVER_PY" >&2
    echo "Run bash init.sh or bash scripts/env/setup_v1_task_envs.sh first." >&2
    exit 127
  fi
  DRIVER_CMD=("$DRIVER_PY")
fi

run_driver() {
  "${DRIVER_CMD[@]}" "$@"
}

run_cpu_batch() {
  run_driver -m frontier_eval.batch \
    --matrix "$V1_MATRIX" \
    --exclude-tasks Robotics/QuadrupedGaitOptimization \
    --exclude-tasks Robotics/RobotArmCycleTimeOptimization \
    --exclude-tasks Aerodynamics/CarAerodynamicsSensing \
    --exclude-tasks KernelEngineering/MLA \
    --exclude-tasks KernelEngineering/TriMul \
    --exclude-tasks KernelEngineering/FlashAttention \
    --exclude-tasks engdesign \
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

run_kernel_batch() {
  CUDA_VISIBLE_DEVICES="$GPU_DEVICES" \
    run_driver -m frontier_eval.batch \
    --matrix "$V1_MATRIX" \
    --tasks KernelEngineering/MLA \
    --tasks KernelEngineering/TriMul \
    --tasks KernelEngineering/FlashAttention \
    --base-dir "$RUN_BASE_DIR" \
    --override algorithm.iterations=0
}

run_cpu_batch
run_gpu_batch
run_kernel_batch
