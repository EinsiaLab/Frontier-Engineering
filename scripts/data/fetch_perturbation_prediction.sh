#!/usr/bin/env bash
set -euo pipefail

ROOT="${FRONTIER_ENGINEERING_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_DIR="$ROOT/benchmarks/SingleCellAnalysis/perturbation_prediction/resources_cache/neurips-2023-data"
BASE_URL="https://openproblems-data.s3.amazonaws.com/resources/task_perturbation_prediction/datasets/neurips-2023-data"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$DATA_DIR"

download() {
  local name="$1"
  local url="$BASE_URL/$name"
  local dest="$DATA_DIR/$name"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "$url -> $dest"
  else
    wget -c -O "$dest" "$url"
  fi
}

download de_train.h5ad
download de_test.h5ad
download id_map.csv
