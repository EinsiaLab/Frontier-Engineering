#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BENCHMARK_DIR="$ROOT/benchmarks/SingleCellAnalysis/denoising"
TASK_DIR="$BENCHMARK_DIR/task_denoising"
TOOLS_DIR="$BENCHMARK_DIR/.tools"
TOOLS_BIN="$TOOLS_DIR/bin"
JAVA_HOME_LOCAL="$TOOLS_DIR/jdk-17"
CACHE_DIR="$BENCHMARK_DIR/.cache"
NXF_HOME_LOCAL="$CACHE_DIR/nextflow"
CAPSULE_DIR_LOCAL="$CACHE_DIR/capsule"
VIASH_HOME_LOCAL="$CACHE_DIR/viash"
DRIVER_PY="$ROOT/.venvs/frontier-eval-driver/bin/python"

REPO_URL="${DENOISING_REPO_URL:-https://github.com/openproblems-bio/task_denoising.git}"
JDK_URL="${DENOISING_JDK_URL:-https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse?project=jdk}"
VIASH_RELEASE_BASE_URL="${DENOISING_VIASH_RELEASE_BASE_URL:-https://github.com/viash-io/viash/releases/download}"
SYNC_RESOURCES=0
BUILD_COMPONENTS=0
BUILD_CONTAINERS=0
SMOKE=0

usage() {
  cat <<EOF
Usage: bash scripts/bootstrap/setup_denoising_task.sh [options]

Bootstraps repo-local prerequisites for benchmarks/SingleCellAnalysis/denoising.

Options:
  --sync-resources    Run task_denoising/scripts/sync_resources.sh after setup.
  --build-components  Run task_denoising/scripts/project/build_all_components.sh.
  --build-containers  Run task_denoising/scripts/project/build_all_docker_containers.sh.
  --smoke             Run task_denoising/scripts/run_benchmark/run_test_local.sh.
  --help              Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sync-resources)
      SYNC_RESOURCES=1
      ;;
    --build-components)
      BUILD_COMPONENTS=1
      ;;
    --build-containers)
      BUILD_CONTAINERS=1
      ;;
    --smoke)
      SMOKE=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

mkdir -p "$TOOLS_BIN" "$CACHE_DIR" "$NXF_HOME_LOCAL" "$CAPSULE_DIR_LOCAL" "$VIASH_HOME_LOCAL"

require_cmd() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Missing required command: $name" >&2
    exit 1
  fi
}

require_cmd git
require_cmd curl
require_cmd tar

java_major_version() {
  local java_bin="$1"
  local version_line raw major
  version_line="$("$java_bin" -version 2>&1 | head -n 1)"
  raw="$(printf '%s' "$version_line" | sed -E 's/.*version "([^"]+)".*/\1/')"
  major="${raw%%.*}"
  if [[ "$major" == "1" ]]; then
    major="$(printf '%s' "$raw" | cut -d. -f2)"
  fi
  printf '%s\n' "$major"
}

use_local_java=0
if command -v java >/dev/null 2>&1; then
  system_java_major="$(java_major_version java || true)"
  if [[ -n "${system_java_major:-}" ]] && [[ "$system_java_major" =~ ^[0-9]+$ ]] && (( system_java_major >= 17 )); then
    echo "[java] host Java is already >= 17: $(java -version 2>&1 | head -n 1)"
  else
    use_local_java=1
  fi
else
  use_local_java=1
fi

if (( use_local_java == 1 )); then
  echo "[java] install local JDK 17 into $JAVA_HOME_LOCAL"
  tmp_archive="$(mktemp /tmp/denoising-jdk17-XXXXXX.tar.gz)"
  rm -rf "$JAVA_HOME_LOCAL"
  mkdir -p "$JAVA_HOME_LOCAL"
  curl -fsSL "$JDK_URL" -o "$tmp_archive"
  tar -xzf "$tmp_archive" --strip-components=1 -C "$JAVA_HOME_LOCAL"
  rm -f "$tmp_archive"
fi

if [[ -x "$JAVA_HOME_LOCAL/bin/java" ]]; then
  export JAVA_HOME="$JAVA_HOME_LOCAL"
  export PATH="$JAVA_HOME/bin:$PATH"
fi

install_viash() {
  if [[ -x "$TOOLS_BIN/viash" ]]; then
    echo "[viash] already present: $("$TOOLS_BIN/viash" --version 2>/dev/null | head -n 1 || true)"
    return
  fi
  echo "[viash] install local binary into $TOOLS_BIN"
  local installer
  installer="$(mktemp /tmp/denoising-viash-installer-XXXXXX.sh)"
  curl -fsSL "https://dl.viash.io" -o "$installer"
  (
    cd "$TOOLS_BIN"
    bash "$installer"
  )
  rm -f "$installer"
  chmod +x "$TOOLS_BIN/viash"
}

install_nextflow() {
  if [[ -x "$TOOLS_BIN/nextflow" ]]; then
    echo "[nextflow] already present: $("$TOOLS_BIN/nextflow" -version 2>/dev/null | head -n 1 || true)"
    return
  fi
  echo "[nextflow] install local launcher into $TOOLS_BIN"
  curl -fsSL "https://get.nextflow.io" -o "$TOOLS_BIN/nextflow"
  chmod +x "$TOOLS_BIN/nextflow"
  export CAPSULE_LOG=none
  export NXF_HOME="$NXF_HOME_LOCAL"
  "$TOOLS_BIN/nextflow" -version >/dev/null
}

install_viash
install_nextflow

export PATH="$TOOLS_BIN:$PATH"
export NXF_HOME="$NXF_HOME_LOCAL"
export CAPSULE_DIR="$CAPSULE_DIR_LOCAL"
export VIASH_HOME="$VIASH_HOME_LOCAL"

echo "[tooling] viash: $("$TOOLS_BIN/viash" --version 2>/dev/null | head -n 1 || true)"
echo "[tooling] nextflow: $("$TOOLS_BIN/nextflow" -version 2>/dev/null | head -n 1 || true)"

if [[ ! -d "$TASK_DIR/.git" ]]; then
  echo "[repo] clone $REPO_URL -> $TASK_DIR"
  git clone --recurse-submodules "$REPO_URL" "$TASK_DIR"
else
  echo "[repo] reuse existing checkout at $TASK_DIR"
  git -C "$TASK_DIR" submodule update --init --recursive
fi

ensure_pinned_viash_release() {
  local viash_yaml="$TASK_DIR/_viash.yaml"
  local pinned_version=""
  local pinned_bin=""
  if [[ ! -f "$viash_yaml" ]]; then
    return
  fi
  pinned_version="$(sed -n -E 's/^viash_version:[[:space:]]*([0-9.]+)[[:space:]]*$/\1/p' "$viash_yaml" | head -n 1)"
  if [[ -z "$pinned_version" ]]; then
    return
  fi
  pinned_bin="$VIASH_HOME_LOCAL/releases/$pinned_version/viash"
  if [[ -x "$pinned_bin" ]] && [[ -s "$pinned_bin" ]]; then
    echo "[viash] pinned runtime already cached: $pinned_version"
    return
  fi
  echo "[viash] prefetch pinned runtime $pinned_version"
  mkdir -p "$(dirname "$pinned_bin")"
  curl -fsSL "$VIASH_RELEASE_BASE_URL/$pinned_version/viash" -o "$pinned_bin"
  chmod +x "$pinned_bin"
}

ensure_pinned_viash_release

mkdir -p "$TASK_DIR/src/methods/submission"
if [[ ! -f "$TASK_DIR/src/methods/submission/config.vsh.yaml" ]]; then
  cp "$BENCHMARK_DIR/submission_template/method_submission/config.vsh.yaml" \
    "$TASK_DIR/src/methods/submission/config.vsh.yaml"
fi
if [[ ! -f "$TASK_DIR/src/methods/submission/script.py" ]]; then
  cp "$BENCHMARK_DIR/submission_template/method_submission/script.py" \
    "$TASK_DIR/src/methods/submission/script.py"
fi

apply_patch_if_needed() {
  local patch_path="$1"
  if git -C "$TASK_DIR" apply --check "$patch_path" >/dev/null 2>&1; then
    git -C "$TASK_DIR" apply "$patch_path"
    echo "[patch] applied $(basename "$patch_path")"
    return
  fi
  if git -C "$TASK_DIR" apply --reverse --check "$patch_path" >/dev/null 2>&1; then
    echo "[patch] already applied $(basename "$patch_path")"
    return
  fi
  echo "[patch] unable to apply cleanly: $patch_path" >&2
  exit 1
}

apply_patch_if_needed "$BENCHMARK_DIR/submission_template/patches/run_benchmark_main.nf.patch"
apply_patch_if_needed "$BENCHMARK_DIR/submission_template/patches/run_benchmark_config.vsh.yaml.patch"
apply_patch_if_needed "$BENCHMARK_DIR/submission_template/patches/python310_compat.patch"

if [[ -x "$DRIVER_PY" ]]; then
  export FRONTIER_EVAL_DENOISING_PYTHON="$DRIVER_PY"
fi

if (( SYNC_RESOURCES == 1 )); then
  echo "[resources] sync benchmark resources"
  (cd "$TASK_DIR" && bash scripts/sync_resources.sh)
fi

if (( BUILD_COMPONENTS == 1 )); then
  echo "[build] build all components"
  (cd "$TASK_DIR" && bash scripts/project/build_all_components.sh)
fi

if (( BUILD_CONTAINERS == 1 )); then
  echo "[build] build all docker containers"
  (cd "$TASK_DIR" && bash scripts/project/build_all_docker_containers.sh)
fi

if (( SMOKE == 1 )); then
  echo "[smoke] run local benchmark test"
  (cd "$TASK_DIR" && bash scripts/run_benchmark/run_test_local.sh)
fi

cat <<EOF

[ready] denoising bootstrap complete
  benchmark: $BENCHMARK_DIR
  task repo:  $TASK_DIR
  tools bin:  $TOOLS_BIN
  nxf home:   $NXF_HOME_LOCAL

Use in a new shell:
  source benchmarks/SingleCellAnalysis/denoising/env.sh

Then optional manual steps:
  cd benchmarks/SingleCellAnalysis/denoising/task_denoising
  viash ns build --parallel --setup cachedbuild --query '^(methods/submission|workflows/run_benchmark)$'
  bash scripts/run_benchmark/run_test_local.sh
EOF
