#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/bootstrap/install_openff_dev.sh [options]

Options:
  --prefix PATH          Environment prefix. Default: <repo>/.venvs/openff-dev
  --python-version VER   Python version for the env. Default: 3.11
  --installer NAME       Force installer: mamba or conda
  --skip-verify          Skip the post-install verification step
  --dry-run              Print commands without executing them
  -h, --help             Show this help message

Notes:
  - This script intentionally uses mamba/conda for the OpenFF stack.
  - It creates a dedicated runtime at `.venvs/openff-dev` so existing
    `task.runtime.env_name=openff-dev` configs keep working.
EOF
}

PREFIX=""
PYTHON_VERSION="3.11"
INSTALLER="${OPENFF_DEV_INSTALLER:-}"
SKIP_VERIFY=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      PREFIX="$2"
      shift
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift
      ;;
    --installer)
      INSTALLER="$2"
      shift
      ;;
    --skip-verify)
      SKIP_VERIFY=1
      ;;
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

if [[ -z "$PREFIX" ]]; then
  PREFIX="$ROOT/.venvs/openff-dev"
fi
PREFIX="$(python3 - "$PREFIX" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

STATE_ROOT="${FRONTIER_ENG_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/frontier-eng}"
CACHE_ROOT="${FRONTIER_ENG_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/frontier-eng}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CACHE_ROOT/conda-pkgs}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$CACHE_ROOT/mamba-root}"
mkdir -p "$CONDA_PKGS_DIRS" "$MAMBA_ROOT_PREFIX" "$(dirname "$PREFIX")"

quote_cmd() {
  printf '%q' "$1"
  shift || true
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+ '
    quote_cmd "$@"
    return 0
  fi
  "$@"
}

pick_installer() {
  if [[ -n "$INSTALLER" ]]; then
    if ! command -v "$INSTALLER" >/dev/null 2>&1; then
      echo "Requested installer not found: $INSTALLER" >&2
      exit 1
    fi
    echo "$INSTALLER"
    return 0
  fi
  if command -v mamba >/dev/null 2>&1; then
    echo "mamba"
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    echo "conda"
    return 0
  fi
  cat >&2 <<'EOF'
Neither `mamba` nor `conda` was found.

Install one of them first, then rerun:
  bash scripts/bootstrap/install_openff_dev.sh
EOF
  exit 1
}

INSTALLER_BIN="$(pick_installer)"
ENV_PYTHON="$PREFIX/bin/python"
COMMON_PACKAGES=(
  "python=$PYTHON_VERSION"
  numpy
  scipy
  rdkit
  openmm
  ambertools
  openff-toolkit
)
CHANNEL_ARGS=(
  --override-channels
  --strict-channel-priority
  -c conda-forge
)

if [[ -d "$PREFIX/conda-meta" ]]; then
  echo "[openff-dev] Updating existing environment at $PREFIX using $INSTALLER_BIN"
  run_cmd "$INSTALLER_BIN" install -y -p "$PREFIX" "${CHANNEL_ARGS[@]}" "${COMMON_PACKAGES[@]}"
else
  echo "[openff-dev] Creating environment at $PREFIX using $INSTALLER_BIN"
  run_cmd "$INSTALLER_BIN" create -y -p "$PREFIX" "${CHANNEL_ARGS[@]}" "${COMMON_PACKAGES[@]}"
fi

echo "[openff-dev] Installing benchmark-local Python requirements"
run_cmd "$ENV_PYTHON" -m pip install -r "$ROOT/benchmarks/MolecularMechanics/requirements.txt"

if [[ "$SKIP_VERIFY" != "1" ]]; then
  echo "[openff-dev] Running verification"
  run_cmd "$ENV_PYTHON" "$ROOT/scripts/bootstrap/verify_openff_dev.py" --repo-root "$ROOT"
fi

cat <<EOF

openff-dev is ready at:
  $PREFIX

You can use it directly with:
  $ENV_PYTHON -m pip list

And Frontier Eval can now resolve:
  task.runtime.env_name=openff-dev
EOF
