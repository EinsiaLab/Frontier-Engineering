#!/usr/bin/env bash
set -euo pipefail

STATE_ROOT="${FRONTIER_ENG_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/frontier-eng}"
CACHE_ROOT="${FRONTIER_ENG_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/frontier-eng}"

OCTAVE_ENV_PREFIX="${OCTAVE_ENV_PREFIX:-$STATE_ROOT/octave}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$CACHE_ROOT/conda-pkgs}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$CACHE_ROOT/mamba-root}"

mkdir -p "$CONDA_PKGS_DIRS" "$MAMBA_ROOT_PREFIX"

exec mamba run -p "$OCTAVE_ENV_PREFIX" octave-cli "$@"
