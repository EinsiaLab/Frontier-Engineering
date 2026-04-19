#!/usr/bin/env bash
# ------------------------------------------------------------
# Frontier-Eng init script
# ------------------------------------------------------------
# This script bootstraps the conda driver environment and installs
# all required Python packages. It now includes:
#   • Automatic detection of common conda installation paths.
#   • Non-interactive acceptance of the Conda Terms-of-Service.
#   • Export of PYTHONNOUSERSITE=1 to guarantee a clean, isolated
#     Python environment (important for reproducibility).
# ------------------------------------------------------------
set -euo pipefail

# One-shot bootstrap: conda driver env + frontier_eval deps.
# Optional (TTY): prompts to fill .env API fields.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

ENV_NAME="frontier-eval-2"
PYTHON_VERSION="3.12"

# Export isolation flag early
export PYTHONNOUSERSITE=1
prompt_yn() {
  local prompt="$1"
  local def="${2:-n}"
  local hint="y/N"
  [[ "$def" == "y" ]] && hint="Y/n"
  local reply=""
  read -r -p "$prompt [$hint] " reply || true
  reply="${reply:-$def}"
  reply_lc="$(printf '%s' "$reply" | tr '[:upper:]' '[:lower:]')"
  case "$reply_lc" in
    y|yes) return 0 ;;
    *) return 1 ;;
  esac
}

_fe_patch_env_var() {
  local key="$1"
  local val_file="$2"
  conda run -n "$ENV_NAME" python - "$ROOT/.env" "$key" "$val_file" <<'PY'
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
key = sys.argv[2]
val_path = Path(sys.argv[3])
val = val_path.read_text(encoding="utf-8").rstrip("\n\r")
prefix = key + "="
text = env_path.read_text(encoding="utf-8")
lines = text.splitlines()
out: list[str] = []
seen = False
for line in lines:
    if line.startswith(prefix):
        out.append(prefix + val)
        seen = True
    else:
        out.append(line)
if not seen:
    out.append(prefix + val)
body = "\n".join(out)
if text.endswith("\n") or not text:
    env_path.write_text(body + "\n", encoding="utf-8")
else:
    env_path.write_text(body, encoding="utf-8")
PY
}

if ! command -v conda >/dev/null 2>&1; then
  # Try to auto-detect common conda paths
  if [[ -f "$HOME/miniconda3/bin/conda" ]]; then
    export PATH="$HOME/miniconda3/bin:$PATH"
  elif [[ -f "$HOME/anaconda3/bin/conda" ]]; then
    export PATH="$HOME/anaconda3/bin:$PATH"
  elif [[ -f "/opt/conda/bin/conda" ]]; then
    export PATH="/opt/conda/bin:$PATH"
  else
    cat >&2 <<'EOF'
conda not found.

Install Miniconda/Anaconda first and ensure it is in your PATH.
EOF
    exit 127
  fi
fi

# Try to accept Conda ToS for Anaconda defaults if running matching conda versions
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

echo ""
echo "== Frontier-Eng init: conda driver + framework (env: $ENV_NAME) =="
echo "    This env runs python -m frontier_eval. Per-benchmark runtimes are separate — see frontier_eval/README.md."
echo "    Agent skill sources (no installer): .claude/skills/"
echo ""

# 1) Create (or update) the environment.
echo "[1/4] Conda environment (Python $PYTHON_VERSION)"
ENV_EXISTS=0
if conda env list | awk '!/^#/{print $1}' | grep -qx "$ENV_NAME"; then
  ENV_EXISTS=1
fi

if [[ "$ENV_EXISTS" -eq 0 ]]; then
  echo "+ conda create -n $ENV_NAME python=$PYTHON_VERSION -y"
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
else
  echo "+ conda install -n $ENV_NAME python=$PYTHON_VERSION -y"
  conda install -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

echo ""
echo "[2/4] Octave (conda-forge: octave, octave-signal, octave-control — used by some validators, e.g. astrodynamics)"
if [[ "$(uname -m)" == "aarch64" ]]; then
  echo "+ conda install -n $ENV_NAME -c conda-forge octave -y (Skipping signal/control for aarch64 compatibility)"
  conda install -n "$ENV_NAME" -c conda-forge octave -y
else
  echo "+ conda install -n $ENV_NAME -c conda-forge octave octave-signal octave-control -y"
  conda install -n "$ENV_NAME" -c conda-forge octave octave-signal octave-control -y
fi

# 3) Install Python dependencies.
echo ""
echo "[3/4] Python packages (frontier_eval/requirements.txt — Hydra driver, openevolve, torch stack, etc.)"
echo "+ conda run -n $ENV_NAME python -m pip install -U pip"
conda run -n "$ENV_NAME" python -m pip install -U pip
echo "+ conda run -n $ENV_NAME python -m pip install -r frontier_eval/requirements.txt"
conda run -n "$ENV_NAME" python -m pip install -r frontier_eval/requirements.txt

# 4) Create a local .env (ignored by git) for OpenAI-compatible API settings.
echo ""
echo "[4/4] Local .env (OpenAI-compatible API; not committed to git)"
if [[ ! -f "$ROOT/.env" && -f "$ROOT/.env.example" ]]; then
  echo "+ cp .env.example .env"
  cp "$ROOT/.env.example" "$ROOT/.env"
  echo "    Created .env from .env.example — edit for evolution runs (API key / base URL)."
elif [[ -f "$ROOT/.env" ]]; then
  echo "    .env already present — left unchanged."
else
  echo "    No .env.example; skip .env bootstrap."
fi

# 5) Optional .env prompts (skip with n / Enter).
if [[ -t 0 ]] && [[ -t 1 ]]; then
  echo ""
  echo "== Optional: set API keys in .env =="
  echo "    Say 'n' to skip; you can always edit .env manually."
  if [[ -f "$ROOT/.env" ]]; then
    if prompt_yn "Set or update OPENAI_API_KEY in .env now?" "n"; then
      KEYFILE="${TMPDIR:-/tmp}/fe_init_key_$$.tmp"
      read -r -s -p "OPENAI_API_KEY (hidden): " api_key
      echo ""
      if [[ -n "${api_key:-}" ]]; then
        printf '%s' "$api_key" >"$KEYFILE"
        _fe_patch_env_var OPENAI_API_KEY "$KEYFILE"
        rm -f "$KEYFILE"
        echo "Updated OPENAI_API_KEY in .env"
      else
        echo "(empty — skipped)"
      fi
    fi
    if prompt_yn "Set or update OPENAI_API_BASE in .env now?" "n"; then
      read -r -p "OPENAI_API_BASE (e.g. https://api.openai.com/v1): " api_base
      if [[ -n "${api_base:-}" ]]; then
        KEYFILE="${TMPDIR:-/tmp}/fe_init_base_$$.tmp"
        printf '%s' "$api_base" >"$KEYFILE"
        _fe_patch_env_var OPENAI_API_BASE "$KEYFILE"
        rm -f "$KEYFILE"
        echo "Updated OPENAI_API_BASE in .env"
      else
        echo "(empty — skipped)"
      fi
    fi
  fi
fi

cat <<EOF

== Done ==

Driver env:
  conda activate $ENV_NAME

Quick smoke (no benchmark deps):
  python -m frontier_eval algorithm.iterations=0

Evolution (algorithm.iterations > 0) needs OPENAI_API_KEY (and optional OPENAI_API_BASE) in .env.

Per-benchmark setup: frontier_eval/README.md
Agent skill sources (point your agent or copy into your tool): .claude/skills/
EOF
