#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Simple one-shot environment bootstrap.
#
# This script creates/updates a conda env and installs all dependencies needed
# for the current minimal workflow:
#   - Python deps: frontier_eval/requirements.txt
#   - Octave + signal/control (used by the MannedLunarLanding validators)
#
# If you want a different env name or Python version, edit the two variables
# below and re-run the script.

ENV_NAME="frontier-eval-2"
PYTHON_VERSION="3.12"

if ! command -v conda >/dev/null 2>&1; then
  cat >&2 <<'EOF'
conda not found.

Install Miniconda/Anaconda first, then rerun:
  bash init.sh
EOF
  exit 127
fi

# 1) Create (or update) the environment.
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

# 2) Install Octave packages used by validators (conda-forge).
echo "+ conda install -n $ENV_NAME -c conda-forge octave octave-signal octave-control -y"
conda install -n "$ENV_NAME" -c conda-forge octave octave-signal octave-control -y

# 3) Install Python dependencies.
echo "+ conda run -n $ENV_NAME python -m pip install -U pip"
conda run -n "$ENV_NAME" python -m pip install -U pip
echo "+ conda run -n $ENV_NAME python -m pip install -r frontier_eval/requirements.txt"
conda run -n "$ENV_NAME" python -m pip install -r frontier_eval/requirements.txt

# 4) Create a local .env (ignored by git) for OpenAI-compatible API settings.
if [[ ! -f "$ROOT/.env" && -f "$ROOT/.env.example" ]]; then
  echo "+ cp .env.example .env"
  cp .env.example .env
fi

cat <<EOF

Done.

Next:
  conda activate $ENV_NAME
  python -m frontier_eval algorithm.iterations=0

If you want to run evolution (algorithm.iterations > 0), fill in .env:
  OPENAI_API_KEY=...
  OPENAI_API_BASE=...   # optional
EOF
