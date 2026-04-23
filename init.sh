#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

source "$ROOT/scripts/env/lib_uv_env.sh"

ENV_NAME="${ENV_NAME:-frontier-eval-driver}"
MANIFEST="${MANIFEST:-$ROOT/scripts/env/specs/frontier-eval-driver.json}"
PYTHONNOUSERSITE=1
export PYTHONNOUSERSITE

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
  local driver_python
  driver_python="$(uv_env_python "$ROOT" "$ENV_NAME")"
  "$driver_python" - "$ROOT/.env" "$key" "$val_file" <<'PY'
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

ensure_uv_in_path

echo ""
echo "== Frontier-Eng init: uv driver + framework (env: $ENV_NAME) =="
echo "    This environment runs python -m frontier_eval."
echo "    Task runtimes live under $(uv_envs_dir "$ROOT") — see frontier_eval/README.md."
echo ""

echo "[1/3] Create or update the driver environment"
python3 "$ROOT/scripts/env/ensure_uv_env.py" \
  "$MANIFEST" \
  --root "$ROOT" \
  --envs-dir "$(uv_envs_dir "$ROOT")"

echo ""
echo "[2/3] Check optional system tools"
if command -v octave >/dev/null 2>&1; then
  echo "Octave: found ($(command -v octave))"
else
  echo "Octave: not found"
  echo "  Install it with: bash scripts/bootstrap/install_host_deps.sh --octave"
fi

echo ""
echo "[3/3] Local .env (OpenAI-compatible API; not committed to git)"
if [[ ! -f "$ROOT/.env" && -f "$ROOT/.env.example" ]]; then
  echo "+ cp .env.example .env"
  cp "$ROOT/.env.example" "$ROOT/.env"
  echo "    Created .env from .env.example."
elif [[ -f "$ROOT/.env" ]]; then
  echo "    .env already present — left unchanged."
else
  echo "    No .env.example; skip .env bootstrap."
fi

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

Activate the driver environment:
  source .venvs/$ENV_NAME/bin/activate

Quick smoke (no benchmark-local deps):
  python -m frontier_eval task=smoke algorithm=openevolve algorithm.iterations=0

Per-benchmark setup:
  frontier_eval/README.md

Host dependency bootstrap:
  bash scripts/bootstrap/install_host_deps.sh --octave
  bash scripts/bootstrap/install_host_deps.sh --docker --configure-docker-group

Benchmark / algorithm asset bootstrap:
  python scripts/bootstrap/fetch_task_assets.py --list
  python scripts/bootstrap/fetch_task_assets.py --target v1-baseline-assets
  python scripts/bootstrap/fetch_task_assets.py --target algorithms

Special runtime bootstrap:
  bash scripts/bootstrap/install_openff_dev.sh
EOF
