#!/usr/bin/env bash

ensure_uv_in_path() {
  export PATH="$HOME/.local/bin:$PATH"
  if ! command -v uv >/dev/null 2>&1; then
    cat >&2 <<'EOF'
uv not found.

Install it first:
  curl -LsSf https://astral.sh/uv/install.sh | sh
EOF
    return 127
  fi
}

uv_envs_dir() {
  local root="$1"
  printf '%s\n' "${FRONTIER_EVAL_UV_ENVS_DIR:-$root/.venvs}"
}

uv_env_dir() {
  local root="$1"
  local env_name="$2"
  printf '%s/%s\n' "$(uv_envs_dir "$root")" "$env_name"
}

uv_env_python() {
  local root="$1"
  local env_name="$2"
  printf '%s/bin/python\n' "$(uv_env_dir "$root" "$env_name")"
}

prepend_uv_env_to_path() {
  local root="$1"
  local env_name="$2"
  local env_dir
  env_dir="$(uv_env_dir "$root" "$env_name")"
  export VIRTUAL_ENV="$env_dir"
  export PATH="$env_dir/bin:$PATH"
}
