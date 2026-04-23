#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/bootstrap/install_host_deps.sh [options]

Options:
  --all                    Install all supported host dependencies.
  --octave                 Install GNU Octave via apt.
  --docker                 Install Docker via apt.
  --configure-docker-group Add the current user to the docker group after install.
  --dry-run                Print commands instead of executing them.
  -h, --help               Show this help message.

Notes:
  - This script currently supports Debian/Ubuntu-style systems with `apt-get`.
  - It intentionally handles host tools only. Python environments still go through `uv`.
EOF
}

need_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

require_apt() {
  need_cmd apt-get
  if [[ ! -f /etc/debian_version ]]; then
    echo "This installer currently supports Debian/Ubuntu-style hosts only." >&2
    echo "Install the requested host dependencies manually on your distro." >&2
    exit 2
  fi
}

install_octave() {
  if command -v octave >/dev/null 2>&1; then
    echo "Octave already installed: $(command -v octave)"
    return 0
  fi
  require_apt
  run_cmd sudo apt-get update
  run_cmd sudo apt-get install -y octave
  echo "Installed Octave."
}

install_docker() {
  if command -v docker >/dev/null 2>&1; then
    echo "Docker already installed: $(command -v docker)"
  else
    require_apt
    run_cmd sudo apt-get update
    run_cmd sudo apt-get install -y docker.io
    echo "Installed Docker."
  fi

  if [[ "$CONFIGURE_DOCKER_GROUP" == "1" ]]; then
    run_cmd sudo usermod -aG docker "$USER"
    echo "Added $USER to the docker group. Start a new shell before using Docker without sudo."
  fi
}

INSTALL_OCTAVE=0
INSTALL_DOCKER=0
CONFIGURE_DOCKER_GROUP=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      INSTALL_OCTAVE=1
      INSTALL_DOCKER=1
      ;;
    --octave)
      INSTALL_OCTAVE=1
      ;;
    --docker)
      INSTALL_DOCKER=1
      ;;
    --configure-docker-group)
      CONFIGURE_DOCKER_GROUP=1
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

if [[ "$INSTALL_OCTAVE" == "0" && "$INSTALL_DOCKER" == "0" ]]; then
  usage
  exit 1
fi

if [[ "$INSTALL_OCTAVE" == "1" ]]; then
  install_octave
fi

if [[ "$INSTALL_DOCKER" == "1" ]]; then
  install_docker
fi

echo "Host dependency bootstrap complete."
