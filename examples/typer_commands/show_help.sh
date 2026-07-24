#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

run_help() {
  echo
  echo ">>> $*"
  "$@"
}

run_help poetry run prama-server --help
run_help poetry run prama-server serve-http --help
run_help poetry run prama-server eval --help
run_help poetry run prama-server eval asr --help
run_help poetry run prama-server eval vad --help
run_help poetry run prama-server eval lid --help
run_help poetry run prama-server eval keyword --help
run_help poetry run prama-server eval denoise --help
run_help poetry run python -m prama_server.utils.trim.app --help
run_help poetry run python -m prama_server.utils.vad_select.app --help
