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

run_help poetry run python -m prama_server.commands.app --help
run_help poetry run python -m prama_server.commands.app serve-http --help
run_help poetry run python -m prama_server.commands.app eval --help
run_help poetry run python -m prama_server.commands.app eval asr --help
run_help poetry run python -m prama_server.commands.app eval vad --help
run_help poetry run python -m prama_server.commands.app eval lid --help
run_help poetry run python -m prama_server.commands.app eval keyword --help
run_help poetry run python -m prama_server.commands.app eval denoise --help
run_help poetry run python -m prama_server.utils.trim_vad_data.app --help
