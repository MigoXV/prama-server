#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if command -v poetry >/dev/null 2>&1 && [[ -f "pyproject.toml" ]]; then
  PYTHON=(poetry run python)
else
  PYTHON=(python)
fi

python_pkg_version() {
  "${PYTHON[@]}" - "$1" <<'PY'
from __future__ import annotations

import importlib.metadata
import sys

try:
    print(importlib.metadata.version(sys.argv[1]))
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(1)
PY
}

ensure_package() {
  local package_name="$1"
  local install_spec="${2:-$1}"

  if ! python_pkg_version "${package_name}" >/dev/null 2>&1; then
    "${PYTHON[@]}" -m pip install "${install_spec}"
  fi
}

ensure_package grpcio
GRPCIO_VERSION="$(python_pkg_version grpcio)"

if ! GRPCIO_TOOLS_VERSION="$(python_pkg_version grpcio-tools 2>/dev/null)"; then
  "${PYTHON[@]}" -m pip install "grpcio-tools==${GRPCIO_VERSION}"
elif [[ "${GRPCIO_TOOLS_VERSION}" != "${GRPCIO_VERSION}" ]]; then
  "${PYTHON[@]}" -m pip install "grpcio-tools==${GRPCIO_VERSION}"
fi

ensure_package mypy-protobuf

PROTO_ROOT="${PROTO_ROOT:-src}"
PROTO_DIR="${PROTO_DIR:-src/prama_server/protos}"
OUT_DIR="${OUT_DIR:-src}"

mkdir -p "${OUT_DIR}"
mapfile -t PROTO_FILES < <(find "${PROTO_DIR}" -type f -name '*.proto' | sort)

if [[ "${#PROTO_FILES[@]}" -eq 0 ]]; then
  echo "No proto files found under ${PROTO_DIR}" >&2
  exit 1
fi

"${PYTHON[@]}" -m grpc_tools.protoc \
  -I "${PROTO_ROOT}" \
  --python_out="${OUT_DIR}" \
  --grpc_python_out="${OUT_DIR}" \
  --mypy_out="${OUT_DIR}" \
  --mypy_grpc_out="${OUT_DIR}" \
  "${PROTO_FILES[@]}"
