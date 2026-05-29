#!/usr/bin/env bash

set -euo pipefail

VERSION="${VERSION:-0.2.0a4}"
REGISTRY="${REGISTRY:-registry.cn-hangzhou.aliyuncs.com/migo-dl}"
FRONTEND_IMAGE="${FRONTEND_IMAGE:-${REGISTRY}/prama-server-frontend:${VERSION}}"
BACKEND_IMAGE="${BACKEND_IMAGE:-${REGISTRY}/prama-server-backend:${VERSION}}"

docker build \
  -f docker/frontend.dockerfile \
  -t "${FRONTEND_IMAGE}" \
  .

docker build \
  -f docker/backend.dockerfile \
  -t "${BACKEND_IMAGE}" \
  .
