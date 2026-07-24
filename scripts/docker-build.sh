#!/usr/bin/env bash

set -euo pipefail

VERSION="${VERSION:-0.8.0a2}"
REGISTRY="${REGISTRY:-registry.cn-hangzhou.aliyuncs.com/migo-dl}"
IMAGE="${IMAGE:-${REGISTRY}/prama-server:${VERSION}}"

docker build -t "${IMAGE}" .

docker push "${IMAGE}"
