#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUN_LIVE="${RUN_LIVE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/typer-command-examples}"
ASR_TARGET="${ASR_TARGET:-192.168.0.222:50011}"
VAD_TARGET="${VAD_TARGET:-192.168.0.222:50021}"
LID_TARGET="${LID_TARGET:-192.168.0.222:50026}"
DENOISE_TARGET="${DENOISE_TARGET:-192.168.0.222:50031}"
MOS_TARGET="${MOS_TARGET:-}"
SNR_TARGET="${SNR_TARGET:-}"

run_or_print() {
  echo
  echo ">>> $*"
  if [[ "${RUN_LIVE}" == "1" ]]; then
    "$@"
  fi
}

run_or_print poetry run python -m prama_server.utils.trim.app \
  --dataset-path data-bin/raw-vad \
  --split test \
  --output "${OUTPUT_DIR}/vad-trim-demo" \
  --chunk-seconds 10 \
  --sample-rate 16000 \
  --overwrite

run_or_print poetry run prama-server eval asr \
  --target "${ASR_TARGET}" \
  --dataset-path data-bin/audiofolder/asr-demo \
  --limit 2 \
  --min-reference-words 0 \
  --output "${OUTPUT_DIR}/asr.tsv"

run_or_print poetry run prama-server eval vad \
  --target "${VAD_TARGET}" \
  --dataset-path data-bin/audiofolder/vad-demo \
  --limit 2 \
  --output "${OUTPUT_DIR}/vad.tsv"

run_or_print poetry run prama-server eval lid \
  --target "${LID_TARGET}" \
  --dataset-path data-bin/audiofolder/lid-demo \
  --limit 2 \
  --output "${OUTPUT_DIR}/lid.tsv"

run_or_print poetry run prama-server eval keyword \
  --target "${ASR_TARGET}" \
  --dataset-path data-bin/audiofolder/keyword-demo \
  --limit 2 \
  --output "${OUTPUT_DIR}/keyword.tsv"

if [[ -n "${MOS_TARGET}" || -n "${SNR_TARGET}" ]]; then
  denoise_cmd=(
    poetry run prama-server eval denoise
    --target "${DENOISE_TARGET}"
    --dataset-path data-bin/audiofolder/denoise-demo
    --limit 2
    --output "${OUTPUT_DIR}/denoise.tsv"
  )
  if [[ -n "${MOS_TARGET}" ]]; then
    denoise_cmd+=(--mos-target "${MOS_TARGET}")
  fi
  if [[ -n "${SNR_TARGET}" ]]; then
    denoise_cmd+=(--snr-target "${SNR_TARGET}")
  fi
  run_or_print "${denoise_cmd[@]}"
else
  echo
  echo ">>> 未配置 MOS_TARGET/SNR_TARGET，跳过 SE 示例"
fi
