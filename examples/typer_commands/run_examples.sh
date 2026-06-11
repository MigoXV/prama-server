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
DENOISE_TARGET="${DENOISE_TARGET:-192.168.0.222:50027}"
MOS_TARGET="${MOS_TARGET:-}"
SNR_TARGET="${SNR_TARGET:-}"

run_or_print() {
  echo
  echo ">>> $*"
  if [[ "${RUN_LIVE}" == "1" ]]; then
    "$@"
  fi
}

if [[ "${RUN_LIVE}" != "1" ]]; then
  echo "RUN_LIVE is not 1; commands will be printed only."
fi

run_or_print poetry run python -m prama_server.utils.trim_vad_data.app \
  --dataset-path data-bin/audiofolder/vad-demo \
  --split test \
  --output "${OUTPUT_DIR}/vad-trim-demo" \
  --chunk-seconds 10 \
  --overlap-seconds 0 \
  --sample-rate 16000 \
  --overwrite

run_or_print poetry run python -m prama_server.commands.app eval asr \
  --target "${ASR_TARGET}" \
  --dataset-path data-bin/audiofolder/asr-demo \
  --split test \
  --limit 2 \
  --sample-rate 16000 \
  --language-code en-US \
  --min-reference-words 0 \
  --output "${OUTPUT_DIR}/asr.tsv"

run_or_print poetry run python -m prama_server.commands.app eval vad \
  --target "${VAD_TARGET}" \
  --dataset-path data-bin/audiofolder/vad-demo \
  --split test \
  --limit 2 \
  --sample-rate 16000 \
  --mask-frame-seconds 0.01 \
  --hit-threshold 0.9 \
  --output "${OUTPUT_DIR}/vad.tsv"

run_or_print poetry run python -m prama_server.commands.app eval lid \
  --target "${LID_TARGET}" \
  --dataset-path data-bin/audiofolder/lid-demo \
  --split test \
  --limit 2 \
  --sample-rate 16000 \
  --lid-confidence-threshold 0 \
  --output "${OUTPUT_DIR}/lid.tsv"

run_or_print poetry run python -m prama_server.commands.app eval keyword \
  --target "${ASR_TARGET}" \
  --dataset-path data-bin/audiofolder/keyword-demo \
  --split test \
  --limit 2 \
  --sample-rate 16000 \
  --language-code en-US \
  --output "${OUTPUT_DIR}/keyword.tsv"

if [[ -n "${MOS_TARGET}" || -n "${SNR_TARGET}" ]]; then
  denoise_cmd=(
    poetry run python -m prama_server.commands.app eval denoise
    --target "${DENOISE_TARGET}"
    --dataset-path data-bin/audiofolder/denoise-demo
    --split test
    --limit 2
    --sample-rate 16000
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
  echo ">>> skip denoise example because MOS_TARGET and SNR_TARGET are empty"
fi
