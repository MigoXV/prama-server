from __future__ import annotations

import logging
from pathlib import Path

import soundfile as sf

from prama_server.inferencers.asr import AsrGrpcInferencer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    audio_path = Path("data-bin/long/audio.wav")
    target = "192.168.0.213:50003"
    audio_len_max = 60.0

    logger.info("读取音频: %s", audio_path)
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    audio = audio[: int(audio_len_max * sample_rate)]

    logger.info("调用 ASR 推理器: target=%s sample_rate=%s", target, sample_rate)
    with AsrGrpcInferencer(
        target=target,
        sample_rate=sample_rate,
    ) as inferencer:
        for transcript, is_final in inferencer.infer(audio):
            if is_final:
                print("\n[Final] ", transcript, flush=True)


if __name__ == "__main__":
    main()
