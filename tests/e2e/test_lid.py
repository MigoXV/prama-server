from __future__ import annotations

import logging
import wave
from pathlib import Path

from google.protobuf.json_format import MessageToJson

import grpc
from prama_server.inferencers.grpc_options import create_insecure_channel
from prama_server.protos.lid import lid_pb2, lid_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

TARGET = "192.168.0.222:50026"
WAV_PATH = Path(
    "data-bin/cdb-en-test-wav-audiofolder-test-20260601-065716/test/"
    "2026-05-11T04-30-09-502-000001-7beddbe1-682b-4a21-8d24-1b51eb641804.wav"
)
REQUEST_TIMEOUT_SECONDS = 30.0


def main() -> None:
    pcm, sample_rate, channels, bits_per_sample = _read_wav_pcm(WAV_PATH)

    logger.info(
        "调用 LID 引擎: target=%s wav=%s sample_rate=%s channels=%s bits=%s pcm_bytes=%s",
        TARGET,
        WAV_PATH,
        sample_rate,
        channels,
        bits_per_sample,
        len(pcm),
    )

    with create_insecure_channel(TARGET) as channel:
        grpc.channel_ready_future(channel).result(timeout=REQUEST_TIMEOUT_SECONDS)
        stub = lid_pb2_grpc.LidServiceStub(channel)

        languages = stub.GetLanguages(
            lid_pb2.MsgLidGetLanguagesReq(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        print("GetLanguages raw:")
        print(languages)
        print("GetLanguages json:")
        print(MessageToJson(languages, ensure_ascii=False))

        response = stub.Process(
            lid_pb2.MsgLidProcessReq(
                pcm=pcm,
                sampleRate=sample_rate,
                channel=channels,
                bitsPerSample=bits_per_sample,
            ),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        print("Process raw:")
        print(response)
        print("Process json:")
        print(MessageToJson(response, ensure_ascii=False))


def _read_wav_pcm(wav_path: Path) -> tuple[bytes, int, int, int]:
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV 文件不存在: {wav_path}")

    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        pcm = wav_file.readframes(frame_count)

    return pcm, sample_rate, channels, sample_width * 8


if __name__ == "__main__":
    main()
