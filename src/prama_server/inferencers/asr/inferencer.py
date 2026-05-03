from __future__ import annotations

import logging
import string
from collections.abc import Iterable, Iterator

import grpc
import numpy as np

from prama_server.inferencers.base import Inferencer
from prama_server.protos.asr import ux_speech_pb2, ux_speech_pb2_grpc

logger = logging.getLogger(__name__)
PUNCTUATION_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})
DIGIT_WORDS = {
    "0": "ZERO",
    "1": "ONE",
    "2": "TWO",
    "3": "THREE",
    "4": "FOUR",
    "5": "FIVE",
    "6": "SIX",
    "7": "SEVEN",
    "8": "EIGHT",
    "9": "NINE",
}


class AsrGrpcInferencer(Inferencer):
    def __init__(
        self,
        target: str = "192.168.1.24:50008",
        *,
        sample_rate: int = 16000,
        language_code: str = "en-US",
        hotwords: Iterable[str] | None = None,
        hotword_bias: float = 0.0,
        request_timeout_seconds: float = 60.0,
        connect_timeout_seconds: float | None = 10.0,
    ) -> None:
        self.target = target
        self.sample_rate = sample_rate
        self.language_code = language_code
        self.hotwords = list(hotwords or [])
        self.hotword_bias = hotword_bias
        self.request_timeout_seconds = request_timeout_seconds
        self.channel = grpc.insecure_channel(target)
        if connect_timeout_seconds is not None:
            grpc.channel_ready_future(self.channel).result(timeout=connect_timeout_seconds)
        self.stub = ux_speech_pb2_grpc.UxSpeechStub(self.channel)
        logger.info("ASR gRPC inferencer initialized: target=%s", target)

    def infer(self, audio: np.ndarray) -> str:
        audio_content = self._audio_to_linear16(audio)
        responses = self.stub.StreamingRecognize(
            self._build_requests(audio_content),
            timeout=self.request_timeout_seconds,
        )

        final_parts: list[str] = []
        latest_text = ""
        for response in responses:
            for result in response.results:
                transcript = self._normalize_text(result.alternative.transcript)
                if not transcript:
                    continue
                if result.is_final:
                    final_parts.append(transcript)
                else:
                    latest_text = transcript

        return self._normalize_text(" ".join(final_parts) or latest_text)

    def close(self) -> None:
        self.channel.close()
        logger.info("ASR gRPC inferencer closed: target=%s", self.target)

    def __enter__(self) -> AsrGrpcInferencer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _build_requests(
        self,
        audio_content: bytes,
    ) -> Iterator[ux_speech_pb2.StreamingRecognizeRequest]:
        yield ux_speech_pb2.StreamingRecognizeRequest(
            streaming_config=ux_speech_pb2.StreamingRecognitionConfig(
                config=ux_speech_pb2.RecognitionConfig(
                    encoding=ux_speech_pb2.RecognitionConfig.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    language_code=self.language_code,
                    enable_automatic_punctuation=False,
                    hotwords=self.hotwords,
                    hotword_bias=self.hotword_bias,
                ),
                interim_results=False,
            )
        )
        yield ux_speech_pb2.StreamingRecognizeRequest(audio_content=audio_content)

    def _audio_to_linear16(self, audio: np.ndarray) -> bytes:
        array = np.asarray(audio)
        array = np.squeeze(array)
        if array.ndim == 2:
            channel_axis = 0 if array.shape[0] <= array.shape[1] else 1
            array = array.mean(axis=channel_axis)
        if array.ndim != 1:
            raise ValueError(f"不支持的音频维度: {array.shape}")

        if np.issubdtype(array.dtype, np.integer):
            pcm = np.clip(array, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype("<i2")
        else:
            pcm = (np.clip(array, -1.0, 1.0) * np.iinfo(np.int16).max).astype("<i2")

        return pcm.tobytes()

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.translate(PUNCTUATION_TRANSLATION)
        text = "".join(f" {DIGIT_WORDS[char]} " if char in DIGIT_WORDS else char for char in text)
        return " ".join(text.strip().split()).upper()
