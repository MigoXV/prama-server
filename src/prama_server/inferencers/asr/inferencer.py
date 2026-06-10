from __future__ import annotations

import logging
from collections.abc import Iterator, Generator
from typing import List, Optional, Tuple

import grpc
import numpy as np

from prama_server.inferencers.grpc_options import create_insecure_channel
from prama_server.protos.asr import ux_speech_pb2, ux_speech_pb2_grpc

logger = logging.getLogger(__name__)


class AsrGrpcInferencer:
    def __init__(
        self,
        target: str,
        sample_rate: int,
        language_code: str = "auto",
        hotwords: Optional[List[str]] = None,
        hotword_bias: float = 0.0,
        request_timeout_seconds: float = 60.0,
        connect_timeout_seconds: float | None = 10.0,
        interim_results: bool = True,
    ) -> None:
        self.target = target
        self.sample_rate = sample_rate
        self.language_code = language_code
        self.hotwords = list(hotwords or [])
        self.hotword_bias = hotword_bias
        self.request_timeout_seconds = request_timeout_seconds
        self.interim_results = interim_results
        self.channel = create_insecure_channel(target)
        if connect_timeout_seconds is not None:
            grpc.channel_ready_future(self.channel).result(
                timeout=connect_timeout_seconds
            )
        self.stub = ux_speech_pb2_grpc.UxSpeechStub(self.channel)

    def infer(self, audio: np.ndarray) -> Generator[Tuple[str, bool], None, None]:
        audio_content = self._audio_to_linear16(audio)
        responses = self.stub.StreamingRecognize(
            self._build_requests(audio_content),
            timeout=self.request_timeout_seconds,
        )
        for response in responses:
            for result in response.results:
                transcript = result.alternative.transcript.strip()
                if not transcript:
                    continue
                yield transcript, result.is_final

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
                interim_results=self.interim_results,
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
            pcm = np.clip(array, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(
                "<i2"
            )
        else:
            pcm = (np.clip(array, -1.0, 1.0) * np.iinfo(np.int16).max).astype("<i2")

        return pcm.tobytes()
