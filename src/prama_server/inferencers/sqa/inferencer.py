from __future__ import annotations

import logging
from dataclasses import dataclass

import grpc
import numpy as np

from prama_server.inferencers.grpc_options import create_insecure_channel
from prama_server.protos.sqa import ux_sqa_pb2, ux_sqa_pb2_grpc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SqaResult:
    score: float


class SqaGrpcInferencer:
    def __init__(
        self,
        target: str,
        sample_rate: int = 16000,
        request_timeout_seconds: float = 60.0,
        connect_timeout_seconds: float | None = 10.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate 必须大于 0: {sample_rate}")

        self.target = target
        self.sample_rate = sample_rate
        self.request_timeout_seconds = request_timeout_seconds
        self.channel = create_insecure_channel(target)
        if connect_timeout_seconds is not None:
            grpc.channel_ready_future(self.channel).result(
                timeout=connect_timeout_seconds
            )
        self.stub = ux_sqa_pb2_grpc.UxSqaStub(self.channel)

    def infer(self, audio: np.ndarray) -> SqaResult:
        response = self.stub.Assess(
            ux_sqa_pb2.AssessRequest(
                config=ux_sqa_pb2.QualityConfig(
                    encoding=ux_sqa_pb2.QualityConfig.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                ),
                audio=self._audio_to_linear16(audio),
            ),
            timeout=self.request_timeout_seconds,
        )
        return SqaResult(score=float(response.score))

    def close(self) -> None:
        self.channel.close()
        logger.info("SQA gRPC inferencer closed: target=%s", self.target)

    def __enter__(self) -> SqaGrpcInferencer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

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
