from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import grpc
import numpy as np

from prama_server.protos.lid import lid_pb2, lid_pb2_grpc

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LidResult:
    lang: str
    score: float


class LidGrpcInferencer:
    def __init__(
        self,
        target: str = "192.168.0.222:50026",
        sample_rate: int = 16000,
        request_timeout_seconds: float = 60.0,
        connect_timeout_seconds: float | None = 10.0,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.25,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate 必须大于 0: {sample_rate}")

        self.target = target
        self.sample_rate = sample_rate
        self.request_timeout_seconds = request_timeout_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.channel = grpc.insecure_channel(target)
        if connect_timeout_seconds is not None:
            grpc.channel_ready_future(self.channel).result(
                timeout=connect_timeout_seconds
            )
        self.stub = lid_pb2_grpc.LidServiceStub(self.channel)

    def infer(self, audio: np.ndarray) -> LidResult:
        request = lid_pb2.MsgLidProcessReq(
            pcm=self._audio_to_linear16(audio),
            sampleRate=self.sample_rate,
            channel=1,
            bitsPerSample=16,
        )
        response = self._process_with_retries(request)
        return LidResult(lang=response.lang, score=float(response.score))

    def _process_with_retries(
        self,
        request: lid_pb2.MsgLidProcessReq,
    ) -> lid_pb2.MsgLidProcessRes:
        for attempt in range(self.max_retries + 1):
            try:
                return self.stub.Process(
                    request,
                    timeout=self.request_timeout_seconds,
                )
            except grpc.RpcError as error:
                if attempt >= self.max_retries or not _is_retryable_rpc_error(error):
                    raise
                logger.warning(
                    "LID 推理失败，准备重试: target=%s attempt=%s/%s code=%s detail=%s",
                    self.target,
                    attempt + 1,
                    self.max_retries,
                    error.code(),
                    error.details(),
                )
                time.sleep(self.retry_delay_seconds * (attempt + 1))
        raise RuntimeError("LID 推理重试流程异常结束")

    def close(self) -> None:
        self.channel.close()
        logger.info("LID gRPC inferencer closed: target=%s", self.target)

    def __enter__(self) -> LidGrpcInferencer:
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


def _is_retryable_rpc_error(error: grpc.RpcError) -> bool:
    return error.code() in {
        grpc.StatusCode.INTERNAL,
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.UNKNOWN,
    }
