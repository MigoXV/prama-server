from __future__ import annotations

import logging

import grpc
import numpy as np

from prama_server.inferencers.grpc_options import create_insecure_channel
from prama_server.protos.denoise import ux_denoise_pb2, ux_denoise_pb2_grpc

logger = logging.getLogger(__name__)


class DenoiseGrpcInferencer:
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
        self.stub = ux_denoise_pb2_grpc.UxDenoiseStub(self.channel)

    def infer(self, audio: np.ndarray) -> np.ndarray:
        request = ux_denoise_pb2.DenoiseRequest(
            audio_in_config=ux_denoise_pb2.AudioInConfig(
                encoding=ux_denoise_pb2.LINEAR16,
                sample_rate_hertz=self.sample_rate,
            ),
            audio_out_config=ux_denoise_pb2.AudioOutConfig(
                encoding=ux_denoise_pb2.LINEAR16,
                sample_rate_hertz=self.sample_rate,
            ),
            audio_in=self._audio_to_linear16(audio),
        )
        response = self.stub.Denoise(request, timeout=self.request_timeout_seconds)
        denoised = self._linear16_to_audio(response.audio_out)
        if denoised.size == 0:
            raise ValueError("SE 服务返回空音频")
        return denoised

    def close(self) -> None:
        self.channel.close()
        logger.info("Denoise gRPC inferencer closed: target=%s", self.target)

    def __enter__(self) -> DenoiseGrpcInferencer:
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
            pcm = np.round(np.clip(array, -1.0, 1.0) * np.iinfo(np.int16).max).astype(
                "<i2"
            )
        return pcm.tobytes()

    def _linear16_to_audio(self, payload: bytes) -> np.ndarray:
        pcm = np.frombuffer(payload, dtype="<i2")
        return (pcm.astype(np.float32) / np.iinfo(np.int16).max).astype(
            np.float32,
            copy=False,
        )
