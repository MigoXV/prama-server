from __future__ import annotations

import logging
import math
from collections.abc import Iterator, Sequence

import grpc
import numpy as np

from prama_server.protos.vad import ux_vad_pb2, ux_vad_pb2_grpc

logger = logging.getLogger(__name__)


class VadGrpcInferencer:
    def __init__(
        self,
        target: str = "192.168.0.222:50021",
        sample_rate: int = 16000,
        mask_frame_seconds: float = 0.01,
        chunk_duration_seconds: float = 0.1,
        request_timeout_seconds: float = 60.0,
        connect_timeout_seconds: float | None = 10.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate 必须大于 0: {sample_rate}")
        if mask_frame_seconds <= 0:
            raise ValueError(
                f"mask_frame_seconds 必须大于 0: {mask_frame_seconds}"
            )
        if chunk_duration_seconds <= 0:
            raise ValueError(
                f"chunk_duration_seconds 必须大于 0: {chunk_duration_seconds}"
            )

        self.target = target
        self.sample_rate = sample_rate
        self.mask_frame_seconds = mask_frame_seconds
        self.chunk_duration_seconds = chunk_duration_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.channel = grpc.insecure_channel(target)
        if connect_timeout_seconds is not None:
            grpc.channel_ready_future(self.channel).result(
                timeout=connect_timeout_seconds
            )
        self.stub = ux_vad_pb2_grpc.UxVoiceActivityDetectorStub(self.channel)

    def infer(self, audio: np.ndarray) -> np.ndarray:
        audio_array = self._prepare_audio(audio)
        response = self.stub.Detect(
            ux_vad_pb2.DetectRequest(
                config=self._build_config(),
                audio=self._audio_to_linear16(audio_array),
            ),
            timeout=self.request_timeout_seconds,
        )
        return self._results_to_mask(response.results, len(audio_array))

    def stream_infer(self, audio: np.ndarray) -> np.ndarray:
        audio_array = self._prepare_audio(audio)
        responses = self.stub.StreamingDetect(
            self._build_streaming_requests(audio_array),
            timeout=self.request_timeout_seconds,
        )
        results: list[ux_vad_pb2.DetectionResult] = []
        for response in responses:
            results.extend(response.results)
        return self._results_to_mask(results, len(audio_array))

    def close(self) -> None:
        self.channel.close()
        logger.info("VAD gRPC inferencer closed: target=%s", self.target)

    def __enter__(self) -> VadGrpcInferencer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _build_config(self) -> ux_vad_pb2.DetectionConfig:
        return ux_vad_pb2.DetectionConfig(
            encoding=ux_vad_pb2.DetectionConfig.LINEAR16,
            sample_rate_hertz=self.sample_rate,
        )

    def _build_streaming_requests(
        self,
        audio: np.ndarray,
    ) -> Iterator[ux_vad_pb2.StreamingDetectRequest]:
        yield ux_vad_pb2.StreamingDetectRequest(config=self._build_config())

        chunk_size = max(1, int(self.sample_rate * self.chunk_duration_seconds))
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start : start + chunk_size]
            if len(chunk) == 0:
                continue
            yield ux_vad_pb2.StreamingDetectRequest(
                audio_content=self._audio_to_linear16(chunk)
            )

    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        array = np.asarray(audio)
        array = np.squeeze(array)
        if array.ndim == 2:
            channel_axis = 0 if array.shape[0] <= array.shape[1] else 1
            array = array.mean(axis=channel_axis)
        if array.ndim != 1:
            raise ValueError(f"不支持的音频维度: {array.shape}")
        return array

    def _audio_to_linear16(self, audio: np.ndarray) -> bytes:
        array = self._prepare_audio(audio)
        if np.issubdtype(array.dtype, np.integer):
            pcm = np.clip(array, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(
                "<i2"
            )
        else:
            pcm = (np.clip(array, -1.0, 1.0) * np.iinfo(np.int16).max).astype("<i2")
        return pcm.tobytes()

    def _results_to_mask(
        self,
        results: Sequence[ux_vad_pb2.DetectionResult],
        audio_sample_count: int,
    ) -> np.ndarray:
        mask_length = self._audio_sample_count_to_mask_length(audio_sample_count)
        mask = np.zeros(mask_length, dtype=bool)
        for result in results:
            self._mark_segment(mask, result.start_time, result.end_time)
        return mask

    def _audio_sample_count_to_mask_length(self, audio_sample_count: int) -> int:
        if audio_sample_count <= 0:
            return 0
        duration_seconds = audio_sample_count / self.sample_rate
        return int(math.ceil(duration_seconds / self.mask_frame_seconds))

    def _mark_segment(
        self,
        mask: np.ndarray,
        start_time: object,
        end_time: object,
    ) -> None:
        start_seconds = self._duration_to_seconds(start_time)
        end_seconds = self._duration_to_seconds(end_time)
        if start_seconds < 0 or end_seconds <= start_seconds:
            return

        start_index = max(0, int(math.floor(start_seconds / self.mask_frame_seconds)))
        end_index = min(
            len(mask),
            int(math.ceil(end_seconds / self.mask_frame_seconds)),
        )
        if end_index <= start_index:
            return
        mask[start_index:end_index] = True

    @staticmethod
    def _duration_to_seconds(duration: object) -> float:
        seconds = getattr(duration, "seconds")
        nanos = getattr(duration, "nanos")
        return float(seconds) + float(nanos) / 1_000_000_000
