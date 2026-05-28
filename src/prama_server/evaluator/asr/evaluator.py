from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from typing import Dict, Tuple

import numpy as np
from datasets import Dataset

from prama_server.evaluator.asr.types import (
    EvaluationInferenceResult,
    EvaluationPartialInferenceResult,
)
from prama_server.inferencers.asr import AsrGrpcInferencer
from prama_server.metrics.prama_warpper import get_cer_pd, get_wer_pd
from prama_server.session.session import EvaluationSession

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        dataset: Dataset,
        *,
        sample_rate: int = 16000,
        tag: str | None = None,
        reference_postprocess: Callable[[str], str] | None = None,
        hypothesis_postprocess: Callable[[str], str] | None = None,
    ) -> None:
        self.dataset = dataset
        self.sample_rate = sample_rate
        self.tag = tag
        self.reference_postprocess = reference_postprocess
        self.hypothesis_postprocess = hypothesis_postprocess

    def close(self) -> None:
        logger.info("Evaluator closed: dataset_size=%s", len(self.dataset))

    def __enter__(self) -> Evaluator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def evaluate(self, inferencer: AsrGrpcInferencer) -> Dict[str, float]:
        return self.iter_evaluate(inferencer)

    def get_data_itr(self) -> Generator[Tuple[str, np.ndarray, str], None, None]:
        for sample in self.dataset:
            utterance_id = sample["id"]
            reference = " ".join(sample["text"].strip().split())
            if self.reference_postprocess is not None:
                reference = self.reference_postprocess(reference)

            audio = sample["audio"]
            audio_array, audio_sample_rate = _decode_audio(audio)

            audio_array = np.squeeze(audio_array)
            if audio_array.ndim == 2:
                channel_axis = 0 if audio_array.shape[0] <= audio_array.shape[1] else 1
                audio_array = audio_array.mean(axis=channel_axis)
            if audio_array.ndim != 1:
                raise ValueError(
                    f"不支持的音频维度: id={utterance_id} shape={audio_array.shape}"
                )
            if audio_sample_rate != self.sample_rate:
                raise ValueError(
                    f"样本采样率与 ASR 配置不一致: id={utterance_id} "
                    f"audio_sample_rate={audio_sample_rate} sample_rate={self.sample_rate}"
                )

            yield utterance_id, audio_array, reference

    def iter_evaluate(
        self,
        inferencer: AsrGrpcInferencer,
        on_infer_result: Callable[[EvaluationInferenceResult], None] | None = None,
        on_partial_infer_result: Callable[
            [EvaluationPartialInferenceResult], None
        ]
        | None = None,
    ) -> Dict[str, float]:
        data_itr = self.get_data_itr()
        session = EvaluationSession(
            data_itr=data_itr,
            inferencer=inferencer,
            metric_fns={"wer": get_wer_pd, "cer": get_cer_pd},
            hypothesis_postprocess=self.hypothesis_postprocess,
        )

        def publish_partial(
            data_id: str,
            reference: str,
            hypothesis: str,
            is_final: bool,
        ) -> None:
            if on_partial_infer_result is None:
                return
            on_partial_infer_result(
                EvaluationPartialInferenceResult(
                    tag=self.tag,
                    id=data_id,
                    reference=reference,
                    hypothesis=hypothesis,
                    is_final=is_final,
                )
            )

        for infer_results in session.infer(on_partial_result=publish_partial):
            if on_infer_result is None:
                continue
            latest_result = infer_results.iloc[-1]
            on_infer_result(
                EvaluationInferenceResult(
                    tag=self.tag,
                    id=str(latest_result["id"]),
                    reference=str(latest_result["reference"]),
                    hypothesis=str(latest_result["hypothesis"]),
                )
            )
        session_metrics = session.get_metrics()
        return session_metrics


def _decode_audio(audio: object) -> tuple[np.ndarray, int]:
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        data = samples.data
        audio_array = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
        return np.asarray(audio_array), int(samples.sample_rate)
    if isinstance(audio, dict):
        return np.asarray(audio["array"]), int(audio["sampling_rate"])
    raise TypeError(f"不支持的音频字段类型: {type(audio).__name__}")
