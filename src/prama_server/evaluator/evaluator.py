from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from typing import Dict

import numpy as np
from datasets import Dataset
from typing import Tuple

from prama_server.evaluator.types import EvaluationInferenceResult
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
            audio_array = audio["array"]
            audio_sample_rate = audio["sampling_rate"]

            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1)
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
    ) -> Dict[str, float]:
        data_itr = self.get_data_itr()
        session = EvaluationSession(
            data_itr=data_itr,
            inferencer=inferencer,
            metric_fns={"wer": get_wer_pd, "cer": get_cer_pd},
            hypothesis_postprocess=self.hypothesis_postprocess,
        )
        for infer_results in session.infer():
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
