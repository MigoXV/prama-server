import logging
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from prama_server.inferencers.asr import AsrGrpcInferencer

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSession:
    data_itr: Iterable[Tuple[str, np.ndarray, str]]
    inferencer: AsrGrpcInferencer
    metric_fns: Dict[str, Callable]
    hypothesis_postprocess: Callable[[str], str] | None = None
    inference_concurrency: int = 0
    infer_results: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["id", "reference", "hypothesis"]
        )
    )

    def infer(
        self,
        on_partial_result: Callable[[str, str, str, bool], None] | None = None,
    ) -> Generator[pd.DataFrame, None, None]:
        for data_id, reference, hypothesis in self.iter_infer(on_partial_result):
            if self.hypothesis_postprocess is not None:
                hypothesis = self.hypothesis_postprocess(hypothesis)
            self.infer_results = pd.concat(
                [
                    self.infer_results,
                    pd.DataFrame(
                        [
                            {
                                "id": data_id,
                                "reference": reference,
                                "hypothesis": hypothesis,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            yield self.infer_results

    def iter_infer(
        self,
        on_partial_result: Callable[[str, str, str, bool], None] | None = None,
    ) -> Generator[Tuple[str, str, str], None, None]:
        if self.inference_concurrency > 0:
            yield from self._iter_infer_concurrently(on_partial_result)
            return

        with self.inferencer as active_inferencer:
            for data_id, audio, reference in self.data_itr:
                result = self._infer_one(
                    active_inferencer=active_inferencer,
                    data_id=data_id,
                    audio=audio,
                    reference=reference,
                    on_partial_result=on_partial_result,
                )
                if result is not None:
                    yield result

    def _iter_infer_concurrently(
        self,
        on_partial_result: Callable[[str, str, str, bool], None] | None = None,
    ) -> Generator[Tuple[str, str, str], None, None]:
        samples = list(self.data_itr)
        max_workers = min(self.inference_concurrency, len(samples))
        if max_workers <= 0:
            return

        with self.inferencer as active_inferencer:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        self._infer_one,
                        active_inferencer=active_inferencer,
                        data_id=data_id,
                        audio=audio,
                        reference=reference,
                        on_partial_result=on_partial_result,
                    )
                    for data_id, audio, reference in samples
                ]
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        yield result

    def _infer_one(
        self,
        *,
        active_inferencer: AsrGrpcInferencer,
        data_id: str,
        audio: np.ndarray,
        reference: str,
        on_partial_result: Callable[[str, str, str, bool], None] | None,
    ) -> Tuple[str, str, str] | None:
        results = ""
        for hypothesis, is_final in active_inferencer.infer(audio):
            if self.hypothesis_postprocess is not None:
                display_hypothesis = self.hypothesis_postprocess(hypothesis)
            else:
                display_hypothesis = hypothesis
            if on_partial_result is not None:
                on_partial_result(
                    data_id,
                    reference,
                    display_hypothesis,
                    is_final,
                )
            if is_final:
                results += hypothesis
        if not results:
            return None
        return data_id, reference, results

    def get_metrics(self):
        metrics = {}
        for metric_name, metric_fn in self.metric_fns.items():
            metrics[metric_name] = metric_fn(self.infer_results)
        return metrics
