import logging
from collections.abc import Callable, Generator, Iterable
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
    infer_results: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(
            columns=["id", "reference", "hypothesis"]
        )
    )

    def infer(
        self,
    ) -> Generator[pd.DataFrame, None, None]:
        for data_id, reference, hypothesis in self.iter_infer():
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

    def iter_infer(self) -> Generator[Tuple[str, str, str], None, None]:
        with self.inferencer as active_inferencer:
            for data_id, audio, reference in self.data_itr:
                results = ""
                for hypothesis, is_final in active_inferencer.infer(audio):
                    if is_final:
                        results += hypothesis
                if results:
                    yield data_id, reference, results

    def get_metrics(self):
        metrics = {}
        for metric_name, metric_fn in self.metric_fns.items():
            metrics[metric_name] = metric_fn(self.infer_results)
        return metrics
