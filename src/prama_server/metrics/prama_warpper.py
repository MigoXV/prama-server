from __future__ import annotations

import pandas as pd
from prama.evaluator.evaluator import get_cer, get_wer


def get_metric_inputs(
    infer_results: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    evaluated_results = infer_results.dropna(
        subset=["id", "reference", "hypothesis"]
    )
    return (
        evaluated_results["reference"].astype(str).tolist(),
        evaluated_results["hypothesis"].astype(str).tolist(),
        evaluated_results["id"].astype(str).tolist(),
    )


def get_wer_pd(infer_results: pd.DataFrame) -> float:
    references, hypotheses, utterance_ids = get_metric_inputs(infer_results)
    if not references:
        return 0.0
    return get_wer(references, hypotheses, utterance_ids).summary.wer


def get_cer_pd(infer_results: pd.DataFrame) -> float:
    references, hypotheses, utterance_ids = get_metric_inputs(infer_results)
    if not references:
        return 0.0
    return get_cer(references, hypotheses, utterance_ids).summary.wer
