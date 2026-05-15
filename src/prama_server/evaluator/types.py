from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd


@dataclass(frozen=True)
class EvaluationResult:
    wer: float
    cer: float
    details: pd.DataFrame
    wer_result: Any
    cer_result: Any


@dataclass(frozen=True)
class EvaluationInferenceResult:
    tag: str | None
    id: str
    reference: str
    hypothesis: str


@dataclass(frozen=True)
class EvaluationProgress:
    status: Literal["started", "running", "completed"]
    total: int
    processed: int
    evaluated: int
    current_id: str | None = None
    reference: str | None = None
    hypothesis: str | None = None
    running_wer: float | None = None
    running_cer: float | None = None
    result: EvaluationResult | None = None
