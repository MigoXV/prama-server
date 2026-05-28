from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VadEvaluationResult:
    frame_total: int
    frame_speech: int
    frame_non_speech: int
    frame_true_positive: int
    frame_true_negative: int
    frame_false_positive: int
    frame_false_negative: int
    frame_accuracy: float
    frame_recall: float
    frame_precision: float
    frame_f1: float
    frame_specificity: float
    frame_false_alarm_rate: float
    frame_miss_rate: float
    frame_balanced_accuracy: float
    segment_hit_count: int
    segment_miss_count: int
    segment_false_alarm_count: int
    segment_recall: float
    segment_precision: float
    segment_f1: float
    segment_miss_rate: float
    segment_false_alarm_rate: float
    reference_segment_count: int
    prediction_segment_count: int
