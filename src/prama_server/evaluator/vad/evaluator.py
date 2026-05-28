from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage

from prama_server.evaluator.vad.types import VadEvaluationResult

Segment = tuple[int, int]


class VadEvaluator:
    def __init__(self, hit_threshold: float = 0.9) -> None:
        self.hit_threshold = hit_threshold

    def evaluate(
        self,
        reference_mask: np.ndarray,
        prediction_mask: np.ndarray,
    ) -> VadEvaluationResult:
        return evaluate_masks(
            reference_mask,
            prediction_mask,
            hit_threshold=self.hit_threshold,
        )


def evaluate_masks(
    reference_mask: np.ndarray,
    prediction_mask: np.ndarray,
    *,
    hit_threshold: float = 0.9,
) -> VadEvaluationResult:
    if not 0.0 <= hit_threshold <= 1.0:
        raise ValueError(f"hit_threshold 必须在 [0, 1] 内: {hit_threshold}")

    reference = _as_bool_mask(reference_mask, name="reference_mask")
    prediction = _as_bool_mask(prediction_mask, name="prediction_mask")
    if reference.shape != prediction.shape:
        raise ValueError(
            "reference_mask 与 prediction_mask 形状必须一致: "
            f"{reference.shape} != {prediction.shape}"
        )

    tp = int(np.count_nonzero(reference & prediction))
    tn = int(np.count_nonzero(~reference & ~prediction))
    fp = int(np.count_nonzero(~reference & prediction))
    fn = int(np.count_nonzero(reference & ~prediction))

    total = reference.size
    speech_frames = int(np.count_nonzero(reference))
    non_speech_frames = total - speech_frames
    frame_accuracy = (tp + tn) / total if total else 1.0
    frame_recall = _safe_divide(tp, tp + fn)
    frame_precision = _safe_divide(tp, tp + fp)
    frame_f1 = _f1(frame_precision, frame_recall)
    frame_specificity = _safe_divide(tn, tn + fp)
    frame_false_alarm_rate = _safe_divide(fp, fp + tn)
    frame_miss_rate = _safe_divide(fn, fn + tp)
    frame_balanced_accuracy = (frame_recall + frame_specificity) / 2

    reference_segments = _connected_segments(reference)
    prediction_segments = _connected_segments(prediction)
    segment_hit_count = _segment_hit_count(
        source_segments=reference_segments,
        target_segments=prediction_segments,
        hit_threshold=hit_threshold,
    )
    prediction_hit_count = _segment_overlap_count(
        source_segments=prediction_segments,
        target_segments=reference_segments,
    )
    segment_false_alarm_count = len(prediction_segments) - prediction_hit_count
    segment_miss_count = len(reference_segments) - segment_hit_count
    segment_recall = _segment_score(
        source_segments=reference_segments,
        target_segments=prediction_segments,
        hit_threshold=hit_threshold,
    )
    segment_precision = _safe_divide(prediction_hit_count, len(prediction_segments))
    segment_f1 = _f1(segment_precision, segment_recall)
    segment_miss_rate = _safe_divide(segment_miss_count, len(reference_segments))
    segment_false_alarm_rate = _safe_divide(
        segment_false_alarm_count,
        len(prediction_segments),
    )

    return VadEvaluationResult(
        frame_total=total,
        frame_speech=speech_frames,
        frame_non_speech=non_speech_frames,
        frame_true_positive=tp,
        frame_true_negative=tn,
        frame_false_positive=fp,
        frame_false_negative=fn,
        frame_accuracy=frame_accuracy,
        frame_recall=frame_recall,
        frame_precision=frame_precision,
        frame_f1=frame_f1,
        frame_specificity=frame_specificity,
        frame_false_alarm_rate=frame_false_alarm_rate,
        frame_miss_rate=frame_miss_rate,
        frame_balanced_accuracy=frame_balanced_accuracy,
        segment_hit_count=segment_hit_count,
        segment_miss_count=segment_miss_count,
        segment_false_alarm_count=segment_false_alarm_count,
        segment_recall=segment_recall,
        segment_precision=segment_precision,
        segment_f1=segment_f1,
        segment_miss_rate=segment_miss_rate,
        segment_false_alarm_rate=segment_false_alarm_rate,
        reference_segment_count=len(reference_segments),
        prediction_segment_count=len(prediction_segments),
    )


def _as_bool_mask(mask: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.dtype != np.bool_:
        raise TypeError(f"{name} 必须是 bool np.ndarray: dtype={array.dtype}")
    array = np.squeeze(array)
    if array.ndim != 1:
        raise ValueError(f"{name} 必须是一维 bool mask: shape={array.shape}")
    return array


def _connected_segments(mask: np.ndarray) -> list[Segment]:
    labels, _ = ndimage.label(mask)
    segments: list[Segment] = []
    for slice_tuple in ndimage.find_objects(labels):
        if slice_tuple is None:
            continue
        item = slice_tuple[0]
        segments.append((int(item.start), int(item.stop)))
    return segments


def _segment_score(
    *,
    source_segments: Sequence[Segment],
    target_segments: Sequence[Segment],
    hit_threshold: float,
) -> float:
    if not source_segments:
        return 0.0

    return _segment_hit_count(
        source_segments=source_segments,
        target_segments=target_segments,
        hit_threshold=hit_threshold,
    ) / len(source_segments)


def _segment_hit_count(
    *,
    source_segments: Sequence[Segment],
    target_segments: Sequence[Segment],
    hit_threshold: float,
) -> int:
    hit_count = 0
    for source in source_segments:
        source_len = source[1] - source[0]
        if source_len <= 0:
            continue
        if any(
            _overlap_length(source, target) / source_len >= hit_threshold
            for target in target_segments
        ):
            hit_count += 1
    return hit_count


def _segment_overlap_count(
    *,
    source_segments: Sequence[Segment],
    target_segments: Sequence[Segment],
) -> int:
    return sum(
        1
        for source in source_segments
        if any(_overlap_length(source, target) > 0 for target in target_segments)
    )


def _overlap_length(left: Segment, right: Segment) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
