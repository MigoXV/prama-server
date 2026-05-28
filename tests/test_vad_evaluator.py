from __future__ import annotations

import unittest

import numpy as np

from prama_server.evaluator.vad import VadEvaluator, evaluate_masks


class VadEvaluatorTest(unittest.TestCase):
    def test_identical_masks_are_perfect(self) -> None:
        reference = np.array(
            [False, True, True, False, True, True, True, False],
            dtype=bool,
        )
        result = evaluate_masks(reference, reference)

        self.assertEqual(result.frame_accuracy, 1.0)
        self.assertEqual(result.frame_recall, 1.0)
        self.assertEqual(result.frame_precision, 1.0)
        self.assertEqual(result.frame_f1, 1.0)
        self.assertEqual(result.segment_recall, 1.0)
        self.assertEqual(result.segment_precision, 1.0)
        self.assertEqual(result.reference_segment_count, 2)
        self.assertEqual(result.prediction_segment_count, 2)

    def test_empty_prediction_uses_zero_division_policy(self) -> None:
        reference = np.array([False, True, True, False], dtype=bool)
        prediction = np.zeros_like(reference, dtype=bool)

        result = evaluate_masks(reference, prediction)

        self.assertEqual(result.frame_accuracy, 0.5)
        self.assertEqual(result.frame_recall, 0.0)
        self.assertEqual(result.frame_precision, 0.0)
        self.assertEqual(result.frame_f1, 0.0)
        self.assertEqual(result.segment_recall, 0.0)
        self.assertEqual(result.segment_precision, 0.0)
        self.assertEqual(result.reference_segment_count, 1)
        self.assertEqual(result.prediction_segment_count, 0)

    def test_empty_reference_and_prediction_have_perfect_accuracy(self) -> None:
        reference = np.zeros(4, dtype=bool)
        prediction = np.zeros(4, dtype=bool)

        result = VadEvaluator().evaluate(reference, prediction)

        self.assertEqual(result.frame_accuracy, 1.0)
        self.assertEqual(result.frame_recall, 0.0)
        self.assertEqual(result.frame_precision, 0.0)
        self.assertEqual(result.frame_f1, 0.0)
        self.assertEqual(result.segment_recall, 0.0)
        self.assertEqual(result.segment_precision, 0.0)
        self.assertEqual(result.reference_segment_count, 0)
        self.assertEqual(result.prediction_segment_count, 0)

    def test_segment_threshold_uses_each_source_segment_as_denominator(self) -> None:
        reference = np.zeros(20, dtype=bool)
        prediction = np.zeros(20, dtype=bool)
        reference[0:10] = True
        prediction[0:9] = True

        hit_result = evaluate_masks(reference, prediction, hit_threshold=0.9)
        miss_result = evaluate_masks(reference, prediction, hit_threshold=0.91)

        self.assertEqual(hit_result.segment_recall, 1.0)
        self.assertEqual(hit_result.segment_precision, 1.0)
        self.assertEqual(miss_result.segment_recall, 0.0)
        self.assertEqual(miss_result.segment_precision, 1.0)

    def test_prediction_segment_with_any_reference_overlap_is_not_false_alarm(
        self,
    ) -> None:
        reference = np.zeros(20, dtype=bool)
        prediction = np.zeros(20, dtype=bool)
        reference[5:10] = True
        prediction[0:6] = True

        result = evaluate_masks(reference, prediction, hit_threshold=0.9)

        self.assertEqual(result.segment_false_alarm_count, 0)
        self.assertEqual(result.segment_false_alarm_rate, 0.0)
        self.assertEqual(result.segment_precision, 1.0)

    def test_segment_precision_is_prediction_segments_hitting_reference_over_all_predictions(
        self,
    ) -> None:
        reference = np.zeros(30, dtype=bool)
        prediction = np.zeros(30, dtype=bool)
        reference[5:10] = True
        prediction[0:6] = True
        prediction[20:25] = True

        result = evaluate_masks(reference, prediction, hit_threshold=0.9)

        self.assertEqual(result.prediction_segment_count, 2)
        self.assertEqual(result.segment_precision, 0.5)
        self.assertEqual(result.segment_false_alarm_count, 1)

    def test_rejects_non_bool_masks(self) -> None:
        reference = np.array([0, 1, 1], dtype=np.int64)
        prediction = np.array([False, True, True], dtype=bool)

        with self.assertRaises(TypeError):
            evaluate_masks(reference, prediction)

    def test_rejects_shape_mismatch(self) -> None:
        reference = np.array([False, True], dtype=bool)
        prediction = np.array([False, True, False], dtype=bool)

        with self.assertRaises(ValueError):
            evaluate_masks(reference, prediction)


if __name__ == "__main__":
    unittest.main()
