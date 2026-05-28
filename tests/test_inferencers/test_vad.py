from __future__ import annotations

import unittest

import numpy as np
from google.protobuf.duration_pb2 import Duration

from prama_server.inferencers.vad import VadGrpcInferencer


class VadGrpcInferencerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.inferencer = VadGrpcInferencer.__new__(VadGrpcInferencer)
        self.inferencer.sample_rate = 16000
        self.inferencer.mask_frame_seconds = 0.01
        self.inferencer.chunk_duration_seconds = 0.1

    def test_float_audio_to_linear16(self) -> None:
        audio = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)

        pcm = np.frombuffer(self.inferencer._audio_to_linear16(audio), dtype="<i2")

        np.testing.assert_array_equal(
            pcm,
            np.array([-32767, 0, 32767, 32767], dtype=np.int16),
        )

    def test_int_audio_to_linear16(self) -> None:
        audio = np.array([-40000, -1, 0, 40000], dtype=np.int32)

        pcm = np.frombuffer(self.inferencer._audio_to_linear16(audio), dtype="<i2")

        np.testing.assert_array_equal(
            pcm,
            np.array([-32768, -1, 0, 32767], dtype=np.int16),
        )

    def test_mark_segment_uses_10ms_frames(self) -> None:
        mask = np.zeros(10, dtype=bool)

        self.inferencer._mark_segment(
            mask,
            Duration(seconds=0, nanos=15_000_000),
            Duration(seconds=0, nanos=35_000_000),
        )

        np.testing.assert_array_equal(
            mask,
            np.array(
                [False, True, True, True, False, False, False, False, False, False],
                dtype=bool,
            ),
        )

    def test_results_to_mask_ignores_negative_boundaries(self) -> None:
        result = type("Result", (), {})()
        result.start_time = Duration(seconds=-1)
        result.end_time = Duration(seconds=0, nanos=50_000_000)

        mask = self.inferencer._results_to_mask([result], audio_sample_count=1600)

        np.testing.assert_array_equal(mask, np.zeros(10, dtype=bool))


if __name__ == "__main__":
    unittest.main()
