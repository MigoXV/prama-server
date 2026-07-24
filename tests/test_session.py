from __future__ import annotations

import unittest

import numpy as np

from prama_server.session.session import EvaluationSession


class FakeInferencer:
    def __enter__(self) -> FakeInferencer:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def infer(self, audio: np.ndarray):
        yield f"hyp-{int(audio[0])}", True


class EvaluationSessionTest(unittest.TestCase):
    def test_iter_infer_concurrently_returns_all_results(self) -> None:
        session = EvaluationSession(
            data_itr=[
                ("a", np.array([1]), "ref-a"),
                ("b", np.array([2]), "ref-b"),
                ("c", np.array([3]), "ref-c"),
            ],
            inferencer=FakeInferencer(),
            metric_fns={},
            inference_concurrency=2,
        )

        results = list(session.iter_infer())

        self.assertEqual(
            set(results),
            {
                ("a", "ref-a", "hyp-1"),
                ("b", "ref-b", "hyp-2"),
                ("c", "ref-c", "hyp-3"),
            },
        )


if __name__ == "__main__":
    unittest.main()
