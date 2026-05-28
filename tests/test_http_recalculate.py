from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from prama_server.evaluator.vad import evaluate_masks
from prama_server.servicer.http import (
    EvaluationJob,
    EvaluationRequest,
    app,
    jobs,
    jobs_lock,
)


client = TestClient(app)


class HttpRecalculateTest(unittest.TestCase):
    def test_recalculate_asr_metrics_excludes_selected_sample(self) -> None:
        job = EvaluationJob(
            job_id="test-asr-recalculate",
            request=EvaluationRequest(task="asr"),
            status="completed",
        )
        job.sample_records = {
            "ok": {"id": "ok", "index": 1},
            "bad": {"id": "bad", "index": 2},
        }
        job.asr_inference_rows = [
            {
                "id": "ok",
                "index": 1,
                "reference": "hello world",
                "hypothesis": "hello world",
            },
            {
                "id": "bad",
                "index": 2,
                "reference": "hello world",
                "hypothesis": "bad world",
            },
        ]
        _put_job(job)

        response = client.post(
            f"/api/evaluations/{job.job_id}/metrics/recalculate",
            json={"excluded_sample_ids": ["bad"]},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["wer"], 0.0)
        self.assertEqual(payload["included_sample_count"], 1)
        self.assertEqual(payload["excluded_sample_count"], 1)
        self.assertEqual(payload["excluded_sample_ids"], ["bad"])
        self.assertEqual(
            [item["id"] for item in payload["wer_report"]["utterances"]],
            ["ok"],
        )

    def test_recalculate_vad_metrics_excludes_selected_sample(self) -> None:
        hit = asdict(
            evaluate_masks(
                np.array([False, True, True, False], dtype=bool),
                np.array([False, True, True, False], dtype=bool),
            )
        )
        miss = asdict(
            evaluate_masks(
                np.array([False, True, True, False], dtype=bool),
                np.array([False, False, False, False], dtype=bool),
            )
        )
        job = EvaluationJob(
            job_id="test-vad-recalculate",
            request=EvaluationRequest(task="vad"),
            status="completed",
        )
        job.sample_records = {
            "hit": {"id": "hit", "index": 1},
            "miss": {"id": "miss", "index": 2},
        }
        job.vad_metric_rows = [{"id": "hit", **hit}, {"id": "miss", **miss}]
        job.vad_report_samples = [{"id": "hit"}, {"id": "miss"}]
        _put_job(job)

        response = client.post(
            f"/api/evaluations/{job.job_id}/metrics/recalculate",
            json={"excluded_sample_ids": ["miss"]},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["sample_count"], 1)
        self.assertEqual(payload["frame"]["frame_f1"], 1.0)
        self.assertEqual(payload["included_sample_count"], 1)
        self.assertEqual(payload["excluded_sample_count"], 1)
        self.assertEqual(payload["vad_report"]["samples"], [{"id": "hit"}])

    def test_recalculate_with_all_samples_excluded_returns_empty_report(self) -> None:
        job = EvaluationJob(
            job_id="test-recalculate-empty",
            request=EvaluationRequest(task="asr"),
            status="completed",
        )
        job.sample_records = {"only": {"id": "only", "index": 1}}
        job.asr_inference_rows = [
            {
                "id": "only",
                "index": 1,
                "reference": "hello",
                "hypothesis": "world",
            }
        ]
        _put_job(job)

        response = client.post(
            f"/api/evaluations/{job.job_id}/metrics/recalculate",
            json={"excluded_sample_ids": ["only"]},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["wer"], 0.0)
        self.assertEqual(payload["cer"], 0.0)
        self.assertEqual(payload["wer_report"]["utterances"], [])
        self.assertEqual(payload["included_sample_count"], 0)

    def test_sample_audio_endpoint_only_serves_registered_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "sample.wav"
            sf.write(audio_path, np.zeros(160, dtype=np.float32), 16000)
            job = EvaluationJob(
                job_id="test-audio",
                request=EvaluationRequest(task="asr"),
                status="completed",
            )
            job.sample_records = {
                "sample": {
                    "id": "sample",
                    "index": 1,
                    "audio_path": str(audio_path),
                }
            }
            _put_job(job)

            response = client.get(f"/api/evaluations/{job.job_id}/samples/sample/audio")
            missing_response = client.get(
                f"/api/evaluations/{job.job_id}/samples/missing/audio"
            )

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith("audio/"))
            self.assertEqual(missing_response.status_code, 404)


def _put_job(job: EvaluationJob) -> None:
    with jobs_lock:
        jobs[job.job_id] = job


if __name__ == "__main__":
    unittest.main()
