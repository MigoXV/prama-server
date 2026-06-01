from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from prama_server.evaluator.vad import evaluate_masks
from prama_server.servicer.http import (
    EvaluationJob,
    EvaluationRequest,
    _build_cer_report,
    _build_wer_report,
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
                "duration_seconds": 12.5,
            },
            {
                "id": "bad",
                "index": 2,
                "reference": "hello world",
                "hypothesis": "bad world",
                "duration_seconds": 7.5,
            },
        ]
        job.result = {"processing_elapsed_seconds": 5.0}
        _put_job(job)

        response = client.post(
            f"/api/evaluations/{job.job_id}/metrics/recalculate",
            json={"excluded_sample_ids": ["bad"]},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["wer"], 0.0)
        self.assertEqual(payload["cer"], 0.0)
        self.assertEqual(payload["included_sample_count"], 1)
        self.assertEqual(payload["excluded_sample_count"], 1)
        self.assertEqual(payload["excluded_sample_ids"], ["bad"])
        self.assertEqual(payload["audio_duration_seconds"], 12.5)
        self.assertEqual(payload["processing_elapsed_seconds"], 5.0)
        self.assertEqual(payload["realtime_factor"], 2.5)
        self.assertEqual(
            [item["id"] for item in payload["wer_report"]["utterances"]],
            ["ok"],
        )
        self.assertEqual(
            [item["id"] for item in payload["cer_report"]["utterances"]],
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
        job.vad_report_samples = [
            {"id": "hit", "duration_seconds": 8.0},
            {"id": "miss", "duration_seconds": 12.0},
        ]
        job.result = {"processing_elapsed_seconds": 4.0}
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
        self.assertEqual(payload["audio_duration_seconds"], 8.0)
        self.assertEqual(payload["processing_elapsed_seconds"], 4.0)
        self.assertEqual(payload["realtime_factor"], 2.0)
        self.assertEqual(
            payload["vad_report"]["samples"],
            [{"id": "hit", "duration_seconds": 8.0}],
        )

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
        self.assertEqual(payload["cer_report"]["utterances"], [])
        self.assertEqual(payload["included_sample_count"], 0)

    def test_asr_alignment_reports_include_word_and_character_tokens(self) -> None:
        rows = [
            {
                "id": "zh",
                "index": 1,
                "reference": "我们刚刚上一段会议",
                "hypothesis": "我们刚上一段会",
            }
        ]

        wer_report = _build_wer_report(rows)
        cer_report = _build_cer_report(rows)

        self.assertEqual(len(wer_report["utterances"][0]["tokens"]), 1)
        self.assertGreater(len(cer_report["utterances"][0]["tokens"]), 1)
        self.assertEqual(cer_report["utterances"][0]["tokens"][0]["ref"], "我")
        self.assertLess(cer_report["summary"]["wer"], wer_report["summary"]["wer"])

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

    def test_directory_browser_lists_root_and_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "data-bin"
            dataset_dir = root / "dataset"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "metadata.jsonl").write_text("{}", encoding="utf-8")
            (root / "sample.wav").write_bytes(b"audio")

            with patch.dict("os.environ", {"PRAMA_WORKDIR": str(root)}):
                response = client.get("/api/files/directories")
                child_response = client.get(
                    "/api/files/directories",
                    params={"path": str(dataset_dir)},
                )
                escape_response = client.get(
                    "/api/files/directories",
                    params={"path": str(root.parent)},
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["currentPath"], str(root))
            self.assertIn(
                {"name": "dataset", "path": str(dataset_dir), "kind": "directory"},
                payload["entries"],
            )
            self.assertIn(
                {"name": "sample.wav", "path": str(root / "sample.wav"), "kind": "file"},
                payload["entries"],
            )
            self.assertEqual(child_response.status_code, 200)
            self.assertEqual(child_response.json()["parentPath"], str(root))
            self.assertEqual(escape_response.status_code, 403)

    def test_upload_dataset_saves_supported_files_and_skips_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir) / "workdir"
            with patch.dict("os.environ", {"PRAMA_WORKDIR": str(workdir)}):
                response = client.post(
                    "/api/datasets/upload",
                    files=[
                        ("files", ("audio.wav", b"audio", "audio/wav")),
                        ("files", ("metadata.jsonl", b"{}", "application/json")),
                        ("files", ("notes.exe", b"skip", "application/octet-stream")),
                    ],
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            dataset_path = Path(payload["dataset_path"])
            self.assertEqual(payload["imported_count"], 2)
            self.assertEqual(payload["skipped_count"], 1)
            self.assertEqual(dataset_path.parent, workdir)
            self.assertTrue((dataset_path / "audio.wav").exists())
            self.assertTrue((dataset_path / "metadata.jsonl").exists())
            self.assertFalse((dataset_path / "notes.exe").exists())

    def test_upload_dataset_preserves_uploaded_directory_under_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workdir = Path(temp_dir) / "workdir"
            with patch.dict("os.environ", {"PRAMA_WORKDIR": str(workdir)}):
                response = client.post(
                    "/api/datasets/upload",
                    files=[
                        (
                            "files",
                            ("demo-dataset/audio.wav", b"audio", "audio/wav"),
                        ),
                        (
                            "files",
                            ("demo-dataset/metadata.jsonl", b"{}", "application/json"),
                        ),
                    ],
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            dataset_path = Path(payload["dataset_path"])
            self.assertEqual(dataset_path, workdir / "demo-dataset")
            self.assertTrue((dataset_path / "audio.wav").exists())
            self.assertTrue((dataset_path / "metadata.jsonl").exists())

    def test_upload_dataset_rejects_all_unsupported_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(
                "os.environ",
                {"PRAMA_WORKDIR": str(Path(temp_dir) / "workdir")},
            ):
                response = client.post(
                    "/api/datasets/upload",
                    files=[("files", ("bad.exe", b"skip", "application/octet-stream"))],
                )

            self.assertEqual(response.status_code, 400)


def _put_job(job: EvaluationJob) -> None:
    with jobs_lock:
        jobs[job.job_id] = job


if __name__ == "__main__":
    unittest.main()
