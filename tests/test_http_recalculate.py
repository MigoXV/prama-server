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
    SqaAssessor,
    _build_denoise_report,
    _build_sqa_summary,
    _build_cer_report,
    _build_wer_report,
    _build_lid_report,
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
                "sqa_scores": [
                    {
                        "engine_name": "MOS",
                        "target": "127.0.0.1:50111",
                        "score": 4.0,
                        "error": None,
                    }
                ],
            },
            {
                "id": "bad",
                "index": 2,
                "reference": "hello world",
                "hypothesis": "bad world",
                "duration_seconds": 7.5,
                "sqa_scores": [
                    {
                        "engine_name": "MOS",
                        "target": "127.0.0.1:50111",
                        "score": 1.0,
                        "error": None,
                    }
                ],
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
        self.assertEqual(payload["sqa_summary"][0]["mean_score"], 4.0)
        self.assertEqual(
            payload["wer_report"]["utterances"][0]["sqa_scores"][0]["score"],
            4.0,
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

    def test_recalculate_lid_metrics_excludes_selected_sample(self) -> None:
        job = EvaluationJob(
            job_id="test-lid-recalculate",
            request=EvaluationRequest(task="lid"),
            status="completed",
        )
        job.sample_records = {
            "ok": {"id": "ok", "index": 1},
            "bad": {"id": "bad", "index": 2},
        }
        job.lid_report_samples = [
            {
                "id": "ok",
                "index": 1,
                "duration_seconds": 3.0,
                "reference_language": "en",
                "predicted_language": "en",
                "raw_language": "en",
                "confidence": 0.9,
                "correct": True,
            },
            {
                "id": "bad",
                "index": 2,
                "duration_seconds": 2.0,
                "reference_language": "zh",
                "predicted_language": "en",
                "raw_language": "en",
                "confidence": 0.7,
                "correct": False,
            },
        ]
        job.result = {"processing_elapsed_seconds": 2.5}
        _put_job(job)

        response = client.post(
            f"/api/evaluations/{job.job_id}/metrics/recalculate",
            json={"excluded_sample_ids": ["bad"]},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["accuracy"], 1.0)
        self.assertEqual(payload["recall"], 1.0)
        self.assertEqual(payload["correct_count"], 1)
        self.assertEqual(payload["sample_count"], 1)
        self.assertEqual(payload["audio_duration_seconds"], 3.0)
        self.assertEqual(payload["processing_elapsed_seconds"], 2.5)
        self.assertEqual(payload["realtime_factor"], 1.2)
        self.assertEqual(
            [item["id"] for item in payload["lid_report"]["samples"]],
            ["ok"],
        )

    def test_recalculate_denoise_metrics_excludes_selected_sample(self) -> None:
        job = EvaluationJob(
            job_id="test-denoise-recalculate",
            request=EvaluationRequest(
                task="denoise",
                enable_mos=True,
                mos_target="mos:50111",
            ),
            status="completed",
        )
        job.sample_records = {
            "ok": {"id": "ok", "index": 1},
            "bad": {"id": "bad", "index": 2},
        }
        job.denoise_report_samples = [
            {
                "id": "ok",
                "index": 1,
                "duration_seconds": 3.0,
                "original_snr": 10.0,
                "denoised_snr": 12.0,
                "snr_delta": 2.0,
                "original_mos": 3.0,
                "denoised_mos": 3.5,
                "mos_delta": 0.5,
            },
            {
                "id": "bad",
                "index": 2,
                "duration_seconds": 5.0,
                "original_snr": 9.0,
                "denoised_snr": 8.0,
                "snr_delta": -1.0,
                "original_mos": 2.0,
                "denoised_mos": 1.5,
                "mos_delta": -0.5,
            },
        ]
        job.result = {"processing_elapsed_seconds": 2.0}
        _put_job(job)

        response = client.post(
            f"/api/evaluations/{job.job_id}/metrics/recalculate",
            json={"excluded_sample_ids": ["bad"]},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mean_snr_delta"], 2.0)
        self.assertEqual(payload["mean_mos_delta"], 0.5)
        self.assertEqual(payload["included_sample_count"], 1)
        self.assertEqual(payload["excluded_sample_ids"], ["bad"])
        self.assertEqual(payload["audio_duration_seconds"], 3.0)
        self.assertEqual(
            [item["id"] for item in payload["denoise_report"]["samples"]],
            ["ok"],
        )

    def test_recalculate_lid_metrics_uses_macro_recall_by_reference_language(self) -> None:
        job = EvaluationJob(
            job_id="test-lid-macro-recall",
            request=EvaluationRequest(task="lid"),
            status="completed",
        )
        job.sample_records = {
            "en-ok": {"id": "en-ok", "index": 1},
            "en-bad": {"id": "en-bad", "index": 2},
            "zh-ok": {"id": "zh-ok", "index": 3},
        }
        job.lid_report_samples = [
            {
                "id": "en-ok",
                "index": 1,
                "duration_seconds": 1.0,
                "reference_language": "en",
                "predicted_language": "en",
                "raw_language": "en",
                "confidence": 0.9,
                "correct": True,
            },
            {
                "id": "en-bad",
                "index": 2,
                "duration_seconds": 1.0,
                "reference_language": "en",
                "predicted_language": "zh",
                "raw_language": "zh",
                "confidence": 0.8,
                "correct": False,
            },
            {
                "id": "zh-ok",
                "index": 3,
                "duration_seconds": 1.0,
                "reference_language": "zh",
                "predicted_language": "zh",
                "raw_language": "zh",
                "confidence": 0.9,
                "correct": True,
            },
        ]
        _put_job(job)

        response = client.post(
            f"/api/evaluations/{job.job_id}/metrics/recalculate",
            json={"excluded_sample_ids": []},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["accuracy"], 2 / 3)
        self.assertEqual(payload["recall"], 0.75)

    def test_lid_metrics_use_strict_language_labels(self) -> None:
        report = _build_lid_report(
            [
                {
                    "id": "strict-mismatch",
                    "reference_language": "cn",
                    "predicted_language": "zh",
                    "confidence": 0.99,
                    "correct": False,
                },
                {
                    "id": "strict-match",
                    "reference_language": "en",
                    "predicted_language": "en",
                    "confidence": 0.1,
                    "correct": True,
                },
            ]
        )

        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["recall"], 0.5)
        self.assertFalse(report["lid_report"]["samples"][0]["correct"])
        self.assertEqual(
            report["lid_report"]["samples"][0]["predicted_language"],
            "zh",
        )

    def test_sqa_request_requires_target_when_enabled(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationRequest(task="asr", enable_mos=True, mos_target="")

        with self.assertRaises(ValueError):
            EvaluationRequest(
                task="asr",
                enable_snr=True,
                snr_target="",
            )

        request = EvaluationRequest(task="asr", enable_mos=False, enable_snr=False)
        self.assertEqual(request.mos_target, "")
        self.assertEqual(request.snr_target, "")

    def test_denoise_request_requires_mos_or_snr(self) -> None:
        with self.assertRaises(ValueError):
            EvaluationRequest(task="denoise", enable_mos=False, enable_snr=False)

        mos_request = EvaluationRequest(
            task="denoise",
            enable_mos=True,
            mos_target="mos:50111",
        )
        self.assertTrue(mos_request.enable_mos)

        snr_request = EvaluationRequest(
            task="denoise",
            enable_snr=True,
            snr_target="snr:50112",
        )
        self.assertTrue(snr_request.enable_snr)

        with self.assertRaises(ValueError):
            EvaluationRequest(task="denoise", enable_mos=True, mos_target="")

    def test_denoise_report_averages_score_deltas(self) -> None:
        report = _build_denoise_report(
            [
                {
                    "id": "first",
                    "original_snr": 10.0,
                    "denoised_snr": 13.0,
                    "snr_delta": 3.0,
                    "original_mos": 3.0,
                    "denoised_mos": 3.4,
                    "mos_delta": 0.4,
                },
                {
                    "id": "second",
                    "original_snr": 7.0,
                    "denoised_snr": 8.0,
                    "snr_delta": 1.0,
                    "original_mos": 2.0,
                    "denoised_mos": 2.6,
                    "mos_delta": 0.6,
                },
                {"id": "failed", "error": "denoise failed"},
            ]
        )

        self.assertEqual(report["mean_snr_delta"], 2.0)
        self.assertEqual(report["mean_mos_delta"], 0.5)
        self.assertEqual(report["scored_snr_sample_count"], 2)
        self.assertEqual(report["scored_mos_sample_count"], 2)
        self.assertEqual(report["failed_sample_count"], 1)

    def test_sqa_summary_averages_successes_and_counts_failures(self) -> None:
        summary = _build_sqa_summary(
            [
                {
                    "id": "ok",
                    "sqa_scores": [
                        {
                            "engine_name": "MOS",
                            "target": "127.0.0.1:50111",
                            "score": 4.0,
                            "error": None,
                        },
                        {
                            "engine_name": "SNR",
                            "target": "127.0.0.1:50112",
                            "score": None,
                            "error": "unavailable",
                        },
                    ],
                },
                {
                    "id": "bad",
                    "sqa_scores": [
                        {
                            "engine_name": "MOS",
                            "target": "127.0.0.1:50111",
                            "score": 2.0,
                            "error": None,
                        }
                    ],
                },
            ]
        )

        self.assertEqual(summary[0]["engine_name"], "MOS")
        self.assertEqual(summary[0]["mean_score"], 3.0)
        self.assertEqual(summary[0]["scored_count"], 2)
        self.assertEqual(summary[0]["failed_count"], 0)
        self.assertEqual(summary[1]["engine_name"], "SNR")
        self.assertIsNone(summary[1]["mean_score"])
        self.assertEqual(summary[1]["scored_count"], 0)
        self.assertEqual(summary[1]["failed_count"], 1)

    def test_sqa_assessor_records_success_and_failure_per_sample(self) -> None:
        class FakeSqaResult:
            def __init__(self, score: float) -> None:
                self.score = score

        class FakeSqaInferencer:
            def __init__(self, target: str, **_: object) -> None:
                self.target = target

            def infer(self, audio: np.ndarray) -> FakeSqaResult:
                if self.target == "bad:50111":
                    raise RuntimeError("sqa failed")
                return FakeSqaResult(score=float(len(audio)))

            def close(self) -> None:
                return None

        request = EvaluationRequest(
            task="asr",
            enable_mos=True,
            mos_target="ok:50111",
            enable_snr=True,
            snr_target="bad:50111",
            sqa_inference_concurrency=2,
        )
        with (
            patch("prama_server.servicer.http.SqaGrpcInferencer", FakeSqaInferencer),
            patch("prama_server.servicer.http.logger.exception"),
        ):
            assessor = SqaAssessor(request)
            scores = assessor.assess(np.zeros(160, dtype=np.float32))
            assessor.close()

        self.assertEqual(scores[0]["engine_name"], "MOS")
        self.assertEqual(scores[0]["score"], 160.0)
        self.assertIsNone(scores[0]["error"])
        self.assertEqual(scores[1]["engine_name"], "SNR")
        self.assertIsNone(scores[1]["score"])
        self.assertIn("sqa failed", scores[1]["error"])

    def test_engine_connectivity_endpoint_reports_success_and_failure(self) -> None:
        class FakeChannel:
            def close(self) -> None:
                return None

        class FakeFuture:
            def __init__(self, should_fail: bool = False) -> None:
                self.should_fail = should_fail

            def result(self, timeout: float) -> None:
                if self.should_fail:
                    raise TimeoutError(f"timeout={timeout}")

        def fake_ready_future(channel: FakeChannel) -> FakeFuture:
            return FakeFuture(should_fail=getattr(channel, "should_fail", False))

        def fake_channel(target: str) -> FakeChannel:
            channel = FakeChannel()
            channel.should_fail = target == "bad:50111"  # type: ignore[attr-defined]
            return channel

        with (
            patch("prama_server.servicer.http.grpc.insecure_channel", fake_channel),
            patch("prama_server.servicer.http.grpc.channel_ready_future", fake_ready_future),
        ):
            ok_response = client.post(
                "/api/engines/connectivity",
                json={"target": "ok:50111", "timeout_seconds": 1},
            )
            bad_response = client.post(
                "/api/engines/connectivity",
                json={"target": "bad:50111", "timeout_seconds": 1},
            )

        self.assertEqual(ok_response.status_code, 200)
        self.assertTrue(ok_response.json()["ok"])
        self.assertEqual(bad_response.status_code, 200)
        self.assertFalse(bad_response.json()["ok"])

    def test_help_endpoint_returns_markdown_with_code_blocks(self) -> None:
        response = client.get("/api/help")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["title"], "数据集格式说明")
        self.assertIn("```jsonl", payload["markdown"])
        self.assertIn("ASR 数据集", payload["markdown"])
        self.assertIn("VAD 数据集", payload["markdown"])
        self.assertIn("LID 数据集", payload["markdown"])
        self.assertIn("SE 评估数据集", payload["markdown"])

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
