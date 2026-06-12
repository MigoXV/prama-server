from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import soundfile as sf
from fastapi.testclient import TestClient

from prama_server.servicer.http import (
    EvaluationJob,
    EvaluationRequest,
    SqaAssessor,
    _asr_word_accuracy_from_report,
    _build_keyword_report,
    _build_denoise_report,
    _build_sqa_summary,
    _build_vad_report,
    _build_cer_report,
    _build_wer_report,
    _build_lid_report,
    _load_vad_dataset,
    _keyword_matches,
    _load_keyword_dataset,
    _normalize_keyword_text,
    _run_keyword_evaluation,
    _lid_predicted_language,
    app,
    jobs,
    jobs_lock,
)
from prama_server.inferencers.grpc_options import (
    GRPC_CHANNEL_OPTIONS,
    GRPC_MAX_MESSAGE_LENGTH_BYTES,
)


client = TestClient(app)


class HttpReportMetricsTest(unittest.TestCase):
    def test_grpc_channel_options_allow_500mb_messages(self) -> None:
        self.assertEqual(GRPC_MAX_MESSAGE_LENGTH_BYTES, 500 * 1024 * 1024)
        self.assertIn(
            ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH_BYTES),
            GRPC_CHANNEL_OPTIONS,
        )
        self.assertIn(
            ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH_BYTES),
            GRPC_CHANNEL_OPTIONS,
        )

    def test_vad_dataset_loads_local_parquet_split_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir)
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, np.zeros(160, dtype=np.float32), 16000, format="WAV")
            pd.DataFrame(
                [
                    {
                        "audio": {"bytes": audio_bytes.getvalue(), "path": None},
                        "id": "sample-1",
                        "seconds": {"starts": [0.0], "durations": [0.01]},
                    }
                ]
            ).to_parquet(dataset_dir / "test-00000-of-00001.parquet")

            dataset = _load_vad_dataset(
                dataset_dir,
                split="test",
                limit=None,
                sample_rate=16000,
            )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["id"], "sample-1")
        self.assertIn("audio", dataset.column_names)
        self.assertEqual(dataset[0]["seconds"]["starts"][0], 0.0)

    def test_vad_report_weights_recall_by_reference_and_precision_by_prediction(
        self,
    ) -> None:
        eps = 1e-9
        rows = [
            self._vad_metric_row(
                frame_recall=0.8,
                frame_precision=0.5,
                frame_miss_rate=0.2,
                frame_false_alarm_rate=0.5,
                segment_recall=0.5,
                segment_precision=0.25,
                segment_miss_rate=0.5,
                segment_false_alarm_rate=0.75,
                reference_segment_count=2,
                prediction_segment_count=4,
                segment_hit_count=1,
                segment_miss_count=1,
                segment_false_alarm_count=3,
            ),
            self._vad_metric_row(
                frame_recall=0.0,
                frame_precision=0.1,
                frame_miss_rate=1.0,
                frame_false_alarm_rate=0.9,
                segment_recall=0.0,
                segment_precision=0.2,
                segment_miss_rate=0.0,
                segment_false_alarm_rate=0.8,
                reference_segment_count=0,
                prediction_segment_count=10,
                segment_hit_count=0,
                segment_miss_count=0,
                segment_false_alarm_count=8,
            ),
            self._vad_metric_row(
                frame_recall=0.25,
                frame_precision=0.0,
                frame_miss_rate=0.75,
                frame_false_alarm_rate=0.0,
                segment_recall=0.25,
                segment_precision=0.0,
                segment_miss_rate=0.75,
                segment_false_alarm_rate=0.0,
                reference_segment_count=8,
                prediction_segment_count=0,
                segment_hit_count=2,
                segment_miss_count=6,
                segment_false_alarm_count=0,
            ),
        ]

        report = _build_vad_report(rows, samples=[])

        reference_weights = [2 + eps, eps, 8 + eps]
        prediction_weights = [4 + eps, 10 + eps, eps]
        expected_frame_recall = self._weighted(
            [0.8, 0.0, 0.25],
            reference_weights,
        )
        expected_frame_precision = self._weighted(
            [0.5, 0.1, 0.0],
            prediction_weights,
        )
        expected_segment_recall = self._weighted(
            [0.5, 0.0, 0.25],
            reference_weights,
        )
        expected_segment_precision = self._weighted(
            [0.25, 0.2, 0.0],
            prediction_weights,
        )
        expected_segment_f1 = (
            2
            * expected_segment_precision
            * expected_segment_recall
            / (expected_segment_precision + expected_segment_recall)
        )

        self.assertAlmostEqual(report["frame_recall"], expected_frame_recall)
        self.assertAlmostEqual(report["frame"]["frame_recall"], expected_frame_recall)
        self.assertAlmostEqual(report["frame_precision"], expected_frame_precision)
        self.assertAlmostEqual(
            report["frame"]["frame_precision"],
            expected_frame_precision,
        )
        self.assertAlmostEqual(report["segment_recall"], expected_segment_recall)
        self.assertAlmostEqual(
            report["segment"]["segment_recall"],
            expected_segment_recall,
        )
        self.assertAlmostEqual(report["segment_precision"], expected_segment_precision)
        self.assertAlmostEqual(
            report["segment"]["segment_precision"],
            expected_segment_precision,
        )
        self.assertAlmostEqual(report["segment_f1"], expected_segment_f1)
        self.assertAlmostEqual(report["segment"]["segment_f1"], expected_segment_f1)
        self.assertEqual(report["reference_segment_count"], 10)
        self.assertEqual(report["prediction_segment_count"], 14)
        self.assertEqual(report["segment_hit_count"], 3)
        self.assertEqual(report["segment_miss_count"], 7)
        self.assertEqual(report["segment_false_alarm_count"], 11)

    def test_lid_metrics_use_open_set_metrics(self) -> None:
        payload = _build_lid_report(
            [
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
                    "id": "en-reject",
                    "index": 2,
                    "duration_seconds": 1.0,
                    "reference_language": "en",
                    "predicted_language": "<others>",
                    "raw_language": "en",
                    "confidence": 0.2,
                    "correct": False,
                },
                {
                    "id": "cn-bad",
                    "index": 3,
                    "duration_seconds": 1.0,
                    "reference_language": "cn",
                    "predicted_language": "en",
                    "raw_language": "en",
                    "confidence": 0.8,
                    "correct": False,
                },
                {
                    "id": "unknown-ok",
                    "index": 4,
                    "duration_seconds": 1.0,
                    "reference_language": "<others>",
                    "predicted_language": "<others>",
                    "raw_language": "<others>",
                    "confidence": 0.9,
                    "correct": True,
                },
                {
                    "id": "unknown-false",
                    "index": 5,
                    "duration_seconds": 1.0,
                    "reference_language": "<others>",
                    "predicted_language": "cn",
                    "raw_language": "cn",
                    "confidence": 0.9,
                    "correct": False,
                },
            ]
        )

        self.assertEqual(payload["accuracy"], 1 / 3)
        self.assertEqual(payload["precision"], 0.25)
        self.assertEqual(payload["known_accuracy"], 1 / 3)
        self.assertEqual(payload["macro_precision"], 0.25)
        self.assertEqual(payload["recall"], 0.25)
        self.assertEqual(payload["macro_recall"], 0.25)
        self.assertEqual(payload["known_correct_count"], 1)
        self.assertEqual(payload["known_sample_count"], 3)
        self.assertEqual(payload["overall_correct_count"], 2)
        self.assertEqual(payload["unknown_false_accept_count"], 1)
        self.assertEqual(payload["known_reject_count"], 1)
        self.assertEqual(
            {
                item["language"]: (
                    item["correct_count"],
                    item["sample_count"],
                    item["predicted_count"],
                    item["precision"],
                    item["recall"],
                )
                for item in payload["lid_language_recalls"]
            },
            {"cn": (0, 1, 1, 0.0, 0.0), "en": (1, 2, 2, 0.5, 0.5)},
        )
        confusion_rows = {
            row["reference_language"]: row["counts"]
            for row in payload["lid_confusion_matrix"]["rows"]
        }
        self.assertEqual(confusion_rows["en"]["en"], 1)
        self.assertEqual(confusion_rows["en"]["<others>"], 1)
        self.assertEqual(confusion_rows["cn"]["en"], 1)
        self.assertEqual(confusion_rows["<others>"]["<others>"], 1)
        self.assertEqual(confusion_rows["<others>"]["cn"], 1)

    def _vad_metric_row(self, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "frame_accuracy": 0.0,
            "frame_recall": 0.0,
            "frame_precision": 0.0,
            "frame_f1": 0.0,
            "frame_specificity": 0.0,
            "frame_false_alarm_rate": 0.0,
            "frame_miss_rate": 0.0,
            "frame_balanced_accuracy": 0.0,
            "frame_total": 0,
            "frame_speech": 0,
            "frame_non_speech": 0,
            "frame_true_positive": 0,
            "frame_true_negative": 0,
            "frame_false_positive": 0,
            "frame_false_negative": 0,
            "segment_recall": 0.0,
            "segment_precision": 0.0,
            "segment_f1": 0.0,
            "segment_miss_rate": 0.0,
            "segment_false_alarm_rate": 0.0,
            "reference_segment_count": 0,
            "prediction_segment_count": 0,
            "segment_hit_count": 0,
            "segment_miss_count": 0,
            "segment_false_alarm_count": 0,
        }
        row.update(overrides)
        return row

    def _weighted(self, values: list[float], weights: list[float]) -> float:
        return sum(value * weight for value, weight in zip(values, weights)) / sum(
            weights
        )

    def test_keyword_matching_normalizes_case_punctuation_and_word_boundaries(self) -> None:
        self.assertEqual(
            _normalize_keyword_text("Austrian, Seven!"),
            "austrian seven",
        )
        self.assertTrue(_keyword_matches("Austrian, seven papa", "AUSTRIAN"))
        self.assertTrue(_keyword_matches("Austrian seven papa", "AUSTRIAN SEVEN"))
        self.assertFalse(_keyword_matches("catch tower", "CAT"))
        self.assertTrue(_keyword_matches("我们刚刚上一段会议", "刚刚"))

    def test_keyword_report_counts_hits_misses_false_alarms_and_rejects(self) -> None:
        report = _build_keyword_report(
            [
                {"expected_hit": True, "predicted_hit": True},
                {"expected_hit": True, "predicted_hit": False},
                {"expected_hit": False, "predicted_hit": True},
                {"expected_hit": False, "predicted_hit": False},
            ]
        )

        self.assertEqual(report["hit_count"], 1)
        self.assertEqual(report["miss_count"], 1)
        self.assertEqual(report["false_alarm_count"], 1)
        self.assertEqual(report["correct_reject_count"], 1)
        self.assertEqual(report["positive_sample_count"], 2)
        self.assertEqual(report["negative_sample_count"], 2)
        self.assertEqual(report["accuracy"], 0.5)
        self.assertEqual(report["precision"], 0.5)
        self.assertEqual(report["recall"], 0.5)
        self.assertEqual(report["f1"], 0.5)

    def test_keyword_report_handles_empty_positive_and_negative_sets(self) -> None:
        self.assertEqual(_build_keyword_report([])["accuracy"], 0.0)

        positive_only = _build_keyword_report(
            [{"expected_hit": True, "predicted_hit": True}]
        )
        self.assertEqual(positive_only["precision"], 1.0)
        self.assertEqual(positive_only["recall"], 1.0)
        self.assertEqual(positive_only["negative_sample_count"], 0)

        negative_only = _build_keyword_report(
            [{"expected_hit": False, "predicted_hit": False}]
        )
        self.assertEqual(negative_only["accuracy"], 1.0)
        self.assertEqual(negative_only["precision"], 0.0)
        self.assertEqual(negative_only["recall"], 0.0)
        self.assertEqual(negative_only["positive_sample_count"], 0)

    def test_keyword_request_is_valid_and_sqa_target_validation_still_applies(self) -> None:
        request = EvaluationRequest(task="keyword")
        self.assertEqual(request.task, "keyword")

        with self.assertRaises(ValueError):
            EvaluationRequest(task="keyword", enable_mos=True, mos_target="")

    def test_keyword_dataset_requires_keyword_expected_hit_and_allows_multiple_keywords(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "keyword"
            test_dir = dataset_dir / "test"
            test_dir.mkdir(parents=True)
            sf.write(test_dir / "sample.wav", np.zeros(160, dtype=np.float32), 16000)
            metadata_path = test_dir / "metadata.jsonl"

            metadata_path.write_text(
                '{"file_name":"sample.wav","id":"missing","keyword":"AUSTRIAN"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "expected_hit"):
                _load_keyword_dataset(
                    dataset_dir,
                    split="test",
                    limit=None,
                    sample_rate=16000,
                )

            metadata_path.write_text(
                "\n".join(
                    [
                        '{"file_name":"sample.wav","id":"first","keyword":"AUSTRIAN","expected_hit":true}',
                        '{"file_name":"sample.wav","id":"second","keyword":"SPEED","expected_hit":false}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            dataset = _load_keyword_dataset(
                dataset_dir,
                split="test",
                limit=None,
                sample_rate=16000,
            )
            self.assertEqual(len(dataset), 2)

    def test_keyword_evaluation_reuses_asr_for_same_audio_keywords(self) -> None:
        class FakeAsrInferencer:
            calls: list[np.ndarray] = []

            def __init__(self, **kwargs: object) -> None:
                pass

            def infer(self, audio: np.ndarray):
                self.calls.append(audio)
                yield ("LEVEL SPEED", True)

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "keyword"
            test_dir = dataset_dir / "test"
            test_dir.mkdir(parents=True)
            sf.write(test_dir / "sample.wav", np.zeros(160, dtype=np.float32), 16000)
            (test_dir / "metadata.jsonl").write_text(
                (
                    '{"file_name":"sample.wav","id":"sample",'
                    '"keywords":['
                    '{"keyword":"LEVEL","expected_hit":true},'
                    '{"keyword":"ALTITUDE","expected_hit":false}'
                    "]}"
                )
                + "\n",
                encoding="utf-8",
            )
            job = EvaluationJob(
                job_id="keyword-group-test",
                request=EvaluationRequest(
                    task="keyword",
                    dataset_path=str(dataset_dir),
                    split="test",
                    sample_rate=16000,
                ),
            )

            with patch(
                "prama_server.servicer.http.AsrGrpcInferencer",
                FakeAsrInferencer,
            ):
                _run_keyword_evaluation(job)

            self.assertEqual(len(FakeAsrInferencer.calls), 1)
            self.assertEqual(job.status, "completed")
            self.assertIsNotNone(job.result)
            assert job.result is not None
            keyword_samples = job.result["keyword_report"]["samples"]
            audio_samples = job.result["keyword_audio_report"]["samples"]
            self.assertEqual(len(keyword_samples), 2)
            self.assertEqual(len(audio_samples), 1)
            self.assertEqual(job.result["sample_count"], 2)
            self.assertEqual(job.result["audio_sample_count"], 1)
            self.assertEqual(job.result["audio_duration_seconds"], 0.01)
            self.assertEqual(audio_samples[0]["id"], "sample")
            self.assertEqual(len(audio_samples[0]["keywords"]), 2)
            self.assertEqual(
                [sample["audio_id"] for sample in keyword_samples],
                ["sample", "sample"],
            )
            self.assertEqual(
                [sample["predicted_hit"] for sample in keyword_samples],
                [True, False],
            )

    def test_keyword_evaluation_keeps_different_audio_files_separate(self) -> None:
        class FakeAsrInferencer:
            calls: list[np.ndarray] = []

            def __init__(self, **kwargs: object) -> None:
                pass

            def infer(self, audio: np.ndarray):
                self.calls.append(audio)
                yield ("LEVEL", True)

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "keyword"
            test_dir = dataset_dir / "test"
            test_dir.mkdir(parents=True)
            sf.write(test_dir / "first.wav", np.zeros(160, dtype=np.float32), 16000)
            sf.write(test_dir / "second.wav", np.ones(160, dtype=np.float32), 16000)
            (test_dir / "metadata.jsonl").write_text(
                "\n".join(
                    [
                        '{"file_name":"first.wav","id":"first","keywords":[{"keyword":"LEVEL","expected_hit":true}]}',
                        '{"file_name":"second.wav","id":"second","keywords":[{"keyword":"LEVEL","expected_hit":true}]}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            job = EvaluationJob(
                job_id="keyword-separate-audio-test",
                request=EvaluationRequest(
                    task="keyword",
                    dataset_path=str(dataset_dir),
                    split="test",
                    sample_rate=16000,
                ),
            )

            with patch(
                "prama_server.servicer.http.AsrGrpcInferencer",
                FakeAsrInferencer,
            ):
                _run_keyword_evaluation(job)

            self.assertEqual(len(FakeAsrInferencer.calls), 2)
            self.assertIsNotNone(job.result)
            assert job.result is not None
            self.assertEqual(job.result["sample_count"], 2)
            self.assertEqual(job.result["audio_sample_count"], 2)
            self.assertEqual(
                [sample["id"] for sample in job.result["keyword_audio_report"]["samples"]],
                ["first", "second"],
            )

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
        self.assertEqual(report["precision"], 0.5)
        self.assertEqual(report["recall"], 0.5)
        self.assertFalse(report["lid_report"]["samples"][0]["correct"])
        self.assertEqual(
            report["lid_report"]["samples"][0]["predicted_language"],
            "zh",
        )

    def test_lid_prediction_uses_confidence_threshold_for_unknown_reject(self) -> None:
        self.assertEqual(_lid_predicted_language("en", 0.89, 0.9), "<others>")
        self.assertEqual(_lid_predicted_language("en", 0.9, 0.9), "en")

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
            patch("prama_server.servicer.http.create_insecure_channel", fake_channel),
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
        self.assertEqual(payload["title"], "数据集与评估指标说明")
        self.assertIn("```jsonl", payload["markdown"])
        self.assertIn("## 目录", payload["markdown"])
        self.assertIn("ASR 数据集", payload["markdown"])
        self.assertIn("VAD 数据集", payload["markdown"])
        self.assertIn("LID 数据集", payload["markdown"])
        self.assertIn("SE 评估数据集", payload["markdown"])
        self.assertIn("## 评估指标定义", payload["markdown"])
        self.assertIn("\\mathrm{Precision}_{\\mathrm{macro}}", payload["markdown"])

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

    def test_asr_word_accuracy_uses_prama_summary_accuracy(self) -> None:
        wer_report = _build_wer_report(
            [
                {
                    "id": "utt1",
                    "index": 1,
                    "reference": "hello world",
                    "hypothesis": "hello word",
                }
            ]
        )

        self.assertEqual(
            _asr_word_accuracy_from_report(wer_report),
            wer_report["summary"]["accuracy"],
        )

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
