from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from prama_server.prama_cli.app import write_task_tsv


class PramaCliTsvTest(unittest.TestCase):
    def test_write_asr_tsv(self) -> None:
        result = {
            "wer_report": {
                "utterances": [
                    {
                        "index": 1,
                        "id": "utt1",
                        "audio_file": "/tmp/utt1.wav",
                        "duration_seconds": 1.25,
                        "summary": {"wer": 0.5},
                        "tokens": [
                            {"ref": "HELLO", "hyp": "HELLO"},
                            {"ref": "WORLD", "hyp": "WORD"},
                        ],
                    }
                ]
            },
            "cer_report": {
                "utterances": [
                    {
                        "id": "utt1",
                        "summary": {"wer": 0.25},
                    }
                ]
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "asr.tsv"
            write_task_tsv(task="asr", result=result, output_path=output)
            rows = _read_tsv(output)

        self.assertEqual(
            list(rows[0]),
            [
                "index",
                "id",
                "audio_file",
                "duration_seconds",
                "wer",
                "cer",
                "reference",
                "hypothesis",
            ],
        )
        self.assertEqual(rows[1]["id"], "utt1")
        self.assertEqual(rows[1]["reference"], "HELLO WORLD")
        self.assertEqual(rows[1]["hypothesis"], "HELLO WORD")

    def test_write_vad_lid_denoise_and_keyword_tsv(self) -> None:
        cases = {
            "vad": {
                "vad_report": {
                    "samples": [
                        {
                            "index": 1,
                            "id": "vad1",
                            "audio_file": "/tmp/vad1.wav",
                            "duration_seconds": 1.0,
                            "metrics": {
                                "frame_accuracy": 1.0,
                                "frame_precision": 1.0,
                                "frame_recall": 1.0,
                                "frame_f1": 1.0,
                                "segment_precision": 1.0,
                                "segment_recall": 1.0,
                                "segment_f1": 1.0,
                                "reference_segment_count": 1,
                                "prediction_segment_count": 1,
                                "segment_miss_count": 0,
                                "segment_false_alarm_count": 0,
                            },
                        }
                    ]
                }
            },
            "lid": {
                "lid_report": {
                    "samples": [
                        {
                            "index": 1,
                            "id": "lid1",
                            "audio_file": "/tmp/lid1.wav",
                            "duration_seconds": 1.0,
                            "reference_language": "en",
                            "predicted_language": "en",
                            "raw_language": "en",
                            "confidence": 0.9,
                            "correct": True,
                        }
                    ]
                }
            },
            "denoise": {
                "denoise_report": {
                    "samples": [
                        {
                            "index": 1,
                            "id": "se1",
                            "audio_file": "/tmp/se1.wav",
                            "denoised_audio_file": "/tmp/se1-denoised.wav",
                            "duration_seconds": 1.0,
                            "error": None,
                            "original_snr": 1.0,
                            "denoised_snr": 2.0,
                            "snr_delta": 1.0,
                            "original_mos": 3.0,
                            "denoised_mos": 3.5,
                            "mos_delta": 0.5,
                        }
                    ]
                }
            },
            "keyword": {
                "keyword_report": {
                    "samples": [
                        {
                            "index": 1,
                            "id": "kw1",
                            "audio_file": "/tmp/kw1.wav",
                            "keyword": "LEVEL",
                            "expected_hit": True,
                            "predicted_hit": True,
                            "correct": True,
                            "transcript": "LEVEL",
                            "match_text": "level",
                        }
                    ]
                }
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            for task, result in cases.items():
                output = Path(temp_dir) / f"{task}.tsv"
                write_task_tsv(task=task, result=result, output_path=output)
                rows = _read_tsv(output)
                self.assertEqual(rows[1]["index"], "1")
                self.assertTrue(rows[1]["id"])


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        return [dict.fromkeys(reader.fieldnames or [])] + list(reader)


if __name__ == "__main__":
    unittest.main()
