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
                        "summary": {"wer": 50.0},
                        "tokens": [
                            {"ref": "HELLO", "hyp": "HELLO"},
                            {"ref": "WORLD", "hyp": "WORD"},
                        ],
                    }
                ]
            },
            "cer_report": {
                "utterances": [{"id": "utt1", "summary": {"wer": 25.0}}]
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "asr.tsv"
            write_task_tsv(task="asr", result=result, output_path=output)
            fieldnames, rows = _read_tsv(output)

        self.assertEqual(
            fieldnames,
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
        self.assertEqual(rows[0]["reference"], "HELLO WORLD")
        self.assertEqual(rows[0]["hypothesis"], "HELLO WORD")

    def test_vad_tsv_keeps_standard_precision_recall_and_f1_columns(self) -> None:
        result = {
            "vad_report": {
                "samples": [
                    {
                        "index": 1,
                        "id": "vad1",
                        "audio_file": "/tmp/vad1.wav",
                        "duration_seconds": 1.0,
                        "metrics": {
                            "frame_accuracy": 0.8,
                            "frame_precision": 0.7,
                            "frame_recall": 0.6,
                            "frame_f1": 0.646,
                            "segment_precision": 0.5,
                            "segment_recall": 0.4,
                            "segment_f1": 0.444,
                            "reference_segment_count": 2,
                            "prediction_segment_count": 3,
                            "segment_miss_count": 1,
                            "segment_false_alarm_count": 2,
                        },
                    }
                ]
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "vad.tsv"
            write_task_tsv(task="vad", result=result, output_path=output)
            fieldnames, rows = _read_tsv(output)

        self.assertIn("frame_accuracy", fieldnames)
        self.assertIn("frame_precision", fieldnames)
        self.assertIn("frame_recall", fieldnames)
        self.assertIn("segment_precision", fieldnames)
        self.assertIn("segment_recall", fieldnames)
        self.assertEqual(rows[0]["frame_precision"], "0.7")


def _read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


if __name__ == "__main__":
    unittest.main()
