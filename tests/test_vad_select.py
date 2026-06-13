from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset
from datasets.exceptions import DatasetGenerationError
from typer.testing import CliRunner

from prama_server.utils.vad_select.app import app as vad_select_app
from prama_server.utils.vad_select import select_vad_dataset


runner = CliRunner()


class VadSelectDatasetTest(unittest.TestCase):
    def test_default_ranges_copy_all_samples_to_loadable_audiofolder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_source_dataset(root)
            result_json = self._write_result_json(root, ["sample_a", "sample_b"])
            output = root / "selected"

            result = select_vad_dataset(
                result_json=result_json,
                dataset_path=source,
                output=output,
                overwrite=False,
            )

            self.assertEqual(result.total_sample_count, 2)
            self.assertEqual(result.selected_sample_count, 2)
            rows = self._read_jsonl(output / "test/metadata.jsonl")
            self.assertEqual([row["id"] for row in rows], ["sample_a", "sample_b"])
            self.assertEqual([row["file_name"] for row in rows], ["audio/a.wav", "audio/b.wav"])
            self.assertTrue((output / "test/audio/a.wav").exists())
            self.assertTrue((output / "test/audio/b.wav").exists())

            dataset = self._load_audiofolder_or_skip(output)
            self.assertEqual(len(dataset), 2)
            self.assertIn("audio", dataset.column_names)
            self.assertIn("seconds", dataset.column_names)

    def test_metric_ranges_filter_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_source_dataset(root)
            result_json = self._write_result_json(root, ["sample_a", "sample_b"])
            output = root / "selected"

            result = select_vad_dataset(
                result_json=result_json,
                dataset_path=source,
                output=output,
                metric_ranges={"frame_recall": (0.8, None), "segment_f1": (None, 0.6)},
            )

            self.assertEqual(result.total_sample_count, 2)
            self.assertEqual(result.selected_sample_count, 1)
            rows = self._read_jsonl(output / "test/metadata.jsonl")
            self.assertEqual([row["id"] for row in rows], ["sample_a"])
            summary = json.loads((output / "test/selection_summary.json").read_text())
            self.assertEqual(summary["selected_ids"], ["sample_a"])
            self.assertEqual(
                summary["enabled_metric_ranges"],
                {
                    "frame_recall": {"min": 0.8, "max": None},
                    "segment_f1": {"min": None, "max": 0.6},
                },
            )

    def test_snapshot_result_shape_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_source_dataset(root)
            result_json = root / "snapshot.json"
            result_json.write_text(
                json.dumps(
                    {
                        "job_id": "vad-job",
                        "result": {
                            "vad_report": {
                                "samples": [self._sample_payload("sample_b", 0.5, 0.9)]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            select_vad_dataset(
                result_json=result_json,
                dataset_path=source,
                output=root / "selected",
            )

            rows = self._read_jsonl(root / "selected/test/metadata.jsonl")
            self.assertEqual([row["id"] for row in rows], ["sample_b"])

    def test_missing_result_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_source_dataset(root)
            result_json = self._write_result_json(root, ["missing"])

            with self.assertRaisesRegex(KeyError, "不存在"):
                select_vad_dataset(
                    result_json=result_json,
                    dataset_path=source,
                    output=root / "selected",
                )

    def test_duplicate_metadata_id_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_source_dataset(root)
            metadata_path = source / "test/metadata.jsonl"
            first_line = metadata_path.read_text(encoding="utf-8").splitlines()[0]
            metadata_path.write_text(first_line + "\n" + first_line + "\n", encoding="utf-8")
            result_json = self._write_result_json(root, ["sample_a"])

            with self.assertRaisesRegex(ValueError, "重复 id"):
                select_vad_dataset(
                    result_json=result_json,
                    dataset_path=source,
                    output=root / "selected",
                )

    def test_missing_audio_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_source_dataset(root)
            (source / "test/audio/a.wav").unlink()
            result_json = self._write_result_json(root, ["sample_a"])

            with self.assertRaisesRegex(FileNotFoundError, "音频文件不存在"):
                select_vad_dataset(
                    result_json=result_json,
                    dataset_path=source,
                    output=root / "selected",
                )

    def test_cli_metric_options_override_file_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = self._write_source_dataset(root)
            result_json = self._write_result_json(root, ["sample_a", "sample_b"])
            output = root / "selected"

            result = runner.invoke(
                vad_select_app,
                [
                    "--result-json",
                    str(result_json),
                    "--dataset-path",
                    str(source),
                    "--output",
                    str(output),
                    "--min-frame-recall",
                    "0.8",
                    "--max-segment-f1",
                    "0.6",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            rows = self._read_jsonl(output / "test/metadata.jsonl")
            self.assertEqual([row["id"] for row in rows], ["sample_a"])
            summary = json.loads((output / "test/selection_summary.json").read_text())
            self.assertEqual(
                summary["enabled_metric_ranges"],
                {
                    "frame_recall": {"min": 0.8, "max": None},
                    "segment_f1": {"min": None, "max": 0.6},
                },
            )

    def _write_source_dataset(self, root: Path) -> Path:
        source = root / "source"
        audio_dir = source / "test/audio"
        audio_dir.mkdir(parents=True)
        sample_rate = 16000
        sf.write(audio_dir / "a.wav", np.zeros(sample_rate // 2, dtype=np.float32), sample_rate)
        sf.write(audio_dir / "b.wav", np.zeros(sample_rate // 2, dtype=np.float32), sample_rate)
        rows = [
            {
                "file_name": "audio/a.wav",
                "id": "sample_a",
                "seconds": {"starts": [0.1], "durations": [0.2]},
            },
            {
                "file_name": "audio/b.wav",
                "id": "sample_b",
                "seconds": {"starts": [], "durations": []},
            },
        ]
        (source / "test/metadata.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        return source

    def _write_result_json(self, root: Path, sample_ids: list[str]) -> Path:
        payload = {
            "vad_report": {
                "samples": [
                    self._sample_payload(sample_id, 0.9, 0.5)
                    if sample_id != "sample_b"
                    else self._sample_payload(sample_id, 0.7, 0.9)
                    for sample_id in sample_ids
                ]
            }
        }
        result_json = root / "vad-result.json"
        result_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return result_json

    def _sample_payload(self, sample_id: str, frame_recall: float, segment_f1: float) -> dict:
        return {
            "id": sample_id,
            "metrics": {
                "frame_recall": frame_recall,
                "frame_precision": 0.8,
                "frame_f1": 0.75,
                "segment_recall": 0.8,
                "segment_precision": 0.7,
                "segment_f1": segment_f1,
                "frame_false_alarm_rate": 0.1,
                "segment_false_alarm_rate": 0.2,
            },
        }

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _load_audiofolder_or_skip(self, output: Path):
        try:
            return load_dataset("audiofolder", data_dir=str(output), split="test")
        except DatasetGenerationError as exc:
            if "torchcodec" in _exception_chain_text(exc):
                self.skipTest(f"当前环境缺少 audiofolder 运行时依赖: {exc}")
            raise


def _exception_chain_text(exc: BaseException) -> str:
    parts = []
    current: BaseException | None = exc
    while current is not None:
        parts.append(str(current))
        current = current.__cause__
    return "\n".join(parts)


if __name__ == "__main__":
    unittest.main()
