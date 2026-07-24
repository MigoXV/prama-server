from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from prama_server.utils.trim import trim_vad_dataset
from prama_server.utils.vad_select import select_vad_dataset


class TrimVadDatasetTest(unittest.TestCase):
    def test_flat_wav_csv_directory_converts_to_chunked_audiofolder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "raw"
            source.mkdir()
            sf.write(source / "sample.wav", np.zeros(8000, dtype=np.float32), 8000)
            (source / "sample.csv").write_text(
                "\n".join(
                    [
                        "Name\tStart\tDuration\tTime Format\tType\tDescription",
                        "0\t0:00.200\t0:00.500\tdecimal\tCue\t",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = trim_vad_dataset(
                dataset_path=source,
                output=root / "audiofolder",
                chunk_seconds=0.5,
                sample_rate=16000,
                show_progress=False,
            )

            rows = _read_jsonl(result.metadata_path)
            self.assertEqual(result.output_sample_count, 2)
            self.assertEqual(rows[0]["seconds"]["starts"], [0.2])
            self.assertEqual(rows[0]["seconds"]["durations"], [0.3])
            self.assertEqual(rows[1]["seconds"]["starts"], [0.0])
            self.assertEqual(rows[1]["seconds"]["durations"], [0.2])
            self.assertTrue(
                (result.output / "test/audio/sample__part_0001.csv").is_file()
            )

    def test_trim_failure_does_not_replace_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "raw"
            source.mkdir()
            sf.write(source / "sample.wav", np.zeros(80, dtype=np.float32), 8000)
            output = root / "output"
            output.mkdir()
            marker = output / "marker.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                trim_vad_dataset(
                    dataset_path=source,
                    output=output,
                    chunk_seconds=0.5,
                    overwrite=True,
                    show_progress=False,
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


class VadSelectDatasetTest(unittest.TestCase):
    def test_metric_ranges_create_selected_audiofolder_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = _write_vad_audiofolder(root)
            result_json = root / "vad-result.json"
            result_json.write_text(
                json.dumps(
                    {
                        "vad_report": {
                            "samples": [
                                {
                                    "id": "a",
                                    "metrics": {
                                        "frame_recall": 0.9,
                                        "segment_f1": 0.5,
                                    },
                                },
                                {
                                    "id": "b",
                                    "metrics": {
                                        "frame_recall": 0.4,
                                        "segment_f1": 0.9,
                                    },
                                },
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            output = root / "selected"

            result = select_vad_dataset(
                result_json=result_json,
                dataset_path=source,
                output=output,
                metric_ranges={
                    "frame_recall": (0.8, None),
                    "segment_f1": (None, 0.6),
                },
            )

            self.assertEqual(result.selected_sample_count, 1)
            self.assertEqual([row["id"] for row in _read_jsonl(result.metadata_path)], ["a"])
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["output"], str(output))
            self.assertEqual(summary["selected_ids"], ["a"])

    def test_invalid_metric_range_does_not_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = _write_vad_audiofolder(root)
            result_json = root / "vad-result.json"
            result_json.write_text(
                '{"vad_report":{"samples":[]}}',
                encoding="utf-8",
            )
            output = root / "selected"

            with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                select_vad_dataset(
                    result_json=result_json,
                    dataset_path=source,
                    output=output,
                    metric_ranges={"frame_recall": (1.1, None)},
                )

            self.assertFalse(output.exists())


def _write_vad_audiofolder(root: Path) -> Path:
    source = root / "source"
    audio_dir = source / "test/audio"
    audio_dir.mkdir(parents=True)
    sf.write(audio_dir / "a.wav", np.zeros(160, dtype=np.float32), 16000)
    sf.write(audio_dir / "b.wav", np.zeros(160, dtype=np.float32), 16000)
    (source / "test/metadata.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "file_name": "audio/a.wav",
                        "id": "a",
                        "seconds": {"starts": [0.0], "durations": [0.01]},
                    }
                ),
                json.dumps(
                    {
                        "file_name": "audio/b.wav",
                        "id": "b",
                        "seconds": {"starts": [], "durations": []},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return source


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    unittest.main()
