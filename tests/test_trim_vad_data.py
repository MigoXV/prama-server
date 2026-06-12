from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset

from prama_server.utils.trim import trim_vad_dataset


class TrimVadDatasetTest(unittest.TestCase):
    def test_flat_wav_csv_directory_converts_to_vad_audiofolder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "test01"
            source.mkdir()
            input_sample_rate = 8000
            output_sample_rate = 16000

            sf.write(
                source / "valid_0001_41.365_42.207_聊天呢.wav",
                np.zeros(input_sample_rate, dtype=np.float32),
                input_sample_rate,
            )
            (source / "valid_0001_41.365_42.207_聊天呢.csv").write_text(
                "\n".join(
                    [
                        "Name\tStart\tDuration\tTime Format\tType\tDescription",
                        "0\t0:00.200\t0:00.500\tdecimal\tCue\t",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            sf.write(
                source / "valid_0002_no_csv.wav",
                np.zeros(input_sample_rate // 2, dtype=np.float32),
                input_sample_rate,
            )

            result = trim_vad_dataset(
                dataset_path=source,
                split="test",
                output=None,
                chunk_seconds=0.5,
                overlap_seconds=0.0,
                sample_rate=output_sample_rate,
                overwrite=False,
                show_progress=False,
            )

            self.assertEqual(result.output, root / "test01-audiofolder")
            self.assertEqual(result.input_sample_count, 2)
            self.assertEqual(result.output_sample_count, 2)
            metadata_rows = [
                json.loads(line)
                for line in result.metadata_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(metadata_rows), 2)
            self.assertEqual(
                [row["file_name"] for row in metadata_rows],
                [
                    "audio/valid_0001_41.365_42.207_聊天呢__part_0001.wav",
                    "audio/valid_0001_41.365_42.207_聊天呢__part_0002.wav",
                ],
            )
            self.assertEqual(
                metadata_rows[0]["id"],
                "valid_0001_41.365_42.207_聊天呢__part_0001",
            )
            self.assertEqual(metadata_rows[0]["seconds"]["starts"], [0.2])
            self.assertEqual(metadata_rows[0]["seconds"]["durations"], [0.3])
            self.assertEqual(metadata_rows[1]["seconds"]["starts"], [0.0])
            self.assertEqual(metadata_rows[1]["seconds"]["durations"], [0.2])
            self.assertTrue(
                (
                    result.metadata_path.parent
                    / "audio/valid_0001_41.365_42.207_聊天呢__part_0001.csv"
                ).exists()
            )
            self.assertTrue(
                (
                    result.metadata_path.parent
                    / "audio/valid_0001_41.365_42.207_聊天呢__part_0002.csv"
                ).exists()
            )
            self.assertFalse(
                (
                    result.metadata_path.parent
                    / "audio/valid_0002_no_csv__part_0001.wav"
                ).exists()
            )

            first_audio, first_sample_rate = sf.read(
                result.metadata_path.parent / metadata_rows[0]["file_name"]
            )
            self.assertEqual(first_sample_rate, output_sample_rate)
            self.assertEqual(first_audio.ndim, 1)
            self.assertEqual(len(first_audio), output_sample_rate // 2)

            dataset = load_dataset("audiofolder", data_dir=str(result.output), split="test")
            self.assertEqual(len(dataset), 2)
            self.assertIn("audio", dataset.column_names)
            self.assertIn("seconds", dataset.column_names)

    def test_jsonl_takes_priority_over_csv_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "test01"
            audio_dir = source / "audio"
            audio_dir.mkdir(parents=True)
            sample_rate = 16000

            sf.write(
                audio_dir / "from_jsonl.wav",
                np.zeros(sample_rate, dtype=np.float32),
                sample_rate,
            )
            sf.write(
                source / "from_csv.wav",
                np.zeros(sample_rate, dtype=np.float32),
                sample_rate,
            )
            (source / "from_csv.csv").write_text(
                "\n".join(
                    [
                        "Name\tStart\tDuration\tTime Format\tType\tDescription",
                        "0\t0:00.100\t0:00.200\tdecimal\tCue\t",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (source / "metadata.jsonl").write_text(
                json.dumps(
                    {
                        "file_name": "audio/from_jsonl.wav",
                        "id": "jsonl_sample",
                        "seconds": {
                            "starts": [0.25],
                            "durations": [0.25],
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = trim_vad_dataset(
                dataset_path=source,
                split="test",
                output=None,
                chunk_seconds=0.5,
                overlap_seconds=0.0,
                sample_rate=sample_rate,
                overwrite=False,
                show_progress=False,
            )

            self.assertEqual(result.input_sample_count, 1)
            self.assertEqual(result.output_sample_count, 2)
            metadata_rows = [
                json.loads(line)
                for line in result.metadata_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["file_name"] for row in metadata_rows],
                [
                    "audio/jsonl_sample__part_0001.wav",
                    "audio/jsonl_sample__part_0002.wav",
                ],
            )
            self.assertEqual(metadata_rows[0]["seconds"]["starts"], [0.25])
            self.assertEqual(metadata_rows[0]["seconds"]["durations"], [0.25])
            self.assertEqual(metadata_rows[1]["seconds"]["starts"], [])
            self.assertEqual(metadata_rows[1]["seconds"]["durations"], [])
            self.assertFalse(
                (result.metadata_path.parent / "audio/from_csv__part_0001.wav").exists()
            )


if __name__ == "__main__":
    unittest.main()
