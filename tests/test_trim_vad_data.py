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
                    "audio/valid_0001_41.365_42.207_聊天呢.wav",
                    "audio/valid_0002_no_csv.wav",
                ],
            )
            self.assertEqual(metadata_rows[0]["id"], "valid_0001_41.365_42.207_聊天呢")
            self.assertEqual(metadata_rows[0]["seconds"]["starts"], [0.2])
            self.assertEqual(metadata_rows[0]["seconds"]["durations"], [0.5])
            self.assertEqual(metadata_rows[1]["seconds"], {"starts": [], "durations": []})

            first_audio, first_sample_rate = sf.read(
                result.metadata_path.parent / metadata_rows[0]["file_name"]
            )
            self.assertEqual(first_sample_rate, output_sample_rate)
            self.assertEqual(first_audio.ndim, 1)
            self.assertEqual(len(first_audio), output_sample_rate)

            dataset = load_dataset("audiofolder", data_dir=str(result.output), split="test")
            self.assertEqual(len(dataset), 2)
            self.assertIn("audio", dataset.column_names)
            self.assertIn("seconds", dataset.column_names)


if __name__ == "__main__":
    unittest.main()
