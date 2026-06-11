from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from prama_server.utils.trim import trim_vad_dataset


class TrimVadDatasetTest(unittest.TestCase):
    def test_fixed_chunks_trim_audio_and_labels_together(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "input"
            split_dir = dataset / "test"
            split_dir.mkdir(parents=True)
            sample_rate = 16000
            audio = np.zeros(sample_rate, dtype=np.float32)
            sf.write(split_dir / "sample.wav", audio, sample_rate)
            (split_dir / "metadata.jsonl").write_text(
                json.dumps(
                    {
                        "file_name": "sample.wav",
                        "id": "sample",
                        "seconds": {
                            "starts": [0.2],
                            "durations": [0.5],
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = trim_vad_dataset(
                dataset_path=dataset,
                split="test",
                output=root / "output",
                chunk_seconds=0.5,
                overlap_seconds=0.0,
                sample_rate=sample_rate,
                overwrite=False,
            )

            self.assertEqual(result.input_sample_count, 1)
            self.assertEqual(result.output_sample_count, 2)
            metadata_rows = [
                json.loads(line)
                for line in result.metadata_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [row["id"] for row in metadata_rows],
                ["sample__part_0001", "sample__part_0002"],
            )
            self.assertEqual(
                [row["file_name"] for row in metadata_rows],
                [
                    "audio/sample__part_0001.wav",
                    "audio/sample__part_0002.wav",
                ],
            )
            self.assertEqual(metadata_rows[0]["seconds"]["starts"], [0.2])
            self.assertEqual(metadata_rows[0]["seconds"]["durations"], [0.3])
            self.assertEqual(metadata_rows[1]["seconds"]["starts"], [0.0])
            self.assertEqual(metadata_rows[1]["seconds"]["durations"], [0.2])

            for row in metadata_rows:
                chunk_audio, chunk_sample_rate = sf.read(
                    result.metadata_path.parent / row["file_name"]
                )
                self.assertEqual(chunk_sample_rate, sample_rate)
                self.assertEqual(len(chunk_audio), sample_rate // 2)


if __name__ == "__main__":
    unittest.main()
