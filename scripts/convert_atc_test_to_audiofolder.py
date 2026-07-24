from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把 ATC-ASR-Dataset 的 test split 转成 Hugging Face audiofolder 格式。"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data-bin/jacktol/ATC-ASR-Dataset"),
        help="原始 Hugging Face 数据集目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data-bin/cdb-asr-test"),
        help="输出 audiofolder 数据集目录",
    )
    parser.add_argument("--split", default="test", help="要转换的数据集 split")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出目录存在则先删除",
    )
    args = parser.parse_args()

    convert_split(
        source=args.source,
        output=args.output,
        split=args.split,
        overwrite=args.overwrite,
    )


def convert_split(*, source: Path, output: Path, split: str, overwrite: bool) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在，请使用 --overwrite: {output}")
        logger.info("删除已有输出目录: %s", output)
        shutil.rmtree(output)

    split_dir = output / split
    audio_dir = split_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = split_dir / "metadata.jsonl"

    logger.info("加载数据集: source=%s split=%s", source, split)
    dataset = load_dataset(str(source), split=split)
    logger.info("开始转换: rows=%s output=%s", len(dataset), output)

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for index, sample in enumerate(dataset, start=1):
            sample_id = str(sample["id"])
            audio_array, sample_rate = _decode_audio(sample["audio"])
            audio_name = f"{_safe_stem(sample_id, fallback=f'{index:06d}')}.wav"
            audio_path = audio_dir / audio_name
            sf.write(audio_path, audio_array, sample_rate)
            metadata_file.write(
                json.dumps(
                    {
                        "file_name": f"audio/{audio_name}",
                        "id": sample_id,
                        "text": str(sample["text"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if index % 100 == 0:
                logger.info("已转换 %s/%s", index, len(dataset))

    logger.info("转换完成: dataset=%s metadata=%s", output, metadata_path)


def _decode_audio(audio: Any) -> tuple[np.ndarray, int]:
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        data = samples.data
        array = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
        sample_rate = int(samples.sample_rate)
    elif isinstance(audio, dict):
        array = np.asarray(audio["array"])
        sample_rate = int(audio["sampling_rate"])
    else:
        raise TypeError(f"不支持的音频类型: {type(audio)!r}")

    array = np.squeeze(array)
    if array.ndim == 2:
        channel_axis = 0 if array.shape[0] <= array.shape[1] else 1
        array = array.mean(axis=channel_axis)
    if array.ndim != 1:
        raise ValueError(f"不支持的音频维度: {array.shape}")
    return array.astype(np.float32, copy=False), sample_rate


def _safe_stem(value: str, *, fallback: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in value.strip()
    )
    return safe or fallback


if __name__ == "__main__":
    main()
