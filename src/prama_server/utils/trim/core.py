from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from prama_server.utils.audition_formatter import mask_to_seconds, seconds_to_mask

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrimVadResult:
    output: Path
    split: str
    input_sample_count: int
    output_sample_count: int
    metadata_path: Path


def trim_vad_dataset(
    *,
    dataset_path: Path,
    split: str,
    output: Path,
    chunk_seconds: float,
    overlap_seconds: float = 0.0,
    sample_rate: int = 16000,
    overwrite: bool = False,
) -> TrimVadResult:
    if chunk_seconds <= 0:
        raise ValueError(f"chunk_seconds 必须大于 0: {chunk_seconds}")
    if overlap_seconds < 0:
        raise ValueError(f"overlap_seconds 必须大于等于 0: {overlap_seconds}")
    if overlap_seconds >= chunk_seconds:
        raise ValueError(
            "overlap_seconds 必须小于 chunk_seconds: "
            f"{overlap_seconds} >= {chunk_seconds}"
        )
    if sample_rate <= 0:
        raise ValueError(f"sample_rate 必须大于 0: {sample_rate}")

    dataset_path = dataset_path.resolve()
    output = output.resolve()
    if dataset_path == output:
        raise ValueError("输出目录不能与输入数据集目录相同")

    split_dir = dataset_path / split
    metadata_path = split_dir / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.jsonl 不存在: {metadata_path}")

    output_split_dir = output / split
    output_audio_dir = output_split_dir / "audio"
    output_metadata_path = output_split_dir / "metadata.jsonl"
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在，请使用 --overwrite: {output}")
        logger.info("删除已有输出目录: %s", output)
        shutil.rmtree(output)
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    samples = _read_metadata(metadata_path)
    chunk_size = max(1, int(round(chunk_seconds * sample_rate)))
    step_size = max(1, int(round((chunk_seconds - overlap_seconds) * sample_rate)))
    output_count = 0

    with output_metadata_path.open("w", encoding="utf-8") as metadata_file:
        for input_index, sample in enumerate(samples, start=1):
            sample_id = str(sample.get("id") or input_index)
            audio_path = split_dir / str(sample["file_name"])
            audio_array = _load_mono_audio(audio_path, sample_rate=sample_rate)
            if audio_array.size == 0:
                raise ValueError(f"音频为空: id={sample_id} path={audio_path}")
            reference_mask = _sample_seconds_to_mask(
                sample.get("seconds"),
                length=len(audio_array),
                sample_rate=sample_rate,
                sample_id=sample_id,
            )

            for part_index, start in enumerate(range(0, len(audio_array), step_size), start=1):
                end = min(len(audio_array), start + chunk_size)
                if end <= start:
                    continue
                chunk_audio = audio_array[start:end]
                chunk_mask = reference_mask[start:end]
                starts, durations = mask_to_seconds(chunk_mask, sample_rate)
                part_id = f"{sample_id}__part_{part_index:04d}"
                audio_name = f"{_safe_stem(part_id)}.wav"
                sf.write(output_audio_dir / audio_name, chunk_audio, sample_rate)
                metadata_file.write(
                    json.dumps(
                        {
                            "file_name": f"audio/{audio_name}",
                            "id": part_id,
                            "seconds": {
                                "starts": _float_list(starts),
                                "durations": _float_list(durations),
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_count += 1

    logger.info(
        "VAD 数据集切片完成: input=%s output=%s rows=%s",
        len(samples),
        output,
        output_count,
    )
    return TrimVadResult(
        output=output,
        split=split,
        input_sample_count=len(samples),
        output_sample_count=output_count,
        metadata_path=output_metadata_path,
    )


def _read_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "file_name" not in item:
                raise ValueError(f"metadata 缺少 file_name: line={line_number}")
            if "seconds" not in item:
                raise ValueError(f"metadata 缺少 seconds: line={line_number}")
            samples.append(item)
    if not samples:
        raise ValueError(f"metadata.jsonl 没有样本: {metadata_path}")
    return samples


def _load_mono_audio(audio_path: Path, *, sample_rate: int) -> np.ndarray:
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
    audio_array, source_sample_rate = sf.read(audio_path, always_2d=False)
    audio_array = np.asarray(audio_array)
    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=1)
    if audio_array.ndim != 1:
        raise ValueError(f"不支持的音频维度: path={audio_path} shape={audio_array.shape}")
    audio_array = audio_array.astype(np.float32, copy=False)
    if int(source_sample_rate) != sample_rate:
        audio_array = librosa.resample(
            audio_array,
            orig_sr=int(source_sample_rate),
            target_sr=sample_rate,
            res_type="scipy",
        )
    return audio_array.astype(np.float32, copy=False)


def _sample_seconds_to_mask(
    seconds: Any,
    *,
    length: int,
    sample_rate: int,
    sample_id: str,
) -> np.ndarray:
    if not isinstance(seconds, dict):
        raise ValueError(f"VAD 样本 seconds 必须是对象: id={sample_id}")
    starts = np.asarray(seconds.get("starts", []), dtype=np.float64)
    durations = np.asarray(seconds.get("durations", []), dtype=np.float64)
    if starts.shape != durations.shape:
        raise ValueError(
            f"seconds.starts 与 seconds.durations 长度不一致: "
            f"id={sample_id} {starts.shape} != {durations.shape}"
        )
    return seconds_to_mask(starts, durations, length, sample_rate)


def _float_list(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values.tolist()]


def _safe_stem(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in value.strip()
    )
    return safe or "sample"
