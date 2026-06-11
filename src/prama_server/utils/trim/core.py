from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from prama_server.utils.audition_formatter import au_path_to_mask, mask_to_seconds

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
    output: Path | None = None,
    sample_rate: int = 16000,
    overwrite: bool = False,
) -> TrimVadResult:
    if sample_rate <= 0:
        raise ValueError(f"sample_rate 必须大于 0: {sample_rate}")

    dataset_path = dataset_path.resolve()
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise NotADirectoryError(f"输入目录不存在: {dataset_path}")
    output = (output or dataset_path.with_name(f"{dataset_path.name}-audiofolder")).resolve()
    if dataset_path == output:
        raise ValueError("输出目录不能与输入数据集目录相同")

    output_split_dir = output / split
    output_audio_dir = output_split_dir / "audio"
    output_metadata_path = output_split_dir / "metadata.jsonl"
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在，请使用 --overwrite: {output}")
        logger.info("删除已有输出目录: %s", output)
        shutil.rmtree(output)
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = sorted(dataset_path.glob("*.wav"))
    if not audio_paths:
        raise FileNotFoundError(f"输入目录中没有找到 wav 文件: {dataset_path}")

    used_ids: set[str] = set()
    output_count = 0

    with output_metadata_path.open("w", encoding="utf-8") as metadata_file:
        for input_index, audio_path in enumerate(audio_paths, start=1):
            sample_id = _unique_id(_safe_stem(audio_path.stem), used_ids)
            used_ids.add(sample_id)
            audio_array = _load_mono_audio(audio_path, sample_rate=sample_rate)
            if audio_array.size == 0:
                raise ValueError(f"音频为空: id={sample_id} path={audio_path}")
            csv_path = audio_path.with_suffix(".csv")
            if csv_path.exists():
                reference_mask = au_path_to_mask(
                    csv_path,
                    length=len(audio_array),
                    sr=sample_rate,
                )
                starts, durations = mask_to_seconds(reference_mask, sample_rate)
            else:
                starts = np.asarray([], dtype=np.float64)
                durations = np.asarray([], dtype=np.float64)

            audio_name = f"{sample_id}.wav"
            sf.write(output_audio_dir / audio_name, audio_array, sample_rate)
            metadata_file.write(
                json.dumps(
                    {
                        "file_name": f"audio/{audio_name}",
                        "id": sample_id,
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
        "VAD 扁平目录已转换为 audiofolder: input=%s output=%s rows=%s",
        len(audio_paths),
        output,
        output_count,
    )
    return TrimVadResult(
        output=output,
        split=split,
        input_sample_count=len(audio_paths),
        output_sample_count=output_count,
        metadata_path=output_metadata_path,
    )


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


def _float_list(values: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in values.tolist()]


def _safe_stem(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in value.strip()
    )
    return safe or "sample"


def _unique_id(sample_id: str, used_ids: set[str]) -> str:
    if sample_id not in used_ids:
        return sample_id
    index = 2
    while f"{sample_id}_{index}" in used_ids:
        index += 1
    return f"{sample_id}_{index}"
