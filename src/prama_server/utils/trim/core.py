from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from prama_server.utils.audition_formatter import (
    au_path_to_mask,
    mask_to_au_df,
    mask_to_seconds,
)

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
    chunk_seconds: float,
    overlap_seconds: float = 0.0,
    sample_rate: int = 16000,
    overwrite: bool = False,
    show_progress: bool = True,
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
    labeled_audio_paths = [
        audio_path
        for audio_path in audio_paths
        if audio_path.with_suffix(".csv").exists()
    ]
    skipped_count = len(audio_paths) - len(labeled_audio_paths)
    if not labeled_audio_paths:
        raise FileNotFoundError(f"输入目录中没有找到带同名 csv 标注的 wav 文件: {dataset_path}")

    used_ids: set[str] = set()
    chunk_size = max(1, int(round(chunk_seconds * sample_rate)))
    step_size = max(1, int(round((chunk_seconds - overlap_seconds) * sample_rate)))
    output_count = 0

    with output_metadata_path.open("w", encoding="utf-8") as metadata_file:
        progress = tqdm(
            labeled_audio_paths,
            desc="转换 VAD 样本",
            unit="file",
            disable=not show_progress,
        )
        for input_index, audio_path in enumerate(progress, start=1):
            progress.set_postfix_str(audio_path.name, refresh=False)
            sample_id = _unique_id(_safe_stem(audio_path.stem), used_ids)
            used_ids.add(sample_id)
            audio_array = _load_mono_audio(audio_path, sample_rate=sample_rate)
            if audio_array.size == 0:
                raise ValueError(f"音频为空: id={sample_id} path={audio_path}")
            csv_path = audio_path.with_suffix(".csv")
            reference_mask = au_path_to_mask(
                csv_path,
                length=len(audio_array),
                sr=sample_rate,
            )

            for part_index, start in enumerate(range(0, len(audio_array), step_size), start=1):
                end = min(len(audio_array), start + chunk_size)
                if end <= start:
                    continue
                chunk_audio = audio_array[start:end]
                chunk_mask = reference_mask[start:end]
                starts, durations = mask_to_seconds(chunk_mask, sample_rate)
                part_id = f"{sample_id}__part_{part_index:04d}"
                audio_name = f"{part_id}.wav"
                sf.write(output_audio_dir / audio_name, chunk_audio, sample_rate)
                csv_name = f"{part_id}.csv"
                mask_to_au_df(chunk_mask, sample_rate).to_csv(
                    output_audio_dir / csv_name,
                    sep="\t",
                    index=False,
                )
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
        "VAD 扁平目录已转换为 audiofolder: input=%s skipped_without_csv=%s output=%s rows=%s",
        len(audio_paths),
        skipped_count,
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
