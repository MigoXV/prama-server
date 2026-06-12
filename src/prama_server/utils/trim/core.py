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
from tqdm import tqdm

from prama_server.utils.audition_formatter import (
    au_path_to_mask,
    mask_to_au_df,
    mask_to_seconds,
    seconds_to_mask,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrimVadResult:
    output: Path
    split: str
    input_sample_count: int
    output_sample_count: int
    metadata_path: Path


@dataclass(frozen=True)
class _VadSample:
    sample_id: str
    audio_path: Path
    starts: np.ndarray | None
    durations: np.ndarray | None


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
    if not dataset_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {dataset_path}")
    if not dataset_path.is_dir() and dataset_path.suffix.lower() != ".jsonl":
        raise NotADirectoryError(f"输入目录不存在: {dataset_path}")
    default_output_name = (
        f"{dataset_path.stem}-audiofolder"
        if dataset_path.is_file()
        else f"{dataset_path.name}-audiofolder"
    )
    output = (output or dataset_path.with_name(default_output_name)).resolve()
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

    jsonl_path = _find_jsonl_path(dataset_path)
    if jsonl_path is not None:
        input_samples = _load_jsonl_samples(jsonl_path)
        skipped_count = 0
        source_description = f"jsonl={jsonl_path}"
    else:
        input_samples, skipped_count = _load_csv_samples(dataset_path)
        source_description = f"flat_wav_csv={dataset_path}"

    used_ids: set[str] = set()
    chunk_size = max(1, int(round(chunk_seconds * sample_rate)))
    step_size = max(1, int(round((chunk_seconds - overlap_seconds) * sample_rate)))
    output_count = 0

    with output_metadata_path.open("w", encoding="utf-8") as metadata_file:
        progress = tqdm(
            input_samples,
            desc="转换 VAD 样本",
            unit="file",
            disable=not show_progress,
        )
        for input_index, sample in enumerate(progress, start=1):
            audio_path = sample.audio_path
            progress.set_postfix_str(audio_path.name, refresh=False)
            sample_id = _unique_id(_safe_stem(sample.sample_id), used_ids)
            used_ids.add(sample_id)
            audio_array = _load_mono_audio(audio_path, sample_rate=sample_rate)
            if audio_array.size == 0:
                raise ValueError(f"音频为空: id={sample_id} path={audio_path}")
            if sample.starts is None or sample.durations is None:
                csv_path = audio_path.with_suffix(".csv")
                reference_mask = au_path_to_mask(
                    csv_path,
                    length=len(audio_array),
                    sr=sample_rate,
                )
            else:
                reference_mask = seconds_to_mask(
                    sample.starts,
                    sample.durations,
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
        "VAD 数据已转换为 audiofolder: source=%s input=%s skipped_without_csv=%s output=%s rows=%s",
        source_description,
        len(input_samples) + skipped_count,
        skipped_count,
        output,
        output_count,
    )
    return TrimVadResult(
        output=output,
        split=split,
        input_sample_count=len(input_samples) + skipped_count,
        output_sample_count=output_count,
        metadata_path=output_metadata_path,
    )


def _find_jsonl_path(dataset_path: Path) -> Path | None:
    if dataset_path.is_file() and dataset_path.suffix.lower() == ".jsonl":
        return dataset_path
    jsonl_paths = sorted(dataset_path.glob("*.jsonl"))
    if not jsonl_paths:
        return None
    metadata_path = dataset_path / "metadata.jsonl"
    if metadata_path in jsonl_paths:
        return metadata_path
    return jsonl_paths[0]


def _load_jsonl_samples(jsonl_path: Path) -> list[_VadSample]:
    samples: list[_VadSample] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 解析失败: path={jsonl_path} line={line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL 每行必须是对象: path={jsonl_path} line={line_number}")

            audio_path = _resolve_jsonl_audio_path(row, jsonl_path=jsonl_path, line_number=line_number)
            seconds = row.get("seconds")
            if not isinstance(seconds, dict):
                raise ValueError(
                    f"JSONL 缺少 seconds 对象: path={jsonl_path} line={line_number}"
                )
            starts = _seconds_array(seconds, "starts", jsonl_path=jsonl_path, line_number=line_number)
            durations = _seconds_array(
                seconds,
                "durations",
                jsonl_path=jsonl_path,
                line_number=line_number,
            )
            if starts.shape != durations.shape:
                raise ValueError(
                    "JSONL seconds.starts 与 seconds.durations 长度不一致: "
                    f"path={jsonl_path} line={line_number} "
                    f"{starts.shape} != {durations.shape}"
                )
            sample_id = str(row.get("id") or audio_path.stem)
            samples.append(
                _VadSample(
                    sample_id=sample_id,
                    audio_path=audio_path,
                    starts=starts,
                    durations=durations,
                )
            )
    if not samples:
        raise FileNotFoundError(f"JSONL 文件中没有可处理样本: {jsonl_path}")
    return samples


def _load_csv_samples(dataset_path: Path) -> tuple[list[_VadSample], int]:
    audio_paths = sorted(dataset_path.glob("*.wav"))
    if not audio_paths:
        raise FileNotFoundError(f"输入目录中没有找到 wav 文件: {dataset_path}")
    samples = [
        _VadSample(
            sample_id=audio_path.stem,
            audio_path=audio_path,
            starts=None,
            durations=None,
        )
        for audio_path in audio_paths
        if audio_path.with_suffix(".csv").exists()
    ]
    skipped_count = len(audio_paths) - len(samples)
    if not samples:
        raise FileNotFoundError(f"输入目录中没有找到带同名 csv 标注的 wav 文件: {dataset_path}")
    return samples, skipped_count


def _resolve_jsonl_audio_path(row: dict[str, Any], *, jsonl_path: Path, line_number: int) -> Path:
    path_value = row.get("file_name") or row.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"JSONL 缺少 file_name/path: path={jsonl_path} line={line_number}")
    audio_path = Path(path_value)
    if not audio_path.is_absolute():
        audio_path = jsonl_path.parent / audio_path
    return audio_path


def _seconds_array(
    seconds: dict[str, Any],
    key: str,
    *,
    jsonl_path: Path,
    line_number: int,
) -> np.ndarray:
    value = seconds.get(key, [])
    if not isinstance(value, list):
        raise ValueError(
            f"JSONL seconds.{key} 必须是数组: path={jsonl_path} line={line_number}"
        )
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"JSONL seconds.{key} 包含非数字值: path={jsonl_path} line={line_number}"
        ) from exc


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
