from __future__ import annotations

import json
import logging
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
from prama_server.utils.filesystem import (
    reject_output_inside_source,
    staged_output_directory,
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
    split: str = "test",
    output: Path | None = None,
    chunk_seconds: float,
    overlap_seconds: float = 0.0,
    sample_rate: int = 16000,
    overwrite: bool = False,
    show_progress: bool = True,
) -> TrimVadResult:
    _validate_trim_options(
        split=split,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        sample_rate=sample_rate,
    )
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {dataset_path}")
    if not dataset_path.is_dir() and dataset_path.suffix.lower() != ".jsonl":
        raise NotADirectoryError(f"输入路径必须是目录或 JSONL 文件: {dataset_path}")

    default_output_name = (
        f"{dataset_path.stem}-audiofolder"
        if dataset_path.is_file()
        else f"{dataset_path.name}-audiofolder"
    )
    output = (output or dataset_path.with_name(default_output_name)).resolve()
    reject_output_inside_source(dataset_path, output)

    jsonl_path = _find_jsonl_path(dataset_path, split=split)
    if jsonl_path is not None:
        input_samples = _load_jsonl_samples(jsonl_path)
        skipped_count = 0
        source_description = f"jsonl={jsonl_path}"
    else:
        input_samples, skipped_count = _load_csv_samples(dataset_path)
        source_description = f"flat_wav_csv={dataset_path}"

    with staged_output_directory(output, overwrite=overwrite) as staging:
        output_count = _write_trimmed_dataset(
            input_samples,
            output=staging,
            split=split,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            sample_rate=sample_rate,
            show_progress=show_progress,
        )

    metadata_path = output / split / "metadata.jsonl"
    logger.info(
        "VAD 数据已转换为 audiofolder: source=%s input=%s "
        "skipped_without_csv=%s output=%s rows=%s",
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
        metadata_path=metadata_path,
    )


def _validate_trim_options(
    *,
    split: str,
    chunk_seconds: float,
    overlap_seconds: float,
    sample_rate: int,
) -> None:
    if not split.strip() or "/" in split or "\\" in split:
        raise ValueError(f"split 必须是单个非空目录名: {split!r}")
    if not np.isfinite(chunk_seconds) or chunk_seconds <= 0:
        raise ValueError(f"chunk_seconds 必须是大于 0 的有限值: {chunk_seconds}")
    if not np.isfinite(overlap_seconds) or overlap_seconds < 0:
        raise ValueError(f"overlap_seconds 必须是非负有限值: {overlap_seconds}")
    if overlap_seconds >= chunk_seconds:
        raise ValueError(
            "overlap_seconds 必须小于 chunk_seconds: "
            f"{overlap_seconds} >= {chunk_seconds}"
        )
    if sample_rate <= 0:
        raise ValueError(f"sample_rate 必须大于 0: {sample_rate}")


def _write_trimmed_dataset(
    input_samples: list[_VadSample],
    *,
    output: Path,
    split: str,
    chunk_seconds: float,
    overlap_seconds: float,
    sample_rate: int,
    show_progress: bool,
) -> int:
    output_audio_dir = output / split / "audio"
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output / split / "metadata.jsonl"
    chunk_size = max(1, int(round(chunk_seconds * sample_rate)))
    step_size = max(1, int(round((chunk_seconds - overlap_seconds) * sample_rate)))
    used_ids: set[str] = set()
    output_count = 0

    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        progress = tqdm(
            input_samples,
            desc="转换 VAD 样本",
            unit="file",
            disable=not show_progress,
        )
        for sample in progress:
            progress.set_postfix_str(sample.audio_path.name, refresh=False)
            sample_id = _unique_id(_safe_stem(sample.sample_id), used_ids)
            used_ids.add(sample_id)
            audio_array = _load_mono_audio(sample.audio_path, sample_rate=sample_rate)
            if audio_array.size == 0:
                raise ValueError(f"音频为空: id={sample_id} path={sample.audio_path}")
            reference_mask = _sample_mask(
                sample,
                audio_length=len(audio_array),
                sample_rate=sample_rate,
            )
            for part_index, start in enumerate(
                range(0, len(audio_array), step_size),
                start=1,
            ):
                end = min(len(audio_array), start + chunk_size)
                if end <= start:
                    continue
                part_id = f"{sample_id}__part_{part_index:04d}"
                audio_name = f"{part_id}.wav"
                chunk_audio = audio_array[start:end]
                chunk_mask = reference_mask[start:end]
                sf.write(output_audio_dir / audio_name, chunk_audio, sample_rate)
                mask_to_au_df(chunk_mask, sample_rate).to_csv(
                    output_audio_dir / f"{part_id}.csv",
                    sep="\t",
                    index=False,
                )
                starts, durations = mask_to_seconds(chunk_mask, sample_rate)
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
    return output_count


def _sample_mask(
    sample: _VadSample,
    *,
    audio_length: int,
    sample_rate: int,
) -> np.ndarray:
    if sample.starts is None or sample.durations is None:
        return au_path_to_mask(
            sample.audio_path.with_suffix(".csv"),
            length=audio_length,
            sample_rate=sample_rate,
        )
    return seconds_to_mask(
        sample.starts,
        sample.durations,
        length=audio_length,
        sample_rate=sample_rate,
    )


def _find_jsonl_path(dataset_path: Path, *, split: str) -> Path | None:
    if dataset_path.is_file():
        return dataset_path
    candidates = [
        dataset_path / split / "metadata.jsonl",
        dataset_path / "metadata.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    jsonl_paths = sorted(dataset_path.glob("*.jsonl"))
    return jsonl_paths[0] if jsonl_paths else None


def _load_jsonl_samples(jsonl_path: Path) -> list[_VadSample]:
    samples: list[_VadSample] = []
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL 解析失败: path={jsonl_path} line={line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL 每行必须是对象: path={jsonl_path} line={line_number}"
                )
            audio_path = _resolve_jsonl_audio_path(
                row,
                jsonl_path=jsonl_path,
                line_number=line_number,
            )
            seconds = row.get("seconds")
            if not isinstance(seconds, dict):
                raise ValueError(
                    f"JSONL 缺少 seconds 对象: path={jsonl_path} line={line_number}"
                )
            starts = _seconds_array(
                seconds,
                "starts",
                jsonl_path=jsonl_path,
                line_number=line_number,
            )
            durations = _seconds_array(
                seconds,
                "durations",
                jsonl_path=jsonl_path,
                line_number=line_number,
            )
            if starts.shape != durations.shape:
                raise ValueError(
                    "JSONL seconds.starts 与 seconds.durations 长度不一致: "
                    f"path={jsonl_path} line={line_number}"
                )
            samples.append(
                _VadSample(
                    sample_id=str(row.get("id") or audio_path.stem),
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
        if audio_path.with_suffix(".csv").is_file()
    ]
    if not samples:
        raise FileNotFoundError(
            f"输入目录中没有找到带同名 csv 标注的 wav 文件: {dataset_path}"
        )
    return samples, len(audio_paths) - len(samples)


def _resolve_jsonl_audio_path(
    row: dict[str, Any],
    *,
    jsonl_path: Path,
    line_number: int,
) -> Path:
    path_value = row.get("file_name") or row.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(
            f"JSONL 缺少 file_name/path: path={jsonl_path} line={line_number}"
        )
    audio_path = Path(path_value)
    if not audio_path.is_absolute():
        audio_path = jsonl_path.parent / audio_path
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")
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
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"JSONL seconds.{key} 包含非数字值: path={jsonl_path} line={line_number}"
        ) from exc
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(
            f"JSONL seconds.{key} 必须是一维有限数值数组: "
            f"path={jsonl_path} line={line_number}"
        )
    if key == "durations" and np.any(result < 0):
        raise ValueError(
            f"JSONL seconds.durations 不能包含负数: path={jsonl_path} line={line_number}"
        )
    return result


def _load_mono_audio(audio_path: Path, *, sample_rate: int) -> np.ndarray:
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
