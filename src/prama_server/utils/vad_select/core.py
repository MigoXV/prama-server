from __future__ import annotations

import json
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prama_server.utils.filesystem import (
    reject_output_inside_source,
    staged_output_directory,
)

logger = logging.getLogger(__name__)

MetricRange = tuple[float | None, float | None]

SUPPORTED_VAD_METRICS = frozenset(
    {
        "frame_accuracy",
        "frame_recall",
        "frame_precision",
        "frame_f1",
        "frame_specificity",
        "frame_false_alarm_rate",
        "frame_miss_rate",
        "frame_balanced_accuracy",
        "segment_recall",
        "segment_precision",
        "segment_f1",
        "segment_miss_rate",
        "segment_false_alarm_rate",
    }
)

# 可以在部署封装中覆盖该字典；默认不启用任何筛选条件。
VAD_METRIC_RANGES: dict[str, MetricRange] = {
    metric: (None, None) for metric in sorted(SUPPORTED_VAD_METRICS)
}


@dataclass(frozen=True)
class VadSelectResult:
    output: Path
    split: str
    total_sample_count: int
    selected_sample_count: int
    metadata_path: Path
    summary_path: Path


@dataclass(frozen=True)
class _SelectedSample:
    sample_id: str
    sample: dict[str, Any]
    source_record: dict[str, Any]
    source_audio: Path


def select_vad_dataset(
    *,
    result_json: Path,
    dataset_path: Path,
    output: Path,
    split: str = "test",
    overwrite: bool = False,
    metric_ranges: dict[str, MetricRange] | None = None,
) -> VadSelectResult:
    result_json = result_json.resolve()
    dataset_path = dataset_path.resolve()
    output = output.resolve()
    _validate_split(split)
    if not result_json.is_file():
        raise FileNotFoundError(f"VAD 结果 JSON 不存在: {result_json}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"原始数据集目录不存在: {dataset_path}")
    reject_output_inside_source(dataset_path, output)

    active_ranges = _active_metric_ranges(
        VAD_METRIC_RANGES if metric_ranges is None else metric_ranges
    )
    metadata_path = _find_metadata_path(dataset_path, split=split)
    records_by_id = _read_metadata_by_id(metadata_path)
    samples = _read_vad_samples(result_json)
    selected = _resolve_selected_samples(
        samples,
        records_by_id=records_by_id,
        metadata_path=metadata_path,
        active_ranges=active_ranges,
    )

    with staged_output_directory(output, overwrite=overwrite) as staging:
        _write_selected_dataset(
            selected,
            output=staging,
            final_output=output,
            split=split,
            result_json=result_json,
            dataset_path=dataset_path,
            metadata_path=metadata_path,
            total_sample_count=len(samples),
            active_ranges=active_ranges,
        )

    output_metadata_path = output / split / "metadata.jsonl"
    output_summary_path = output / split / "selection_summary.json"
    logger.info(
        "VAD 筛选数据集已生成: total=%s selected=%s metadata=%s",
        len(samples),
        len(selected),
        output_metadata_path,
    )
    return VadSelectResult(
        output=output,
        split=split,
        total_sample_count=len(samples),
        selected_sample_count=len(selected),
        metadata_path=output_metadata_path,
        summary_path=output_summary_path,
    )


def _validate_split(split: str) -> None:
    if not split.strip() or "/" in split or "\\" in split:
        raise ValueError(f"split 必须是单个非空目录名: {split!r}")


def _active_metric_ranges(
    metric_ranges: dict[str, MetricRange],
) -> dict[str, MetricRange]:
    unknown = sorted(set(metric_ranges) - SUPPORTED_VAD_METRICS)
    if unknown:
        raise ValueError(f"不支持的 VAD 指标: {', '.join(unknown)}")
    active: dict[str, MetricRange] = {}
    for key, bounds in metric_ranges.items():
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError(f"指标范围必须是 (min, max): {key}={bounds!r}")
        minimum, maximum = bounds
        for label, value in (("下限", minimum), ("上限", maximum)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"指标范围{label}必须是数字: {key}={value!r}")
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
                raise ValueError(f"指标范围{label}必须在 [0, 1] 内: {key}={value}")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"指标范围下限不能大于上限: {key} {minimum} > {maximum}")
        if minimum is not None or maximum is not None:
            active[key] = (
                None if minimum is None else float(minimum),
                None if maximum is None else float(maximum),
            )
    return active


def _find_metadata_path(dataset_path: Path, *, split: str) -> Path:
    candidates = [
        dataset_path / split / "metadata.jsonl",
        dataset_path / "metadata.jsonl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "原始 VAD audiofolder metadata.jsonl 不存在: "
        f"{candidates[0]} 或 {candidates[1]}"
    )


def _read_metadata_by_id(metadata_path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with metadata_path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"metadata.jsonl 解析失败: {metadata_path}:{line_number}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"metadata.jsonl 每行必须是对象: {metadata_path}:{line_number}"
                )
            sample_id = payload.get("id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"metadata.jsonl 缺少 id: {metadata_path}:{line_number}")
            if sample_id in records:
                raise ValueError(f"metadata.jsonl 中存在重复 id: {sample_id}")
            seconds = payload.get("seconds")
            if not isinstance(seconds, dict):
                raise ValueError(f"metadata.jsonl 缺少 seconds 对象: id={sample_id}")
            records[sample_id] = payload
    return records


def _read_vad_samples(result_json: Path) -> list[dict[str, Any]]:
    with result_json.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"VAD 结果 JSON 顶层必须是对象: {result_json}")
    result_payload = payload.get("result")
    if isinstance(result_payload, dict) and isinstance(
        result_payload.get("vad_report"),
        dict,
    ):
        report = result_payload["vad_report"]
    else:
        report = payload.get("vad_report")
    if not isinstance(report, dict):
        raise ValueError("VAD 结果 JSON 缺少 vad_report")
    samples = report.get("samples")
    if not isinstance(samples, list):
        raise ValueError("VAD 结果 JSON 缺少 vad_report.samples")

    seen_ids: set[str] = set()
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"vad_report.samples[{index}] 必须是对象")
        sample_id = _sample_id(sample)
        if sample_id in seen_ids:
            raise ValueError(f"vad_report.samples 中存在重复 id: {sample_id}")
        seen_ids.add(sample_id)
    return samples


def _resolve_selected_samples(
    samples: list[dict[str, Any]],
    *,
    records_by_id: dict[str, dict[str, Any]],
    metadata_path: Path,
    active_ranges: dict[str, MetricRange],
) -> list[_SelectedSample]:
    selected: list[_SelectedSample] = []
    for sample in samples:
        if not _sample_matches(sample, active_ranges):
            continue
        sample_id = _sample_id(sample)
        source_record = records_by_id.get(sample_id)
        if source_record is None:
            raise KeyError(f"结果 JSON 中的样本 id 在原始 metadata 中不存在: {sample_id}")
        source_audio = _resolve_audio_path(
            source_record,
            metadata_path=metadata_path,
        )
        selected.append(
            _SelectedSample(
                sample_id=sample_id,
                sample=sample,
                source_record=source_record,
                source_audio=source_audio,
            )
        )
    return selected


def _write_selected_dataset(
    selected: list[_SelectedSample],
    *,
    output: Path,
    final_output: Path,
    split: str,
    result_json: Path,
    dataset_path: Path,
    metadata_path: Path,
    total_sample_count: int,
    active_ranges: dict[str, MetricRange],
) -> None:
    output_split_dir = output / split
    output_audio_dir = output_split_dir / "audio"
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_path = output_split_dir / "metadata.jsonl"
    used_file_names: set[str] = set()
    selected_ids: list[str] = []

    with output_metadata_path.open("w", encoding="utf-8") as metadata_file:
        for item in selected:
            output_file_name = _unique_audio_file_name(
                item.source_audio.name,
                used_file_names,
            )
            used_file_names.add(output_file_name)
            shutil.copy2(item.source_audio, output_audio_dir / output_file_name)
            output_record = dict(item.source_record)
            output_record["file_name"] = f"audio/{output_file_name}"
            output_record.pop("path", None)
            metadata_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            selected_ids.append(item.sample_id)

    summary = {
        "result_json": str(result_json),
        "dataset_path": str(dataset_path),
        "metadata_path": str(metadata_path),
        "output": str(final_output),
        "split": split,
        "total_sample_count": total_sample_count,
        "selected_sample_count": len(selected_ids),
        "enabled_metric_ranges": {
            key: {"min": value[0], "max": value[1]}
            for key, value in active_ranges.items()
        },
        "selected_ids": selected_ids,
    }
    (output_split_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sample_id(sample: dict[str, Any]) -> str:
    value = sample.get("id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("VAD 样本缺少 id")
    return value


def _sample_matches(
    sample: dict[str, Any],
    active_ranges: dict[str, MetricRange],
) -> bool:
    if not active_ranges:
        return True
    metrics = sample.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"VAD 样本缺少 metrics: id={_sample_id(sample)}")
    for key, (minimum, maximum) in active_ranges.items():
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"VAD 样本指标不是数字: id={_sample_id(sample)} metric={key}")
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or not 0 <= numeric_value <= 1:
            raise ValueError(
                f"VAD 样本指标必须在 [0, 1] 内: "
                f"id={_sample_id(sample)} metric={key} value={value}"
            )
        if minimum is not None and numeric_value < minimum:
            return False
        if maximum is not None and numeric_value > maximum:
            return False
    return True


def _resolve_audio_path(record: dict[str, Any], *, metadata_path: Path) -> Path:
    path_value = record.get("file_name") or record.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError(f"metadata.jsonl 缺少 file_name/path: id={record.get('id', '')}")
    audio_path = Path(path_value)
    if not audio_path.is_absolute():
        audio_path = metadata_path.parent / audio_path
    audio_path = audio_path.resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(
            f"音频文件不存在: id={record.get('id', '')} path={audio_path}"
        )
    return audio_path


def _unique_audio_file_name(file_name: str, used_file_names: set[str]) -> str:
    safe_name = _safe_file_name(file_name)
    if safe_name not in used_file_names:
        return safe_name
    path = Path(safe_name)
    index = 2
    while True:
        candidate = f"{path.stem or 'sample'}_{index}{path.suffix}"
        if candidate not in used_file_names:
            return candidate
        index += 1


def _safe_file_name(file_name: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in Path(file_name).name.strip()
    )
    return safe or "sample.wav"
