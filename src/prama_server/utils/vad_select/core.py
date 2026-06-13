from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MetricRange = tuple[float | None, float | None]

# 修改这里即可调整筛选条件；默认全部为 None，等效于不做约束。
VAD_METRIC_RANGES: dict[str, MetricRange] = {
    "frame_recall": (None, None),
    "frame_precision": (None, None),
    "frame_f1": (None, None),
    "segment_recall": (None, None),
    "segment_precision": (None, None),
    "segment_f1": (None, None),
    "frame_false_alarm_rate": (None, None),
    "segment_false_alarm_rate": (None, None),
}


@dataclass(frozen=True)
class VadSelectResult:
    output: Path
    split: str
    total_sample_count: int
    selected_sample_count: int
    metadata_path: Path
    summary_path: Path


def select_vad_dataset(
    *,
    result_json: Path,
    dataset_path: Path,
    split: str = "test",
    output: Path,
    overwrite: bool = False,
    metric_ranges: dict[str, MetricRange] | None = None,
) -> VadSelectResult:
    result_json = result_json.resolve()
    dataset_path = dataset_path.resolve()
    output = output.resolve()
    active_ranges = _active_metric_ranges(
        VAD_METRIC_RANGES if metric_ranges is None else metric_ranges
    )

    if not result_json.exists() or not result_json.is_file():
        raise FileNotFoundError(f"VAD 结果 JSON 不存在: {result_json}")
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise NotADirectoryError(f"原始数据集目录不存在: {dataset_path}")
    if output == dataset_path:
        raise ValueError("输出目录不能与原始数据集目录相同")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在，请使用 --overwrite: {output}")
        logger.info("删除已有输出目录: %s", output)
        shutil.rmtree(output)

    metadata_path = _find_metadata_path(dataset_path, split=split)
    records_by_id = _read_metadata_by_id(metadata_path)
    samples = _read_vad_samples(result_json)
    selected_samples = [
        sample for sample in samples if _sample_matches(sample, active_ranges)
    ]

    output_split_dir = output / split
    output_audio_dir = output_split_dir / "audio"
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_path = output_split_dir / "metadata.jsonl"
    output_summary_path = output_split_dir / "selection_summary.json"

    used_file_names: set[str] = set()
    selected_ids: list[str] = []
    with output_metadata_path.open("w", encoding="utf-8") as metadata_file:
        for sample in selected_samples:
            sample_id = _sample_id(sample)
            source_record = records_by_id.get(sample_id)
            if source_record is None:
                raise KeyError(f"结果 JSON 中的样本 id 在原始 metadata 中不存在: {sample_id}")
            source_audio = _resolve_audio_path(source_record, metadata_path=metadata_path)
            if not source_audio.exists() or not source_audio.is_file():
                raise FileNotFoundError(f"音频文件不存在: id={sample_id} path={source_audio}")

            output_file_name = _unique_audio_file_name(source_audio.name, used_file_names)
            used_file_names.add(output_file_name)
            relative_output_file = f"audio/{output_file_name}"
            shutil.copy2(source_audio, output_audio_dir / output_file_name)

            output_record = dict(source_record)
            output_record["file_name"] = relative_output_file
            metadata_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            selected_ids.append(sample_id)

    summary = {
        "result_json": str(result_json),
        "dataset_path": str(dataset_path),
        "metadata_path": str(metadata_path),
        "output": str(output),
        "split": split,
        "total_sample_count": len(samples),
        "selected_sample_count": len(selected_ids),
        "enabled_metric_ranges": {
            key: {"min": value[0], "max": value[1]}
            for key, value in active_ranges.items()
        },
        "selected_ids": selected_ids,
    }
    output_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "VAD 筛选数据集已生成: total=%s selected=%s metadata=%s",
        len(samples),
        len(selected_ids),
        output_metadata_path,
    )
    return VadSelectResult(
        output=output,
        split=split,
        total_sample_count=len(samples),
        selected_sample_count=len(selected_ids),
        metadata_path=output_metadata_path,
        summary_path=output_summary_path,
    )


def _active_metric_ranges(metric_ranges: dict[str, MetricRange]) -> dict[str, MetricRange]:
    active: dict[str, MetricRange] = {}
    for key, (minimum, maximum) in metric_ranges.items():
        if minimum is None and maximum is None:
            continue
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"指标范围下限不能大于上限: {key} {minimum} > {maximum}")
        active[key] = (minimum, maximum)
    return active


def _find_metadata_path(dataset_path: Path, *, split: str) -> Path:
    candidates = [
        dataset_path / split / "metadata.jsonl",
        dataset_path / "metadata.jsonl",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(
        "原始 VAD audiofolder metadata.jsonl 不存在: "
        f"{candidates[0]} 或 {candidates[1]}"
    )


def _read_metadata_by_id(metadata_path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with metadata_path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"metadata.jsonl 解析失败: {metadata_path}:{line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"metadata.jsonl 每行必须是对象: {metadata_path}:{line_number}")
            sample_id = payload.get("id")
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"metadata.jsonl 缺少 id: {metadata_path}:{line_number}")
            if sample_id in records:
                raise ValueError(f"metadata.jsonl 中存在重复 id: {sample_id}")
            if "seconds" not in payload:
                raise ValueError(f"metadata.jsonl 缺少 seconds: id={sample_id}")
            records[sample_id] = payload
    return records


def _read_vad_samples(result_json: Path) -> list[dict[str, Any]]:
    with result_json.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"VAD 结果 JSON 顶层必须是对象: {result_json}")

    result_payload = payload.get("result")
    if isinstance(result_payload, dict) and isinstance(result_payload.get("vad_report"), dict):
        report = result_payload.get("vad_report")
    else:
        report = payload.get("vad_report")
    if not isinstance(report, dict):
        raise ValueError("VAD 结果 JSON 缺少 vad_report")
    samples = report.get("samples")
    if not isinstance(samples, list):
        raise ValueError("VAD 结果 JSON 缺少 vad_report.samples")
    for index, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            raise ValueError(f"vad_report.samples[{index}] 必须是对象")
        _sample_id(sample)
    return samples


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
        if minimum is not None and value < minimum:
            return False
        if maximum is not None and value > maximum:
            return False
    return True


def _resolve_audio_path(record: dict[str, Any], *, metadata_path: Path) -> Path:
    path_value = record.get("file_name") or record.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        sample_id = record.get("id", "")
        raise ValueError(f"metadata.jsonl 缺少 file_name/path: id={sample_id}")
    audio_path = Path(path_value)
    if not audio_path.is_absolute():
        audio_path = metadata_path.parent / audio_path
    return audio_path


def _unique_audio_file_name(file_name: str, used_file_names: set[str]) -> str:
    safe_name = _safe_file_name(file_name)
    if safe_name not in used_file_names:
        return safe_name
    path = Path(safe_name)
    stem = path.stem or "sample"
    suffix = path.suffix
    index = 2
    while True:
        candidate = f"{stem}_{index}{suffix}"
        if candidate not in used_file_names:
            return candidate
        index += 1


def _safe_file_name(file_name: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in Path(file_name).name.strip()
    )
    return safe or "sample.wav"
