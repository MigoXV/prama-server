from __future__ import annotations

import logging
from pathlib import Path

import typer

from prama_server.utils.vad_select.core import VAD_METRIC_RANGES, select_vad_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="按 VAD 评估结果 JSON 筛选并生成 VAD audiofolder 数据集。")


def _metric_min(metric_name: str) -> float | None:
    return VAD_METRIC_RANGES[metric_name][0]


def _metric_max(metric_name: str) -> float | None:
    return VAD_METRIC_RANGES[metric_name][1]


@app.command()
def main(
    result_json: Path = typer.Option(
        ...,
        "--result-json",
        help="前端下载的 VAD 结果 JSON，或包含 result.vad_report.samples 的后端快照 JSON",
        envvar="PRAMA_VAD_SELECT_RESULT_JSON",
    ),
    dataset_path: Path = typer.Option(
        ...,
        "--dataset-path",
        help="原始 VAD audiofolder 数据集根目录",
        envvar="PRAMA_VAD_SELECT_DATASET_PATH",
    ),
    split: str = typer.Option(
        "test",
        "--split",
        help="数据集 split",
        envvar="PRAMA_VAD_SELECT_SPLIT",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="输出 VAD audiofolder 数据集根目录",
        envvar="PRAMA_VAD_SELECT_OUTPUT",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="覆盖已存在的输出目录",
        envvar="PRAMA_VAD_SELECT_OVERWRITE",
    ),
    min_frame_recall: float | None = typer.Option(
        _metric_min("frame_recall"),
        "--min-frame-recall",
        help="frame_recall 下限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MIN_FRAME_RECALL",
    ),
    max_frame_recall: float | None = typer.Option(
        _metric_max("frame_recall"),
        "--max-frame-recall",
        help="frame_recall 上限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MAX_FRAME_RECALL",
    ),
    min_frame_precision: float | None = typer.Option(
        _metric_min("frame_precision"),
        "--min-frame-precision",
        help="frame_precision 下限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MIN_FRAME_PRECISION",
    ),
    max_frame_precision: float | None = typer.Option(
        _metric_max("frame_precision"),
        "--max-frame-precision",
        help="frame_precision 上限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MAX_FRAME_PRECISION",
    ),
    min_frame_f1: float | None = typer.Option(
        _metric_min("frame_f1"),
        "--min-frame-f1",
        help="frame_f1 下限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MIN_FRAME_F1",
    ),
    max_frame_f1: float | None = typer.Option(
        _metric_max("frame_f1"),
        "--max-frame-f1",
        help="frame_f1 上限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MAX_FRAME_F1",
    ),
    min_segment_recall: float | None = typer.Option(
        _metric_min("segment_recall"),
        "--min-segment-recall",
        help="segment_recall 下限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MIN_SEGMENT_RECALL",
    ),
    max_segment_recall: float | None = typer.Option(
        _metric_max("segment_recall"),
        "--max-segment-recall",
        help="segment_recall 上限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MAX_SEGMENT_RECALL",
    ),
    min_segment_precision: float | None = typer.Option(
        _metric_min("segment_precision"),
        "--min-segment-precision",
        help="segment_precision 下限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MIN_SEGMENT_PRECISION",
    ),
    max_segment_precision: float | None = typer.Option(
        _metric_max("segment_precision"),
        "--max-segment-precision",
        help="segment_precision 上限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MAX_SEGMENT_PRECISION",
    ),
    min_segment_f1: float | None = typer.Option(
        _metric_min("segment_f1"),
        "--min-segment-f1",
        help="segment_f1 下限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MIN_SEGMENT_F1",
    ),
    max_segment_f1: float | None = typer.Option(
        _metric_max("segment_f1"),
        "--max-segment-f1",
        help="segment_f1 上限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MAX_SEGMENT_F1",
    ),
    min_frame_false_alarm_rate: float | None = typer.Option(
        _metric_min("frame_false_alarm_rate"),
        "--min-frame-false-alarm-rate",
        help="frame_false_alarm_rate 下限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MIN_FRAME_FALSE_ALARM_RATE",
    ),
    max_frame_false_alarm_rate: float | None = typer.Option(
        _metric_max("frame_false_alarm_rate"),
        "--max-frame-false-alarm-rate",
        help="frame_false_alarm_rate 上限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MAX_FRAME_FALSE_ALARM_RATE",
    ),
    min_segment_false_alarm_rate: float | None = typer.Option(
        _metric_min("segment_false_alarm_rate"),
        "--min-segment-false-alarm-rate",
        help="segment_false_alarm_rate 下限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MIN_SEGMENT_FALSE_ALARM_RATE",
    ),
    max_segment_false_alarm_rate: float | None = typer.Option(
        _metric_max("segment_false_alarm_rate"),
        "--max-segment-false-alarm-rate",
        help="segment_false_alarm_rate 上限；默认读取 core.py 的 VAD_METRIC_RANGES",
        envvar="PRAMA_VAD_SELECT_MAX_SEGMENT_FALSE_ALARM_RATE",
    ),
) -> None:
    metric_ranges = {
        "frame_recall": (min_frame_recall, max_frame_recall),
        "frame_precision": (min_frame_precision, max_frame_precision),
        "frame_f1": (min_frame_f1, max_frame_f1),
        "segment_recall": (min_segment_recall, max_segment_recall),
        "segment_precision": (min_segment_precision, max_segment_precision),
        "segment_f1": (min_segment_f1, max_segment_f1),
        "frame_false_alarm_rate": (
            min_frame_false_alarm_rate,
            max_frame_false_alarm_rate,
        ),
        "segment_false_alarm_rate": (
            min_segment_false_alarm_rate,
            max_segment_false_alarm_rate,
        ),
    }
    result = select_vad_dataset(
        result_json=result_json,
        dataset_path=dataset_path,
        split=split,
        output=output,
        overwrite=overwrite,
        metric_ranges=metric_ranges,
    )
    logger.info(
        "VAD 样本筛选完成: total=%s selected=%s metadata=%s summary=%s",
        result.total_sample_count,
        result.selected_sample_count,
        result.metadata_path,
        result.summary_path,
    )


if __name__ == "__main__":
    app()
