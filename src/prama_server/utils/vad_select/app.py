from __future__ import annotations

import logging
from pathlib import Path

import typer

from prama_server.utils.vad_select.core import (
    VAD_METRIC_RANGES,
    MetricRange,
    select_vad_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="按逐样本 VAD 指标筛选并生成新的 audiofolder 数据集。")


def _metric_min(metric_name: str) -> float | None:
    return VAD_METRIC_RANGES[metric_name][0]


def _metric_max(metric_name: str) -> float | None:
    return VAD_METRIC_RANGES[metric_name][1]


def _range_option(
    metric: str,
    *,
    lower: bool,
) -> typer.models.OptionInfo:
    bound = "min" if lower else "max"
    chinese_bound = "下限" if lower else "上限"
    default = _metric_min(metric) if lower else _metric_max(metric)
    env_metric = metric.upper()
    return typer.Option(
        default,
        f"--{bound}-{metric.replace('_', '-')}",
        min=0.0,
        max=1.0,
        help=f"{metric} {chinese_bound}，闭区间，取值 0..1",
        envvar=f"PRAMA_VAD_SELECT_{bound.upper()}_{env_metric}",
    )


@app.command()
def main(
    result_json: Path = typer.Option(
        ...,
        "--result-json",
        help="VAD 结果 JSON 或包含 result.vad_report.samples 的任务快照",
        envvar="PRAMA_VAD_SELECT_RESULT_JSON",
    ),
    dataset_path: Path = typer.Option(
        ...,
        "--dataset-path",
        help="原始 VAD audiofolder 数据集根目录",
        envvar="PRAMA_VAD_SELECT_DATASET_PATH",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="输出 VAD audiofolder 数据集根目录",
        envvar="PRAMA_VAD_SELECT_OUTPUT",
    ),
    split: str = typer.Option(
        "test",
        "--split",
        help="数据集 split",
        envvar="PRAMA_VAD_SELECT_SPLIT",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="成功生成后替换已存在的输出目录",
        envvar="PRAMA_VAD_SELECT_OVERWRITE",
    ),
    min_frame_accuracy: float | None = _range_option(
        "frame_accuracy",
        lower=True,
    ),
    max_frame_accuracy: float | None = _range_option(
        "frame_accuracy",
        lower=False,
    ),
    min_frame_recall: float | None = _range_option("frame_recall", lower=True),
    max_frame_recall: float | None = _range_option("frame_recall", lower=False),
    min_frame_precision: float | None = _range_option(
        "frame_precision",
        lower=True,
    ),
    max_frame_precision: float | None = _range_option(
        "frame_precision",
        lower=False,
    ),
    min_frame_f1: float | None = _range_option("frame_f1", lower=True),
    max_frame_f1: float | None = _range_option("frame_f1", lower=False),
    min_segment_recall: float | None = _range_option(
        "segment_recall",
        lower=True,
    ),
    max_segment_recall: float | None = _range_option(
        "segment_recall",
        lower=False,
    ),
    min_segment_precision: float | None = _range_option(
        "segment_precision",
        lower=True,
    ),
    max_segment_precision: float | None = _range_option(
        "segment_precision",
        lower=False,
    ),
    min_segment_f1: float | None = _range_option("segment_f1", lower=True),
    max_segment_f1: float | None = _range_option("segment_f1", lower=False),
    min_frame_false_alarm_rate: float | None = _range_option(
        "frame_false_alarm_rate",
        lower=True,
    ),
    max_frame_false_alarm_rate: float | None = _range_option(
        "frame_false_alarm_rate",
        lower=False,
    ),
    min_segment_false_alarm_rate: float | None = _range_option(
        "segment_false_alarm_rate",
        lower=True,
    ),
    max_segment_false_alarm_rate: float | None = _range_option(
        "segment_false_alarm_rate",
        lower=False,
    ),
) -> None:
    metric_ranges: dict[str, MetricRange] = {
        "frame_accuracy": (min_frame_accuracy, max_frame_accuracy),
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
        output=output,
        split=split,
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
