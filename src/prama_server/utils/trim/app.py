from __future__ import annotations

import logging
from pathlib import Path

import typer

from prama_server.utils.trim.core import trim_vad_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="把 JSONL 或扁平 WAV/CSV 目录转换为 VAD audiofolder 数据。")


@app.command()
def main(
    dataset_path: Path = typer.Option(
        ...,
        "--dataset-path",
        help="输入数据集根目录、split 目录或 JSONL 文件",
        envvar="PRAMA_TRIM_VAD_DATASET_PATH",
    ),
    split: str = typer.Option(
        "test",
        "--split",
        help="输入和输出数据集 split",
        envvar="PRAMA_TRIM_VAD_SPLIT",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="输出根目录；默认使用输入路径旁边的 <name>-audiofolder",
        envvar="PRAMA_TRIM_VAD_OUTPUT",
    ),
    chunk_seconds: float = typer.Option(
        ...,
        "--chunk-seconds",
        min=0.000001,
        help="固定剪辑时长，单位秒",
        envvar="PRAMA_TRIM_VAD_CHUNK_SECONDS",
    ),
    overlap_seconds: float = typer.Option(
        0.0,
        "--overlap-seconds",
        min=0.0,
        help="相邻剪辑重叠时长，单位秒",
        envvar="PRAMA_TRIM_VAD_OVERLAP_SECONDS",
    ),
    sample_rate: int = typer.Option(
        16000,
        "--sample-rate",
        min=1,
        help="输出音频采样率",
        envvar="PRAMA_TRIM_VAD_SAMPLE_RATE",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="成功生成后替换已存在的输出目录",
        envvar="PRAMA_TRIM_VAD_OVERWRITE",
    ),
) -> None:
    result = trim_vad_dataset(
        dataset_path=dataset_path,
        split=split,
        output=output,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        sample_rate=sample_rate,
        overwrite=overwrite,
    )
    logger.info(
        "切片完成: input_samples=%s output_samples=%s metadata=%s",
        result.input_sample_count,
        result.output_sample_count,
        result.metadata_path,
    )


if __name__ == "__main__":
    app()
