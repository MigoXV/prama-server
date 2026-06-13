from __future__ import annotations

import logging
from pathlib import Path

import typer

from prama_server.utils.vad_select.core import select_vad_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="按 VAD 评估结果 JSON 筛选并生成 VAD audiofolder 数据集。")


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
) -> None:
    result = select_vad_dataset(
        result_json=result_json,
        dataset_path=dataset_path,
        split=split,
        output=output,
        overwrite=overwrite,
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
