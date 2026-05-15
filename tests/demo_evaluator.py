from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
from datasets import Audio, Dataset, load_dataset

from prama_server.evaluator import (
    EvaluationProgress,
    EvaluationResult,
    Evaluator,
)
from prama_server.inferencers.asr import AsrGrpcInferencer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationOutputPaths:
    results_tsv: Path
    report_path: Path


def load_evaluation_dataset(
    dataset_path: Path | str,
    *,
    split: str = "test",
    limit: int | None = None,
    sample_rate: int = 16000,
    min_reference_words: int | None = None,
) -> Dataset:
    logger.info("加载数据集: path=%s split=%s", dataset_path, split)
    dataset = load_dataset(str(dataset_path), split=split).cast_column(
        "audio",
        Audio(sampling_rate=sample_rate),
    )
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    if min_reference_words is not None:
        before_filter = len(dataset)
        dataset = dataset.filter(
            lambda sample: len(" ".join(sample["text"].strip().split()).split())
            >= min_reference_words
        )
        logger.info(
            "参考文本词数过滤完成: min_reference_words=%s before=%s after=%s",
            min_reference_words,
            before_filter,
            len(dataset),
        )

    logger.info("数据集已加载: size=%s", len(dataset))
    return dataset


def write_evaluation_outputs(
    result: EvaluationResult,
    *,
    output_dir: Path | str = Path("outputs") / "evaluate",
    report_path: Path | str = Path("outputs/demo_evaluator_wer_report.txt"),
) -> EvaluationOutputPaths:
    results_tsv = build_results_tsv_path(Path(output_dir))
    write_results_tsv(results_tsv, result.details)

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(result.wer_result.report, encoding="utf-8")

    logger.info("逐条结果已写入: %s", results_tsv)
    logger.info("对齐报告已写入: %s", report_path)
    return EvaluationOutputPaths(results_tsv=results_tsv, report_path=report_path)


def build_results_tsv_path(output_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return output_dir / timestamp / "results.tsv"


def write_results_tsv(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, sep="\t", index=False)


def log_progress(progress: EvaluationProgress) -> None:
    if progress.status == "started":
        logger.info("评估开始: total=%s", progress.total)
        return

    if progress.status == "completed":
        logger.info(
            "评估完成: processed=%s evaluated=%s WER=%.2f%% CER=%.2f%%",
            progress.processed,
            progress.evaluated,
            progress.running_wer or 0.0,
            progress.running_cer or 0.0,
        )
        return

    logger.info(
        "推理进度: processed=%s/%s evaluated=%s id=%s",
        progress.processed,
        progress.total,
        progress.evaluated,
        progress.current_id,
    )


def main() -> None:
    target = "192.168.0.213:50003"
    dataset_path = Path("data-bin/jacktol/ATC-ASR-Dataset")
    split = "test"
    limit = 10
    language_code = "auto"
    sample_rate = 16000
    min_reference_words = 5
    connect_timeout_seconds = 10.0
    request_timeout_seconds = 60.0
    report_path = Path("outputs/demo_evaluator_wer_report.txt")

    dataset = load_evaluation_dataset(
        dataset_path,
        split=split,
        limit=limit,
        sample_rate=sample_rate,
        min_reference_words=min_reference_words,
    )

    logger.info("初始化 ASR 推理器: target=%s sample_rate=%s", target, sample_rate)
    result: EvaluationResult | None = None
    inferencer = AsrGrpcInferencer(
        target=target,
        sample_rate=sample_rate,
        language_code=language_code,
        request_timeout_seconds=request_timeout_seconds,
        connect_timeout_seconds=connect_timeout_seconds,
    )
    with Evaluator(
        dataset=dataset,
        sample_rate=sample_rate,
    ) as evaluator:
        session_metrics = evaluator.iter_evaluate(inferencer)
        print(f"Session Metrics: {session_metrics}")



if __name__ == "__main__":
    main()
