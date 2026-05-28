from __future__ import annotations

import logging
import string
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from prama_server.evaluator import EvaluationProgress, EvaluationResult, Evaluator
from prama_server.inferencers.asr import AsrGrpcInferencer
from tests.demo_evaluator import load_evaluation_dataset, write_evaluation_outputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

ASR_CONTEXT = "HOTEL"
PUNCTUATION_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})
DIGIT_WORDS = {
    "0": "ZERO",
    "1": "ONE",
    "2": "TWO",
    "3": "THREE",
    "4": "FOUR",
    "5": "FIVE",
    "6": "SIX",
    "7": "SEVEN",
    "8": "EIGHT",
    "9": "NINE",
}


class AtcTextPostprocessedInferencer:
    def __init__(self, inferencer: AsrGrpcInferencer) -> None:
        self.inferencer = inferencer

    def infer(self, audio: np.ndarray) -> Iterator[tuple[str, bool]]:
        for transcript, is_final in self.inferencer.infer(audio):
            yield normalize_atc_text(transcript), is_final

    def __enter__(self) -> AtcTextPostprocessedInferencer:
        self.inferencer.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self.inferencer.__exit__(*args)


def normalize_atc_text(text: str) -> str:
    text = text.translate(PUNCTUATION_TRANSLATION)
    text = "".join(f" {DIGIT_WORDS[char]} " if char in DIGIT_WORDS else char for char in text)
    return " ".join(text.strip().split()).upper()


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
    target = "192.168.1.24:50008"
    dataset_path = Path("data-bin/jacktol/ATC-ASR-Dataset")
    split = "test"
    limit: int | None = None
    language_code = "en-US"
    sample_rate = 16000
    min_reference_words = 5
    connect_timeout_seconds = 10.0
    request_timeout_seconds = 60.0
    output_dir = Path("outputs/evaluate")
    report_path = Path("outputs/demo01_wer_report.txt")

    dataset = load_evaluation_dataset(
        dataset_path,
        split=split,
        limit=limit,
        sample_rate=sample_rate,
        min_reference_words=min_reference_words,
    )
    logger.info("初始化 ASR 推理器: target=%s sample_rate=%s", target, sample_rate)
    result: EvaluationResult | None = None
    with Evaluator(
        dataset=dataset,
        sample_rate=sample_rate,
    ) as evaluator:
        for progress in evaluator.iter_evaluate(
            AtcTextPostprocessedInferencer(
                AsrGrpcInferencer(
                    target=target,
                    sample_rate=sample_rate,
                    language_code=language_code,
                    hotwords=[ASR_CONTEXT],
                    request_timeout_seconds=request_timeout_seconds,
                    connect_timeout_seconds=connect_timeout_seconds,
                )
            )
        ):
            log_progress(progress)
            if progress.result is not None:
                result = progress.result

    if result is None:
        raise RuntimeError("评估结束但没有生成结果")

    output_paths = write_evaluation_outputs(
        result,
        output_dir=output_dir,
        report_path=report_path,
    )
    wer_summary = result.wer_result.summary
    cer_summary = result.cer_result.summary
    print(f"WER: {wer_summary.wer:.2f}%")
    print(f"CER: {cer_summary.wer:.2f}%")
    print(f"results.tsv: {output_paths.results_tsv}")
    print(f"report: {output_paths.report_path}")
    print(
        "Counts: "
        f"ref={wer_summary.ref_words} hyp={wer_summary.hyp_words} "
        f"C={wer_summary.correct} S={wer_summary.substitutions} "
        f"D={wer_summary.deletions} I={wer_summary.insertions}"
    )


if __name__ == "__main__":
    main()
