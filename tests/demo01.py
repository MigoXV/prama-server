from __future__ import annotations

import logging
from pathlib import Path

from prama_server.evaluator import Evaluator
from prama_server.inferencers.asr import AsrGrpcInferencer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

ASR_CONTEXT = "HOTEL"

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
    report_path = Path("outputs/demo01_wer_report.txt")

    logger.info("初始化 ASR 推理器: target=%s sample_rate=%s", target, sample_rate)
    with Evaluator(
        dataset_path=dataset_path,
        split=split,
        sample_rate=sample_rate,
        limit=limit,
        min_reference_words=min_reference_words,
        report_path=report_path,
    ) as evaluator:
        result = evaluator.evaluate(
            AsrGrpcInferencer(
                target=target,
                sample_rate=sample_rate,
                language_code=language_code,
                hotwords=[ASR_CONTEXT],
                request_timeout_seconds=request_timeout_seconds,
                connect_timeout_seconds=connect_timeout_seconds,
            )
        )

    wer_summary = result.wer_result.summary
    cer_summary = result.cer_result.summary
    print(f"WER: {wer_summary.wer:.2f}%")
    print(f"CER: {cer_summary.wer:.2f}%")
    print(
        "Counts: "
        f"ref={wer_summary.ref_words} hyp={wer_summary.hyp_words} "
        f"C={wer_summary.correct} S={wer_summary.substitutions} "
        f"D={wer_summary.deletions} I={wer_summary.insertions}"
    )


if __name__ == "__main__":
    main()
