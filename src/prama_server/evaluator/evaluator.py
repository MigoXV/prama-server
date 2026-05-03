from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from prama.evaluator.evaluator import get_cer, get_wer
from tqdm import tqdm

from prama_server.inferencers.base import Inferencer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    wer: float
    cer: float
    results_tsv: Path
    report_path: Path
    wer_result: Any
    cer_result: Any


class Evaluator:
    def __init__(
        self,
        dataset_path: Path | str,
        *,
        split: str = "test",
        sample_rate: int = 16000,
        limit: int | None = None,
        min_reference_words: int = 5,
        output_dir: Path | str = Path("outputs") / "evaluate",
        report_path: Path | str = Path("outputs/demo01_wer_report.txt"),
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.sample_rate = sample_rate
        self.limit = limit
        self.min_reference_words = min_reference_words
        self.output_dir = Path(output_dir)
        self.report_path = Path(report_path)

    def close(self) -> None:
        logger.info("Evaluator closed: dataset_path=%s split=%s", self.dataset_path, self.split)

    def __enter__(self) -> Evaluator:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def evaluate(self, inferencer: Inferencer) -> EvaluationResult:
        with inferencer as active_inferencer:
            return self._evaluate(active_inferencer)

    def _evaluate(self, inferencer: Inferencer) -> EvaluationResult:
        logger.info("加载数据集: path=%s split=%s", self.dataset_path, self.split)
        dataset = load_dataset(str(self.dataset_path), split=self.split).cast_column(
            "audio",
            Audio(decode=False),
        )
        if self.limit is not None:
            dataset = dataset.select(range(min(self.limit, len(dataset))))
        logger.info("待评估样本数: %s", len(dataset))

        utterance_ids: list[str] = []
        references: list[str] = []
        hypotheses: list[str] = []
        details: list[dict[str, object]] = []
        skipped_short_refs = 0

        progress_bar = tqdm(dataset, desc="ASR eval")
        for index, row in enumerate(progress_bar):
            utterance_id = str(row.get("id") or index)
            reference = " ".join(row["text"].strip().split())
            if len(reference.split()) < self.min_reference_words:
                skipped_short_refs += 1
                progress_bar.set_postfix(done=len(hypotheses), skipped=skipped_short_refs)
                continue

            audio_array, audio_sample_rate = self._load_audio(row["audio"], utterance_id)

            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1)
            if audio_sample_rate != self.sample_rate:
                raise ValueError(
                    f"样本采样率与 ASR 配置不一致: id={utterance_id} "
                    f"audio_sample_rate={audio_sample_rate} sample_rate={self.sample_rate}"
                )

            hypothesis = inferencer.infer(audio_array)

            utterance_ids.append(utterance_id)
            references.append(reference)
            hypotheses.append(hypothesis)
            details.append(
                {
                    "audio_id": utterance_id,
                    "ref": reference,
                    "hyp": hypothesis,
                }
            )
            progress_bar.set_postfix(done=len(hypotheses))

        logger.info("已跳过过短参考文本样本数: %s", skipped_short_refs)
        if not references:
            raise ValueError(
                f"没有可评估样本: min_reference_words={self.min_reference_words} "
                f"dataset_size={len(dataset)} skipped_short_refs={skipped_short_refs}"
            )

        logger.info("开始计算 WER/CER")
        wer_result = get_wer(references, hypotheses, utterance_ids)
        cer_result = get_cer(references, hypotheses, utterance_ids)
        row_wer = self._build_utterance_error_rates(wer_result)
        row_cer = self._build_utterance_error_rates(cer_result)
        for detail, wer, cer in zip(details, row_wer, row_cer, strict=True):
            detail["wer"] = wer
            detail["cer"] = cer

        results_tsv = self._build_results_tsv_path()
        self._write_results(results_tsv, details)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(wer_result.report, encoding="utf-8")

        logger.info("逐条结果已写入: %s", results_tsv)
        logger.info("对齐报告已写入: %s", self.report_path)

        return EvaluationResult(
            wer=wer_result.summary.wer,
            cer=cer_result.summary.wer,
            results_tsv=results_tsv,
            report_path=self.report_path,
            wer_result=wer_result,
            cer_result=cer_result,
        )

    def _build_results_tsv_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return self.output_dir / timestamp / "results.tsv"

    @staticmethod
    def _load_audio(audio: dict[str, object], utterance_id: str) -> tuple[Any, int]:
        if audio["bytes"] is not None:
            return sf.read(BytesIO(audio["bytes"]), dtype="float32")
        if audio["path"] is not None:
            return sf.read(audio["path"], dtype="float32")
        raise ValueError(f"音频缺少 bytes 和 path: id={utterance_id}")

    @staticmethod
    def _build_utterance_error_rates(result: Any) -> list[float]:
        rates: list[float] = []
        for utterance in result.utterances:
            correct = substitutions = deletions = insertions = 0
            for token in utterance.tokens:
                if token.eval_label == "correct":
                    correct += 1
                elif token.eval_label == "substitution":
                    substitutions += 1
                elif token.eval_label == "deletion":
                    deletions += 1
                elif token.eval_label == "insertion":
                    insertions += 1

            ref_count = correct + substitutions + deletions
            error_count = substitutions + deletions + insertions
            rates.append(error_count / ref_count * 100 if ref_count else 0.0)

        return rates

    @staticmethod
    def _write_results(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["audio_id", "ref", "hyp", "wer", "cer"]).to_csv(
            path,
            sep="\t",
            index=False,
        )
