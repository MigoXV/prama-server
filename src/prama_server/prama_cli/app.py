from __future__ import annotations

import csv
import logging
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

import typer
from tqdm import tqdm

from prama_server.message_manager import MESSAGE_SENTINEL, ManagedMessage
from prama_server.servicer.http import (
    EvaluationJob,
    EvaluationRequest,
    _run_evaluation,
    message_manager,
)

logger = logging.getLogger(__name__)
app = typer.Typer(help="离线运行 ASR、VAD、LID、SE 和关键词评估。")


def _target_option(default: str) -> Any:
    return typer.Option(
        default,
        "--target",
        help="gRPC 服务地址",
        envvar="PRAMA_CLI_TARGET",
    )


def _dataset_path_option(default: str) -> Any:
    return typer.Option(
        Path(default),
        "--dataset-path",
        help="数据集路径",
        envvar="PRAMA_CLI_DATASET_PATH",
    )


def _split_option() -> Any:
    return typer.Option("test", "--split", help="数据集 split", envvar="PRAMA_CLI_SPLIT")


def _limit_option() -> Any:
    return typer.Option(
        None,
        "--limit",
        min=1,
        help="最多评估的音频数；关键词任务按音频而不是关键词条目计数",
        envvar="PRAMA_CLI_LIMIT",
    )


def _sample_rate_option() -> Any:
    return typer.Option(
        16000,
        "--sample-rate",
        min=1,
        help="音频采样率",
        envvar="PRAMA_CLI_SAMPLE_RATE",
    )


def _connect_timeout_option() -> Any:
    return typer.Option(
        10.0,
        "--connect-timeout-seconds",
        min=0.001,
        help="连接超时秒数",
        envvar="PRAMA_CLI_CONNECT_TIMEOUT_SECONDS",
    )


def _request_timeout_option() -> Any:
    return typer.Option(
        60.0,
        "--request-timeout-seconds",
        min=0.001,
        help="单次请求超时秒数",
        envvar="PRAMA_CLI_REQUEST_TIMEOUT_SECONDS",
    )


def _output_option(default: str) -> Any:
    return typer.Option(
        Path(default),
        "--output",
        "-o",
        help="逐样本 TSV 输出路径",
        envvar="PRAMA_CLI_OUTPUT",
    )


@app.command()
def asr(
    target: str = _target_option("192.168.0.222:50011"),
    dataset_path: Path = _dataset_path_option("data-bin/audiofolder/asr-demo"),
    split: str = _split_option(),
    limit: int | None = _limit_option(),
    sample_rate: int = _sample_rate_option(),
    connect_timeout_seconds: float | None = _connect_timeout_option(),
    request_timeout_seconds: float = _request_timeout_option(),
    output: Path = _output_option("outputs/evaluate/asr.tsv"),
    language_code: str = typer.Option(
        "en-US",
        "--language-code",
        help="ASR 识别语种",
        envvar="PRAMA_CLI_LANGUAGE_CODE",
    ),
    min_reference_words: int = typer.Option(
        5,
        "--min-reference-words",
        min=0,
        help="参考文本最少词数",
        envvar="PRAMA_CLI_MIN_REFERENCE_WORDS",
    ),
    hotword: list[str] = typer.Option(
        [],
        "--hotword",
        help="ASR 热词，可重复传入",
        envvar="PRAMA_CLI_HOTWORDS",
    ),
    hotword_bias: float = typer.Option(
        0.0,
        "--hotword-bias",
        help="ASR 热词 bias",
        envvar="PRAMA_CLI_HOTWORD_BIAS",
    ),
    inference_concurrency: int = typer.Option(
        0,
        "--inference-concurrency",
        min=0,
        help="ASR 样本级推理并发数，0 表示串行",
        envvar="PRAMA_CLI_ASR_INFERENCE_CONCURRENCY",
    ),
    interim_results: bool = typer.Option(
        True,
        "--interim-results/--no-interim-results",
        help="是否请求 ASR 临时结果",
        envvar="PRAMA_CLI_INTERIM_RESULTS",
    ),
    remove_punctuation: bool = typer.Option(
        False,
        "--remove-punctuation",
        help="评估前移除标点",
        envvar="PRAMA_CLI_REMOVE_PUNCTUATION",
    ),
) -> None:
    request = EvaluationRequest(
        task="asr",
        target=target,
        dataset_path=str(dataset_path),
        split=split,
        limit=limit,
        language_code=language_code,
        sample_rate=sample_rate,
        min_reference_words=min_reference_words,
        hotwords=hotword,
        hotword_bias=hotword_bias,
        connect_timeout_seconds=connect_timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
        interim_results=interim_results,
        asr_inference_concurrency=inference_concurrency,
        remove_punctuation=remove_punctuation,
    )
    _evaluate_to_tsv(request, output)


@app.command()
def vad(
    target: str = _target_option("192.168.0.222:50021"),
    dataset_path: Path = _dataset_path_option("data-bin/audiofolder/vad-demo"),
    split: str = _split_option(),
    limit: int | None = _limit_option(),
    sample_rate: int = _sample_rate_option(),
    connect_timeout_seconds: float | None = _connect_timeout_option(),
    request_timeout_seconds: float = _request_timeout_option(),
    output: Path = _output_option("outputs/evaluate/vad.tsv"),
    mask_frame_seconds: float = typer.Option(
        0.01,
        "--mask-frame-seconds",
        min=0.000001,
        help="VAD mask 帧长秒数",
        envvar="PRAMA_CLI_MASK_FRAME_SECONDS",
    ),
    chunk_duration_seconds: float = typer.Option(
        0.1,
        "--chunk-duration-seconds",
        min=0.000001,
        help="VAD 流式分块秒数",
        envvar="PRAMA_CLI_CHUNK_DURATION_SECONDS",
    ),
    speech_padding_seconds: float = typer.Option(
        0.0,
        "--speech-padding-seconds",
        min=0.0,
        help="检出语音段前后扩展秒数",
        envvar="PRAMA_CLI_SPEECH_PADDING_SECONDS",
    ),
    hit_threshold: float = typer.Option(
        0.9,
        "--hit-threshold",
        min=0.0,
        max=1.0,
        help="参考段计为命中的最小覆盖比例",
        envvar="PRAMA_CLI_HIT_THRESHOLD",
    ),
    streaming: bool = typer.Option(
        False,
        "--streaming",
        help="使用 VAD 流式接口",
        envvar="PRAMA_CLI_VAD_STREAMING",
    ),
    inference_concurrency: int = typer.Option(
        0,
        "--inference-concurrency",
        min=0,
        help="VAD 样本级推理并发数，0 表示串行",
        envvar="PRAMA_CLI_VAD_INFERENCE_CONCURRENCY",
    ),
) -> None:
    request = EvaluationRequest(
        task="vad",
        target=target,
        dataset_path=str(dataset_path),
        split=split,
        limit=limit,
        sample_rate=sample_rate,
        connect_timeout_seconds=connect_timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
        mask_frame_seconds=mask_frame_seconds,
        chunk_duration_seconds=chunk_duration_seconds,
        speech_padding_seconds=speech_padding_seconds,
        hit_threshold=hit_threshold,
        streaming=streaming,
        vad_inference_concurrency=inference_concurrency,
    )
    _evaluate_to_tsv(request, output)


@app.command()
def lid(
    target: str = _target_option("192.168.0.222:50026"),
    dataset_path: Path = _dataset_path_option("data-bin/audiofolder/lid-demo"),
    split: str = _split_option(),
    limit: int | None = _limit_option(),
    sample_rate: int = _sample_rate_option(),
    connect_timeout_seconds: float | None = _connect_timeout_option(),
    request_timeout_seconds: float = _request_timeout_option(),
    output: Path = _output_option("outputs/evaluate/lid.tsv"),
    lid_confidence_threshold: float = typer.Option(
        0.0,
        "--lid-confidence-threshold",
        min=0.0,
        max=1.0,
        help="低于该置信度的预测按 <others> 处理",
        envvar="PRAMA_CLI_LID_CONFIDENCE_THRESHOLD",
    ),
    inference_concurrency: int = typer.Option(
        0,
        "--inference-concurrency",
        min=0,
        help="LID 样本级推理并发数，0 表示串行",
        envvar="PRAMA_CLI_LID_INFERENCE_CONCURRENCY",
    ),
) -> None:
    request = EvaluationRequest(
        task="lid",
        target=target,
        dataset_path=str(dataset_path),
        split=split,
        limit=limit,
        sample_rate=sample_rate,
        connect_timeout_seconds=connect_timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
        lid_confidence_threshold=lid_confidence_threshold,
        lid_inference_concurrency=inference_concurrency,
    )
    _evaluate_to_tsv(request, output)


@app.command()
def denoise(
    target: str = _target_option("192.168.0.222:50031"),
    dataset_path: Path = _dataset_path_option("data-bin/audiofolder/denoise-demo"),
    split: str = _split_option(),
    limit: int | None = _limit_option(),
    sample_rate: int = _sample_rate_option(),
    connect_timeout_seconds: float | None = _connect_timeout_option(),
    request_timeout_seconds: float = _request_timeout_option(),
    output: Path = _output_option("outputs/evaluate/denoise.tsv"),
    mos_target: str = typer.Option(
        "",
        "--mos-target",
        help="MOS gRPC 地址；非空时启用 MOS",
        envvar="PRAMA_CLI_MOS_TARGET",
    ),
    snr_target: str = typer.Option(
        "",
        "--snr-target",
        help="SNR gRPC 地址；非空时启用 SNR",
        envvar="PRAMA_CLI_SNR_TARGET",
    ),
    inference_concurrency: int = typer.Option(
        0,
        "--inference-concurrency",
        min=0,
        help="SE 样本级推理并发数，0 表示串行",
        envvar="PRAMA_CLI_DENOISE_INFERENCE_CONCURRENCY",
    ),
    sqa_inference_concurrency: int = typer.Option(
        0,
        "--sqa-inference-concurrency",
        min=0,
        help="MOS/SNR 推理并发数，0 表示串行",
        envvar="PRAMA_CLI_SQA_INFERENCE_CONCURRENCY",
    ),
) -> None:
    if not mos_target.strip() and not snr_target.strip():
        raise typer.BadParameter("SE 评估必须提供 --mos-target 或 --snr-target")
    request = EvaluationRequest(
        task="denoise",
        target=target,
        dataset_path=str(dataset_path),
        split=split,
        limit=limit,
        sample_rate=sample_rate,
        connect_timeout_seconds=connect_timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
        enable_mos=bool(mos_target.strip()),
        mos_target=mos_target,
        enable_snr=bool(snr_target.strip()),
        snr_target=snr_target,
        inference_concurrency=inference_concurrency,
        sqa_inference_concurrency=sqa_inference_concurrency,
    )
    _evaluate_to_tsv(request, output)


@app.command()
def keyword(
    target: str = _target_option("192.168.0.222:50011"),
    dataset_path: Path = _dataset_path_option("data-bin/audiofolder/keyword-demo"),
    split: str = _split_option(),
    limit: int | None = _limit_option(),
    sample_rate: int = _sample_rate_option(),
    connect_timeout_seconds: float | None = _connect_timeout_option(),
    request_timeout_seconds: float = _request_timeout_option(),
    output: Path = _output_option("outputs/evaluate/keyword.tsv"),
    language_code: str = typer.Option(
        "en-US",
        "--language-code",
        help="ASR 识别语种",
        envvar="PRAMA_CLI_LANGUAGE_CODE",
    ),
    hotword: list[str] = typer.Option(
        [],
        "--hotword",
        help="ASR 热词，可重复传入",
        envvar="PRAMA_CLI_HOTWORDS",
    ),
    hotword_bias: float = typer.Option(
        0.0,
        "--hotword-bias",
        help="ASR 热词 bias",
        envvar="PRAMA_CLI_HOTWORD_BIAS",
    ),
    inference_concurrency: int = typer.Option(
        0,
        "--inference-concurrency",
        min=0,
        help="关键词音频级推理并发数，0 表示串行",
        envvar="PRAMA_CLI_KEYWORD_INFERENCE_CONCURRENCY",
    ),
    interim_results: bool = typer.Option(
        True,
        "--interim-results/--no-interim-results",
        help="是否请求 ASR 临时结果",
        envvar="PRAMA_CLI_INTERIM_RESULTS",
    ),
) -> None:
    request = EvaluationRequest(
        task="keyword",
        target=target,
        dataset_path=str(dataset_path),
        split=split,
        limit=limit,
        language_code=language_code,
        sample_rate=sample_rate,
        hotwords=hotword,
        hotword_bias=hotword_bias,
        connect_timeout_seconds=connect_timeout_seconds,
        request_timeout_seconds=request_timeout_seconds,
        interim_results=interim_results,
        asr_inference_concurrency=inference_concurrency,
    )
    _evaluate_to_tsv(request, output)


def _evaluate_to_tsv(request: EvaluationRequest, output: Path) -> None:
    result = run_cli_evaluation(request)
    write_task_tsv(task=request.task, result=result, output_path=output)
    summary_keys = {
        "asr": ("word_accuracy", "character_accuracy", "wer", "cer"),
        "vad": ("frame_f1", "segment_f1"),
        "lid": ("known_accuracy", "macro_precision", "macro_recall"),
        "denoise": ("mean_snr_delta", "mean_mos_delta"),
        "keyword": ("accuracy", "precision", "recall", "f1"),
    }[request.task]
    summary = " ".join(f"{key}={result.get(key)}" for key in summary_keys)
    logger.info("评估完成: task=%s %s output=%s", request.task, summary, output)


def run_cli_evaluation(request: EvaluationRequest) -> dict[str, Any]:
    job_id = uuid4().hex
    job = EvaluationJob(job_id=job_id, request=request)
    event_queue = message_manager.register_job(job_id)
    thread = threading.Thread(target=_run_evaluation, args=(job,), daemon=True)
    thread.start()

    with tqdm(total=None, unit="sample", desc=request.task.upper()) as progress_bar:
        while True:
            item = event_queue.get()
            if item is MESSAGE_SENTINEL:
                break
            if not isinstance(item, ManagedMessage):
                continue
            payload = item.payload
            next_total = _as_int(payload.get("total"))
            if next_total is not None and next_total != progress_bar.total:
                progress_bar.total = next_total
                progress_bar.refresh()
            processed = _as_int(payload.get("processed"))
            if processed is not None and processed > progress_bar.n:
                progress_bar.update(processed - progress_bar.n)
            current_id = payload.get("current_id") or payload.get("id")
            if current_id is not None:
                progress_bar.set_postfix_str(str(current_id), refresh=False)

    thread.join()
    snapshot = job.snapshot()
    if snapshot["status"] != "completed":
        raise RuntimeError(f"评估失败: {snapshot.get('error') or snapshot['status']}")
    result = snapshot.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("评估完成但没有生成结果")
    return _attach_audio_paths(result, job)


_TSV_HEADERS = {
    "asr": [
        "index",
        "id",
        "audio_file",
        "duration_seconds",
        "wer",
        "cer",
        "reference",
        "hypothesis",
    ],
    "vad": [
        "index",
        "id",
        "audio_file",
        "duration_seconds",
        "frame_accuracy",
        "frame_precision",
        "frame_recall",
        "frame_f1",
        "segment_precision",
        "segment_recall",
        "segment_f1",
        "reference_segment_count",
        "prediction_segment_count",
        "segment_miss_count",
        "segment_false_alarm_count",
    ],
    "lid": [
        "index",
        "id",
        "audio_file",
        "duration_seconds",
        "reference_language",
        "predicted_language",
        "raw_language",
        "confidence",
        "correct",
    ],
    "denoise": [
        "index",
        "id",
        "audio_file",
        "denoised_audio_file",
        "duration_seconds",
        "error",
        "original_snr",
        "denoised_snr",
        "snr_delta",
        "original_mos",
        "denoised_mos",
        "mos_delta",
    ],
    "keyword": [
        "index",
        "id",
        "audio_file",
        "keyword",
        "expected_hit",
        "predicted_hit",
        "correct",
        "transcript",
        "match_text",
    ],
}


def write_task_tsv(*, task: str, result: dict[str, Any], output_path: Path) -> None:
    row_builders = {
        "asr": _asr_rows,
        "vad": _vad_rows,
        "lid": _lid_rows,
        "denoise": _denoise_rows,
        "keyword": _keyword_rows,
    }
    builder = row_builders.get(task)
    headers = _TSV_HEADERS.get(task)
    if builder is None or headers is None:
        raise ValueError(f"不支持的任务类型: {task}")
    _write_rows(output_path, builder(result), headers=headers)


def _attach_audio_paths(result: dict[str, Any], job: EvaluationJob) -> dict[str, Any]:
    with job.lock:
        records = {
            sample_id: dict(record)
            for sample_id, record in job.sample_records.items()
        }
    for report_key in (
        "vad_report",
        "lid_report",
        "denoise_report",
        "keyword_report",
        "keyword_audio_report",
    ):
        report = result.get(report_key)
        if not isinstance(report, dict):
            continue
        for sample in report.get("samples") or []:
            if not isinstance(sample, dict):
                continue
            record_id = sample.get("audio_id") or sample.get("id")
            record = records.get(str(record_id or ""))
            if record is None:
                continue
            sample["audio_file"] = record.get("audio_path")
            sample["denoised_audio_file"] = record.get("denoised_audio_path")
    for report_key in ("wer_report", "cer_report"):
        report = result.get(report_key)
        if not isinstance(report, dict):
            continue
        for utterance in report.get("utterances") or []:
            if not isinstance(utterance, dict):
                continue
            record = records.get(str(utterance.get("id") or ""))
            if record is not None:
                utterance["audio_file"] = record.get("audio_path")
    return result


def _asr_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    wer_items = _report_samples(result, "wer_report", item_key="utterances")
    cer_by_id = {
        str(item.get("id") or ""): item
        for item in _report_samples(result, "cer_report", item_key="utterances")
    }
    rows: list[dict[str, Any]] = []
    for item in wer_items:
        sample_id = str(item.get("id") or "")
        cer_summary = (cer_by_id.get(sample_id) or {}).get("summary") or {}
        summary = item.get("summary") or {}
        tokens = item.get("tokens") or []
        rows.append(
            {
                "index": item.get("index", ""),
                "id": sample_id,
                "audio_file": item.get("audio_file", ""),
                "duration_seconds": item.get("duration_seconds", ""),
                "wer": summary.get("wer", ""),
                "cer": cer_summary.get("wer", ""),
                "reference": _tokens_to_text(tokens, "ref"),
                "hypothesis": _tokens_to_text(tokens, "hyp"),
            }
        )
    return rows


def _vad_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in _report_samples(result, "vad_report"):
        metrics = sample.get("metrics") or {}
        rows.append(
            {
                "index": sample.get("index", ""),
                "id": sample.get("id", ""),
                "audio_file": sample.get("audio_file", ""),
                "duration_seconds": sample.get("duration_seconds", ""),
                **{
                    key: metrics.get(key, "")
                    for key in _TSV_HEADERS["vad"]
                    if key
                    not in {"index", "id", "audio_file", "duration_seconds"}
                },
            }
        )
    return rows


def _lid_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: sample.get(key, "") for key in _TSV_HEADERS["lid"]}
        for sample in _report_samples(result, "lid_report")
    ]


def _denoise_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: sample.get(key, "") for key in _TSV_HEADERS["denoise"]}
        for sample in _report_samples(result, "denoise_report")
    ]


def _keyword_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: sample.get(key, "") for key in _TSV_HEADERS["keyword"]}
        for sample in _report_samples(result, "keyword_report")
    ]


def _write_rows(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    headers: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=headers,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("TSV 已写入: path=%s rows=%s", output_path, len(rows))


def _report_samples(
    result: dict[str, Any],
    report_key: str,
    *,
    item_key: str = "samples",
) -> list[dict[str, Any]]:
    report = result.get(report_key) or {}
    items = report.get(item_key) if isinstance(report, dict) else []
    return [item for item in (items or []) if isinstance(item, dict)]


def _tokens_to_text(tokens: list[dict[str, Any]], key: str) -> str:
    return " ".join(
        str(token[key])
        for token in tokens
        if isinstance(token, dict) and token.get(key) not in (None, "")
    )


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, int) else None
