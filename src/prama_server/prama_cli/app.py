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
    message_manager,
    _run_evaluation,
)

logger = logging.getLogger(__name__)


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
    return typer.Option(
        "test",
        "--split",
        help="数据集 split",
        envvar="PRAMA_CLI_SPLIT",
    )


def _limit_option() -> Any:
    return typer.Option(
        None,
        "--limit",
        help="最多评估样本数",
        envvar="PRAMA_CLI_LIMIT",
    )


def _sample_rate_option() -> Any:
    return typer.Option(
        16000,
        "--sample-rate",
        help="音频采样率",
        envvar="PRAMA_CLI_SAMPLE_RATE",
    )


def _connect_timeout_option() -> Any:
    return typer.Option(
        10.0,
        "--connect-timeout-seconds",
        help="连接超时秒数",
        envvar="PRAMA_CLI_CONNECT_TIMEOUT_SECONDS",
    )


def _request_timeout_option() -> Any:
    return typer.Option(
        60.0,
        "--request-timeout-seconds",
        help="单次请求超时秒数",
        envvar="PRAMA_CLI_REQUEST_TIMEOUT_SECONDS",
    )


def _output_option(default: str) -> Any:
    return typer.Option(
        Path(default),
        "--output",
        "-o",
        help="输出 TSV 路径",
        envvar="PRAMA_CLI_OUTPUT",
    )


app = typer.Typer(help="命令行评估工具。")

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
        "召回率",
        "准确率",
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
    result = run_cli_evaluation(request)
    write_task_tsv(task="asr", result=result, output_path=output)
    _log_summary(task="asr", result=result, output=output)


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
        help="VAD mask 帧长秒数",
        envvar="PRAMA_CLI_MASK_FRAME_SECONDS",
    ),
    chunk_duration_seconds: float = typer.Option(
        0.1,
        "--chunk-duration-seconds",
        help="VAD 流式分块秒数",
        envvar="PRAMA_CLI_CHUNK_DURATION_SECONDS",
    ),
    speech_padding_seconds: float = typer.Option(
        0.0,
        "--speech-padding-seconds",
        help="VAD 检出语音段前后扩展秒数",
        envvar="PRAMA_CLI_SPEECH_PADDING_SECONDS",
    ),
    hit_threshold: float = typer.Option(
        0.9,
        "--hit-threshold",
        help="VAD 段命中阈值",
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
    result = run_cli_evaluation(request)
    write_task_tsv(task="vad", result=result, output_path=output)
    _log_summary(task="vad", result=result, output=output)


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
        help="LID 置信度阈值，低于该值记为 <others>",
        envvar="PRAMA_CLI_LID_CONFIDENCE_THRESHOLD",
    ),
    inference_concurrency: int = typer.Option(
        0,
        "--inference-concurrency",
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
    result = run_cli_evaluation(request)
    write_task_tsv(task="lid", result=result, output_path=output)
    _log_summary(task="lid", result=result, output=output)


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
        help="MOS gRPC 服务地址；非空时启用 MOS",
        envvar="PRAMA_CLI_MOS_TARGET",
    ),
    snr_target: str = typer.Option(
        "",
        "--snr-target",
        help="SNR gRPC 服务地址；非空时启用 SNR",
        envvar="PRAMA_CLI_SNR_TARGET",
    ),
    inference_concurrency: int = typer.Option(
        0,
        "--inference-concurrency",
        help="SE 样本级推理并发数，0 表示串行",
        envvar="PRAMA_CLI_DENOISE_INFERENCE_CONCURRENCY",
    ),
    sqa_inference_concurrency: int = typer.Option(
        0,
        "--sqa-inference-concurrency",
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
    result = run_cli_evaluation(request)
    write_task_tsv(task="denoise", result=result, output_path=output)
    _log_summary(task="denoise", result=result, output=output)


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
        help="关键词评估样本级推理并发数，0 表示串行",
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
    result = run_cli_evaluation(request)
    write_task_tsv(task="keyword", result=result, output_path=output)
    _log_summary(task="keyword", result=result, output=output)


def run_cli_evaluation(request: EvaluationRequest) -> dict[str, Any]:
    job_id = uuid4().hex
    job = EvaluationJob(job_id=job_id, request=request)
    event_queue = message_manager.register_job(job_id)
    thread = threading.Thread(target=_run_evaluation, args=(job,), daemon=True)
    thread.start()

    total: int | None = None
    with tqdm(total=total, unit="sample", desc=request.task.upper()) as progress_bar:
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


def write_task_tsv(*, task: str, result: dict[str, Any], output_path: Path) -> None:
    if task == "asr":
        _write_rows(output_path, _asr_rows(result), headers=_TSV_HEADERS["asr"])
        return
    if task == "vad":
        _write_rows(output_path, _vad_rows(result), headers=_TSV_HEADERS["vad"])
        return
    if task == "lid":
        _write_rows(output_path, _lid_rows(result), headers=_TSV_HEADERS["lid"])
        return
    if task == "denoise":
        _write_rows(output_path, _denoise_rows(result), headers=_TSV_HEADERS["denoise"])
        return
    if task == "keyword":
        _write_rows(output_path, _keyword_rows(result), headers=_TSV_HEADERS["keyword"])
        return
    raise ValueError(f"不支持的任务类型: {task}")


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
    ):
        report = result.get(report_key)
        if not isinstance(report, dict):
            continue
        for sample in report.get("samples") or []:
            if not isinstance(sample, dict):
                continue
            record = records.get(str(sample.get("id") or ""))
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
                "召回率": metrics.get("frame_recall", ""),
                "准确率": metrics.get("frame_precision", ""),
            }
        )
    return rows


def _lid_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "index": sample.get("index", ""),
            "id": sample.get("id", ""),
            "audio_file": sample.get("audio_file", ""),
            "duration_seconds": sample.get("duration_seconds", ""),
            "reference_language": sample.get("reference_language", ""),
            "predicted_language": sample.get("predicted_language", ""),
            "raw_language": sample.get("raw_language", ""),
            "confidence": sample.get("confidence", ""),
            "correct": sample.get("correct", ""),
        }
        for sample in _report_samples(result, "lid_report")
    ]


def _denoise_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "index": sample.get("index", ""),
            "id": sample.get("id", ""),
            "audio_file": sample.get("audio_file", ""),
            "denoised_audio_file": sample.get("denoised_audio_file", ""),
            "duration_seconds": sample.get("duration_seconds", ""),
            "error": sample.get("error", ""),
            "original_snr": sample.get("original_snr", ""),
            "denoised_snr": sample.get("denoised_snr", ""),
            "snr_delta": sample.get("snr_delta", ""),
            "original_mos": sample.get("original_mos", ""),
            "denoised_mos": sample.get("denoised_mos", ""),
            "mos_delta": sample.get("mos_delta", ""),
        }
        for sample in _report_samples(result, "denoise_report")
    ]


def _keyword_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "index": sample.get("index", ""),
            "id": sample.get("id", ""),
            "audio_file": sample.get("audio_file", ""),
            "keyword": sample.get("keyword", ""),
            "expected_hit": sample.get("expected_hit", ""),
            "predicted_hit": sample.get("predicted_hit", ""),
            "correct": sample.get("correct", ""),
            "transcript": sample.get("transcript", ""),
            "match_text": sample.get("match_text", ""),
        }
        for sample in _report_samples(result, "keyword_report")
    ]


def _write_rows(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    headers: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
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


def _log_summary(*, task: str, result: dict[str, Any], output: Path) -> None:
    summary_keys = {
        "asr": ("wer", "cer"),
        "vad": ("frame_recall", "frame_precision"),
        "lid": ("precision", "recall"),
        "denoise": ("mean_snr_delta", "mean_mos_delta"),
        "keyword": ("accuracy", "precision", "recall", "f1"),
    }[task]
    summary = " ".join(f"{key}={result.get(key)}" for key in summary_keys)
    logger.info("评估完成: task=%s %s output=%s", task, summary, output)


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
