from __future__ import annotations

import json
import logging
import threading
import unicodedata
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import librosa
import numpy as np
from datasets import Audio, Dataset, DatasetDict, load_dataset, load_from_disk
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from prama.evaluator.evaluator import get_wer

from prama_server.evaluator import (
    EvaluationInferenceResult,
    EvaluationPartialInferenceResult,
    Evaluator,
)
from prama_server.evaluator.vad import VadEvaluator
from prama_server.inferencers.asr import AsrGrpcInferencer
from prama_server.inferencers.vad import VadGrpcInferencer
from prama_server.message_manager import MESSAGE_SENTINEL, ManagedMessage, MessageManager

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "completed", "failed"]
EvaluationTask = Literal["asr", "vad"]


class EvaluationRequest(BaseModel):
    task: EvaluationTask = Field("asr", description="评估任务类型")
    target: str = Field("192.168.0.222:50011", description="ASR gRPC 服务地址")
    dataset_path: str = Field("data-bin/audiofolder/asr-demo", description="数据集路径")
    split: str = Field("test", description="数据集 split")
    limit: int | None = Field(None, ge=1, description="最多评估的样本数")
    language_code: str = Field("en-US", description="识别语言")
    sample_rate: int = Field(16000, ge=1, description="音频采样率")
    min_reference_words: int = Field(5, ge=0, description="参考文本最少词数")
    hotwords: list[str] = Field(default_factory=list, description="热词")
    hotword_bias: float = Field(0.0, description="热词 bias")
    connect_timeout_seconds: float | None = Field(10.0, gt=0, description="连接超时")
    request_timeout_seconds: float = Field(60.0, gt=0, description="单次请求超时")
    interim_results: bool = Field(True, description="是否请求 ASR 临时结果")
    remove_punctuation: bool = Field(False, description="ASR 评估时是否移除标点")
    mask_frame_seconds: float = Field(0.01, gt=0, description="VAD mask 帧长秒数")
    chunk_duration_seconds: float = Field(0.1, gt=0, description="VAD 流式分块秒数")
    hit_threshold: float = Field(0.9, ge=0, le=1, description="VAD 段命中阈值")
    streaming: bool = Field(False, description="是否使用 VAD 流式接口")


class EvaluationCreated(BaseModel):
    job_id: str
    status: JobStatus


@dataclass
class EvaluationJob:
    job_id: str
    request: EvaluationRequest
    status: JobStatus = "queued"
    latest_progress: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "job_id": self.job_id,
                "status": self.status,
                "request": self.request.model_dump(),
                "progress": self.latest_progress,
                "result": self.result,
                "error": self.error,
            }


app = FastAPI(title="Prama ASR Evaluation Service")

jobs: dict[str, EvaluationJob] = {}
jobs_lock = threading.Lock()
message_manager = MessageManager()


@app.get("/")
def index() -> dict[str, str]:
    return {"name": "Prama ASR Evaluation Service", "status": "ok"}


@app.post("/api/evaluations", response_model=EvaluationCreated)
def create_evaluation(request: EvaluationRequest) -> EvaluationCreated:
    job_id = uuid4().hex
    job = EvaluationJob(job_id=job_id, request=request)
    message_manager.register_job(job_id)
    with jobs_lock:
        jobs[job_id] = job

    thread = threading.Thread(target=_run_evaluation, args=(job,), daemon=True)
    thread.start()
    logger.info("评估任务已创建: job_id=%s target=%s", job_id, request.target)
    return EvaluationCreated(job_id=job_id, status=job.status)


@app.get("/api/evaluations/{job_id}")
def get_evaluation(job_id: str) -> dict[str, Any]:
    return _get_job(job_id).snapshot()


@app.get("/api/evaluations/{job_id}/events")
def stream_evaluation_events(job_id: str) -> StreamingResponse:
    job = _get_job(job_id)
    event_queue = message_manager.subscribe(job_id)

    def event_stream() -> Any:
        snapshot = job.snapshot()
        if snapshot["progress"] is not None:
            event_name = (
                "inference_result"
                if snapshot["progress"].get("is_final", True)
                else "partial_inference_result"
            )
            yield _format_sse(event_name, snapshot["progress"])
        if snapshot["status"] == "failed" and snapshot["error"] is not None:
            yield _format_sse("error", {"message": snapshot["error"]})
            yield _format_sse("done", snapshot)
            return
        if snapshot["status"] == "completed":
            yield _format_sse("done", snapshot)
            return

        while True:
            item = event_queue.get()
            if item is MESSAGE_SENTINEL:
                yield _format_sse("done", job.snapshot())
                return

            if isinstance(item, ManagedMessage):
                yield _format_sse(item.event_name, item.payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _get_job(job_id: str) -> EvaluationJob:
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"评估任务不存在: {job_id}")
    return job


def _run_evaluation(job: EvaluationJob) -> None:
    request = job.request
    with job.lock:
        job.status = "running"

    try:
        logger.info(
            "评估任务开始: job_id=%s task=%s dataset=%s",
            job.job_id,
            request.task,
            request.dataset_path,
        )
        if request.task == "vad":
            _run_vad_evaluation(job)
            return

        inferencer = AsrGrpcInferencer(
            target=request.target,
            sample_rate=request.sample_rate,
            language_code=request.language_code,
            hotwords=request.hotwords,
            hotword_bias=request.hotword_bias,
            request_timeout_seconds=request.request_timeout_seconds,
            connect_timeout_seconds=request.connect_timeout_seconds,
            interim_results=request.interim_results,
        )
        dataset = _load_evaluation_dataset(
            Path(request.dataset_path),
            split=request.split,
            limit=request.limit,
            sample_rate=request.sample_rate,
            min_reference_words=request.min_reference_words,
        )
        total = len(dataset)
        evaluated = 0
        inference_rows: list[dict[str, str]] = []

        def publish_partial_infer_result(
            result: EvaluationPartialInferenceResult,
        ) -> None:
            if result.is_final:
                return
            payload = {
                "status": "running",
                "tag": result.tag,
                "total": total,
                "processed": evaluated,
                "evaluated": evaluated,
                "id": result.id,
                "current_id": result.id,
                "reference": result.reference,
                "hypothesis": result.hypothesis,
                "is_final": result.is_final,
            }
            with job.lock:
                job.latest_progress = payload
            message_manager.put(
                ManagedMessage(
                    job_id=job.job_id,
                    event_name="partial_inference_result",
                    payload=payload,
                )
            )

        def publish_infer_result(result: EvaluationInferenceResult) -> None:
            nonlocal evaluated
            evaluated += 1
            inference_rows.append(
                {
                    "id": result.id,
                    "reference": result.reference,
                    "hypothesis": result.hypothesis,
                }
            )
            payload = {
                "status": "running",
                "tag": result.tag,
                "total": total,
                "processed": evaluated,
                "evaluated": evaluated,
                "id": result.id,
                "current_id": result.id,
                "reference": result.reference,
                "hypothesis": result.hypothesis,
                "is_final": True,
            }
            with job.lock:
                job.latest_progress = payload
            message_manager.put(
                ManagedMessage(
                    job_id=job.job_id,
                    event_name="inference_result",
                    payload=payload,
                )
            )

        with Evaluator(
            dataset=dataset,
            sample_rate=request.sample_rate,
            tag=job.job_id,
            reference_postprocess=(
                remove_punctuation if request.remove_punctuation else None
            ),
            hypothesis_postprocess=(
                remove_punctuation if request.remove_punctuation else None
            ),
        ) as evaluator:
            metrics = evaluator.iter_evaluate(
                inferencer,
                on_infer_result=publish_infer_result,
                on_partial_infer_result=publish_partial_infer_result,
            )

        report = _build_wer_report(inference_rows)
        with job.lock:
            job.status = "completed"
            job.result = {**metrics, "wer_report": report}
        logger.info("评估任务完成: job_id=%s", job.job_id)
    except Exception as exc:  # noqa: BLE001 - 后台任务需要把异常传给前端
        logger.exception("评估任务失败: job_id=%s", job.job_id)
        with job.lock:
            job.status = "failed"
            job.error = str(exc)
        message_manager.put(
            ManagedMessage(
                job_id=job.job_id,
                event_name="error",
                payload={"message": str(exc)},
            )
        )
    finally:
        message_manager.close_job(job.job_id)


def _load_evaluation_dataset(
    dataset_path: Path,
    *,
    split: str,
    limit: int | None,
    sample_rate: int,
    min_reference_words: int | None,
) -> Dataset:
    logger.info("加载数据集: path=%s split=%s", dataset_path, split)
    if _is_audiofolder_dataset_dir(dataset_path):
        dataset = load_dataset("audiofolder", data_dir=str(dataset_path), split=split)
    else:
        dataset = load_dataset(str(dataset_path), split=split)
    dataset = dataset.cast_column(
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


def remove_punctuation(text: str) -> str:
    without_punctuation = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )
    return " ".join(without_punctuation.split())


def _run_vad_evaluation(job: EvaluationJob) -> None:
    request = job.request
    inferencer = VadGrpcInferencer(
        target=request.target,
        sample_rate=request.sample_rate,
        mask_frame_seconds=request.mask_frame_seconds,
        chunk_duration_seconds=request.chunk_duration_seconds,
        request_timeout_seconds=request.request_timeout_seconds,
        connect_timeout_seconds=request.connect_timeout_seconds,
    )
    dataset = _load_vad_dataset(
        Path(request.dataset_path),
        split=request.split,
        limit=request.limit,
        sample_rate=request.sample_rate,
    )
    evaluator = VadEvaluator(hit_threshold=request.hit_threshold)
    total = len(dataset)
    rows: list[dict[str, Any]] = []
    report_samples: list[dict[str, Any]] = []

    try:
        for index, sample in enumerate(dataset, start=1):
            sample_id = str(sample.get("id") or sample.get("utt_id") or index)
            audio = sample["audio"]
            audio_array = _prepare_vad_audio(audio, sample_rate=request.sample_rate)
            prediction_mask = (
                inferencer.stream_infer(audio_array)
                if request.streaming
                else inferencer.infer(audio_array)
            )
            reference_mask = _seconds_to_mask(
                sample["seconds"],
                length=len(prediction_mask),
                frame_seconds=request.mask_frame_seconds,
            )
            result = evaluator.evaluate(reference_mask, prediction_mask)
            result_dict = asdict(result)
            rows.append(result_dict)
            report_samples.append(
                _build_vad_sample_report(
                    sample_id=sample_id,
                    index=index,
                    reference_mask=reference_mask,
                    prediction_mask=prediction_mask,
                    frame_seconds=request.mask_frame_seconds,
                    hit_threshold=request.hit_threshold,
                    metrics=result_dict,
                )
            )
            payload = {
                "status": "running",
                "tag": job.job_id,
                "total": total,
                "processed": index,
                "evaluated": index,
                "id": sample_id,
                "current_id": sample_id,
                "reference": _format_vad_segments(reference_mask, request.mask_frame_seconds),
                "hypothesis": _format_vad_segments(
                    prediction_mask,
                    request.mask_frame_seconds,
                ),
                "is_final": True,
                "result": result_dict,
            }
            with job.lock:
                job.latest_progress = payload
            message_manager.put(
                ManagedMessage(
                    job_id=job.job_id,
                    event_name="inference_result",
                    payload=payload,
                )
            )

        with job.lock:
            job.status = "completed"
            job.result = _build_vad_report(rows, samples=report_samples)
        logger.info("VAD 评估任务完成: job_id=%s", job.job_id)
    finally:
        inferencer.close()


def _load_vad_dataset(
    dataset_path: Path,
    *,
    split: str,
    limit: int | None,
    sample_rate: int,
) -> Dataset:
    logger.info("加载 VAD 数据集: path=%s split=%s", dataset_path, split)
    if _is_audiofolder_dataset_dir(dataset_path):
        dataset = load_dataset("audiofolder", data_dir=str(dataset_path), split=split)
    elif dataset_path.exists() and dataset_path.is_dir():
        loaded = load_from_disk(str(dataset_path))
        if isinstance(loaded, DatasetDict):
            if split not in loaded:
                available_splits = ", ".join(loaded.keys())
                raise ValueError(
                    f"数据集不包含 split '{split}'，可用 split: {available_splits}"
                )
            dataset = loaded[split]
        else:
            dataset = loaded
    else:
        dataset = load_dataset(str(dataset_path), split=split)

    missing = sorted({"audio", "seconds"} - set(dataset.column_names))
    if missing:
        raise ValueError(f"VAD 数据集缺少必要字段: {', '.join(missing)}")

    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    logger.info("VAD 数据集已加载: size=%s", len(dataset))
    return dataset


def _is_audiofolder_dataset_dir(dataset_path: Path) -> bool:
    if not dataset_path.exists() or not dataset_path.is_dir():
        return False
    if (dataset_path / "metadata.jsonl").exists():
        return True
    return any(
        (dataset_path / split / "metadata.jsonl").exists()
        for split in ("train", "validation", "test")
    )


def _prepare_vad_audio(audio: Any, *, sample_rate: int) -> np.ndarray:
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        data = samples.data
        if hasattr(data, "numpy"):
            audio_array = data.numpy()
        else:
            audio_array = np.asarray(data)
        audio_sample_rate = int(samples.sample_rate)
    elif isinstance(audio, dict):
        audio_array = np.asarray(audio["array"])
        audio_sample_rate = int(audio["sampling_rate"])
    else:
        audio_array = np.asarray(audio)
        audio_sample_rate = sample_rate

    audio_array = np.squeeze(audio_array)
    if audio_array.ndim == 2:
        channel_axis = 0 if audio_array.shape[0] <= audio_array.shape[1] else 1
        audio_array = audio_array.mean(axis=channel_axis)
    if audio_array.ndim != 1:
        raise ValueError(f"不支持的音频维度: {audio_array.shape}")

    audio_array = audio_array.astype(np.float32, copy=False)
    if audio_sample_rate != sample_rate:
        audio_array = librosa.resample(
            audio_array,
            orig_sr=audio_sample_rate,
            target_sr=sample_rate,
            res_type="scipy",
        )
    return audio_array.astype(np.float32, copy=False)


def _seconds_to_mask(
    seconds: dict[str, Any],
    *,
    length: int,
    frame_seconds: float,
) -> np.ndarray:
    starts = np.asarray(seconds.get("starts", []), dtype=np.float64)
    durations = np.asarray(seconds.get("durations", []), dtype=np.float64)
    if starts.shape != durations.shape:
        raise ValueError(
            f"seconds.starts 与 seconds.durations 长度不一致: {starts.shape} != {durations.shape}"
        )
    mask = np.zeros(length, dtype=bool)
    for start, duration in zip(starts, durations):
        end = start + duration
        start_index = max(0, int(round(start / frame_seconds)))
        end_index = min(length, int(round(end / frame_seconds)))
        if end_index > start_index:
            mask[start_index:end_index] = True
    return mask


def _format_vad_segments(mask: np.ndarray, frame_seconds: float) -> str:
    segments: list[str] = []
    padded = np.pad(mask.astype(np.int8), (1, 1), mode="constant")
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    for start, end in zip(starts, ends):
        segments.append(f"{start * frame_seconds:.2f}-{end * frame_seconds:.2f}s")
    return ", ".join(segments) if segments else "无语音段"


def _build_vad_report(
    rows: list[dict[str, Any]],
    *,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "frame_accuracy": 0.0,
            "frame_recall": 0.0,
            "frame_precision": 0.0,
            "frame_f1": 0.0,
            "frame": {},
            "segment": {},
            "vad_report": {"samples": []},
            "segment_recall": 0.0,
            "segment_precision": 0.0,
            "reference_segment_count": 0,
            "prediction_segment_count": 0,
        }

    frame_average_keys = [
        "frame_accuracy",
        "frame_recall",
        "frame_precision",
        "frame_f1",
        "frame_specificity",
        "frame_false_alarm_rate",
        "frame_miss_rate",
        "frame_balanced_accuracy",
    ]
    segment_average_keys = [
        "segment_recall",
        "segment_precision",
        "segment_f1",
        "segment_miss_rate",
        "segment_false_alarm_rate",
    ]
    frame_count_keys = [
        "frame_total",
        "frame_speech",
        "frame_non_speech",
        "frame_true_positive",
        "frame_true_negative",
        "frame_false_positive",
        "frame_false_negative",
    ]
    segment_count_keys = [
        "reference_segment_count",
        "prediction_segment_count",
        "segment_hit_count",
        "segment_miss_count",
        "segment_false_alarm_count",
    ]
    frame_metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in frame_average_keys
    }
    frame_metrics.update({key: int(sum(row[key] for row in rows)) for key in frame_count_keys})
    segment_metrics = {
        key: float(np.mean([row[key] for row in rows]))
        for key in segment_average_keys
    }
    segment_metrics.update(
        {key: int(sum(row[key] for row in rows)) for key in segment_count_keys}
    )
    report = {
        "frame": frame_metrics,
        "segment": segment_metrics,
        "vad_report": {"samples": samples},
    }
    report.update(frame_metrics)
    report.update(segment_metrics)
    report["sample_count"] = len(rows)
    return report


def _build_vad_sample_report(
    *,
    sample_id: str,
    index: int,
    reference_mask: np.ndarray,
    prediction_mask: np.ndarray,
    frame_seconds: float,
    hit_threshold: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    reference_segments = _mask_to_segments(reference_mask)
    prediction_segments = _mask_to_segments(prediction_mask)
    return {
        "id": sample_id,
        "index": index,
        "duration_seconds": len(reference_mask) * frame_seconds,
        "frame_seconds": frame_seconds,
        "metrics": metrics,
        "reference_segments": _classify_reference_segments(
            reference_segments,
            prediction_segments,
            frame_seconds=frame_seconds,
            hit_threshold=hit_threshold,
        ),
        "prediction_segments": _classify_prediction_segments(
            prediction_segments,
            reference_segments,
            frame_seconds=frame_seconds,
            hit_threshold=hit_threshold,
        ),
        "regions": _build_vad_regions(
            reference_mask,
            prediction_mask,
            frame_seconds=frame_seconds,
        ),
    }


def _mask_to_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1), mode="constant")
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _classify_reference_segments(
    reference_segments: list[tuple[int, int]],
    prediction_segments: list[tuple[int, int]],
    *,
    frame_seconds: float,
    hit_threshold: float,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for segment in reference_segments:
        status = (
            "hit"
            if _has_segment_hit(segment, prediction_segments, hit_threshold)
            else "miss"
        )
        segments.append(_segment_to_payload(segment, frame_seconds, status=status))
    return segments


def _classify_prediction_segments(
    prediction_segments: list[tuple[int, int]],
    reference_segments: list[tuple[int, int]],
    *,
    frame_seconds: float,
    hit_threshold: float,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for segment in prediction_segments:
        status = (
            "hit"
            if _has_segment_overlap(segment, reference_segments)
            else "false_alarm"
        )
        segments.append(_segment_to_payload(segment, frame_seconds, status=status))
    return segments


def _segment_to_payload(
    segment: tuple[int, int],
    frame_seconds: float,
    *,
    status: str,
) -> dict[str, Any]:
    start_frame, end_frame = segment
    return {
        "start": start_frame * frame_seconds,
        "end": end_frame * frame_seconds,
        "duration": (end_frame - start_frame) * frame_seconds,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "status": status,
    }


def _has_segment_hit(
    source: tuple[int, int],
    targets: list[tuple[int, int]],
    hit_threshold: float,
) -> bool:
    source_len = source[1] - source[0]
    if source_len <= 0:
        return False
    return any(
        _segment_overlap(source, target) / source_len >= hit_threshold
        for target in targets
    )


def _has_segment_overlap(
    source: tuple[int, int],
    targets: list[tuple[int, int]],
) -> bool:
    return any(_segment_overlap(source, target) > 0 for target in targets)


def _segment_overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _build_vad_regions(
    reference_mask: np.ndarray,
    prediction_mask: np.ndarray,
    *,
    frame_seconds: float,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    if len(reference_mask) == 0:
        return regions

    labels = np.full(len(reference_mask), "correct_reject", dtype=object)
    labels[reference_mask & prediction_mask] = "hit"
    labels[reference_mask & ~prediction_mask] = "miss"
    labels[~reference_mask & prediction_mask] = "false_alarm"

    start = 0
    current_label = str(labels[0])
    for index in range(1, len(labels)):
        label = str(labels[index])
        if label == current_label:
            continue
        regions.append(
            _region_to_payload(start, index, current_label, frame_seconds)
        )
        start = index
        current_label = label
    regions.append(_region_to_payload(start, len(labels), current_label, frame_seconds))
    return regions


def _region_to_payload(
    start_frame: int,
    end_frame: int,
    label: str,
    frame_seconds: float,
) -> dict[str, Any]:
    return {
        "start": start_frame * frame_seconds,
        "end": end_frame * frame_seconds,
        "duration": (end_frame - start_frame) * frame_seconds,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "label": label,
    }


def _build_wer_report(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "summary": {
                "ref_words": 0,
                "hyp_words": 0,
                "correct": 0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "sentence_count": 0,
                "sentence_errors": 0,
                "wer": 0.0,
                "accuracy": 0.0,
            },
            "utterances": [],
        }

    wer_result = get_wer(
        [row["reference"] for row in rows],
        [row["hypothesis"] for row in rows],
        [row["id"] for row in rows],
    )
    summary = wer_result.summary
    utterance_summaries = {
        group.name.strip("()"): group.counts for group in wer_result.groups
    }
    return {
        "summary": {
            "ref_words": summary.ref_words,
            "hyp_words": summary.hyp_words,
            "correct": summary.correct,
            "substitutions": summary.substitutions,
            "deletions": summary.deletions,
            "insertions": summary.insertions,
            "sentence_count": summary.sentence_count,
            "sentence_errors": summary.sentence_errors,
            "wer": summary.wer,
            "accuracy": summary.accuracy,
        },
        "utterances": [
            {
                "id": utterance.id.strip("()"),
                "index": index,
                "summary": _wer_counts_to_payload(
                    utterance_summaries[utterance.id.strip("()")]
                ),
                "tokens": [
                    {
                        "label": token.eval_label,
                        "ref": token.ref_word,
                        "hyp": token.hyp_word,
                    }
                    for token in utterance.tokens
                ],
            }
            for index, utterance in enumerate(wer_result.utterances, start=1)
        ],
    }


def _wer_counts_to_payload(counts: Any) -> dict[str, Any]:
    return {
        "ref_words": counts.ref_words,
        "hyp_words": counts.hyp_words,
        "correct": counts.correct,
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
        "sentence_count": counts.sentence_count,
        "sentence_errors": counts.sentence_errors,
        "wer": counts.wer,
        "accuracy": counts.accuracy,
    }


def _format_sse(event_name: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n"
