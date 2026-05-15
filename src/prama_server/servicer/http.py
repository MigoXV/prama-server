from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from datasets import Audio, Dataset, load_dataset
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from prama.evaluator.evaluator import get_wer

from prama_server.evaluator import (
    EvaluationInferenceResult,
    Evaluator,
)
from prama_server.inferencers.asr import AsrGrpcInferencer
from prama_server.message_manager import MESSAGE_SENTINEL, ManagedMessage, MessageManager

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

JobStatus = Literal["queued", "running", "completed", "failed"]


class EvaluationRequest(BaseModel):
    target: str = Field("192.168.1.24:50008", description="ASR gRPC 服务地址")
    dataset_path: str = Field("data-bin/jacktol/ATC-ASR-Dataset", description="数据集路径")
    split: str = Field("test", description="数据集 split")
    limit: int | None = Field(None, ge=1, description="最多评估的样本数")
    language_code: str = Field("en-US", description="识别语言")
    sample_rate: int = Field(16000, ge=1, description="音频采样率")
    min_reference_words: int = Field(5, ge=0, description="参考文本最少词数")
    hotwords: list[str] = Field(default_factory=lambda: ["HOTEL"], description="热词")
    hotword_bias: float = Field(0.0, description="热词 bias")
    connect_timeout_seconds: float | None = Field(10.0, gt=0, description="连接超时")
    request_timeout_seconds: float = Field(60.0, gt=0, description="单次请求超时")


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
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

jobs: dict[str, EvaluationJob] = {}
jobs_lock = threading.Lock()
message_manager = MessageManager()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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
            yield _format_sse("inference_result", snapshot["progress"])
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
        logger.info("评估任务开始: job_id=%s dataset=%s", job.job_id, request.dataset_path)
        inferencer = AsrGrpcInferencer(
            target=request.target,
            sample_rate=request.sample_rate,
            language_code=request.language_code,
            hotwords=request.hotwords,
            hotword_bias=request.hotword_bias,
            request_timeout_seconds=request.request_timeout_seconds,
            connect_timeout_seconds=request.connect_timeout_seconds,
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
        ) as evaluator:
            metrics = evaluator.iter_evaluate(
                inferencer,
                on_infer_result=publish_infer_result,
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
                "tokens": [
                    {
                        "label": token.eval_label,
                        "ref": token.ref_word,
                        "hyp": token.hyp_word,
                    }
                    for token in utterance.tokens
                ],
            }
            for utterance in wer_result.utterances
        ],
    }


def _format_sse(event_name: str, payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_name}\ndata: {data}\n\n"
