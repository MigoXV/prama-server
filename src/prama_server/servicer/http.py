from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import tempfile
import time
import unicodedata
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote
from uuid import uuid4

import librosa
import grpc
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, Dataset, load_dataset
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator
from prama.evaluator.evaluator import get_cer, get_wer

from prama_server.evaluator import (
    EvaluationInferenceResult,
    EvaluationPartialInferenceResult,
    Evaluator,
)
from prama_server.evaluator.vad import VadEvaluator
from prama_server.inferencers.asr import AsrGrpcInferencer
from prama_server.inferencers.denoise import DenoiseGrpcInferencer
from prama_server.inferencers.grpc_options import create_insecure_channel
from prama_server.inferencers.lid import LidGrpcInferencer
from prama_server.inferencers.sqa import SqaGrpcInferencer
from prama_server.inferencers.vad import VadGrpcInferencer
from prama_server.metrics.prama_warpper import get_cer_pd, get_wer_pd
from prama_server.message_manager import MESSAGE_SENTINEL, ManagedMessage, MessageManager

logger = logging.getLogger(__name__)

JobStatus = Literal["queued", "running", "completed", "failed"]
EvaluationTask = Literal["asr", "vad", "lid", "denoise", "keyword"]
LID_UNKNOWN_LANGUAGE = "<others>"
VAD_REPORT_WEIGHT_EPS = 1e-9


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
    inference_concurrency: int = Field(
        0,
        ge=0,
        description="兼容旧字段；样本级推理并发数，0 表示串行",
    )
    asr_inference_concurrency: int = Field(0, ge=0, description="ASR 样本级推理并发数")
    vad_inference_concurrency: int = Field(0, ge=0, description="VAD 样本级推理并发数")
    lid_inference_concurrency: int = Field(0, ge=0, description="LID 样本级推理并发数")
    enable_mos: bool = Field(False, description="是否启用 MOS 语音质量评估")
    mos_target: str = Field("", description="MOS gRPC 服务地址")
    enable_snr: bool = Field(False, description="是否启用 SNR 语音质量评估")
    snr_target: str = Field("", description="SNR gRPC 服务地址")
    sqa_inference_concurrency: int = Field(
        0,
        ge=0,
        description="MOS/SNR 推理并发数，0 表示串行",
    )
    lid_confidence_threshold: float = Field(
        0.0,
        ge=0,
        le=1,
        description="LID 置信度阈值兼容字段；当前严格按模型输出标签判定",
    )
    remove_punctuation: bool = Field(False, description="ASR 评估时是否移除标点")
    mask_frame_seconds: float = Field(0.01, gt=0, description="VAD mask 帧长秒数")
    chunk_duration_seconds: float = Field(0.1, gt=0, description="VAD 流式分块秒数")
    speech_padding_seconds: float = Field(
        0.0,
        ge=0,
        description="VAD 检出语音段前后扩展秒数",
    )
    hit_threshold: float = Field(0.9, ge=0, le=1, description="VAD 段命中阈值")
    streaming: bool = Field(False, description="是否使用 VAD 流式接口")

    @model_validator(mode="after")
    def validate_sqa_config(self) -> EvaluationRequest:
        self.mos_target = self.mos_target.strip()
        self.snr_target = self.snr_target.strip()
        if self.task == "denoise" and not self.enable_mos and not self.enable_snr:
            raise ValueError("SE 评估必须至少启用 MOS 或 SNR")
        if self.enable_mos and not self.mos_target:
            raise ValueError("启用 MOS 时 MOS 引擎地址不能为空")
        if self.enable_snr and not self.snr_target:
            raise ValueError("启用 SNR 时 SNR 引擎地址不能为空")
        return self


class EvaluationCreated(BaseModel):
    job_id: str
    status: JobStatus


class EngineConnectivityRequest(BaseModel):
    target: str = Field(..., description="gRPC 服务地址")
    timeout_seconds: float = Field(3.0, gt=0, description="连接超时秒数")


class EngineConnectivityResult(BaseModel):
    ok: bool
    target: str
    message: str


class HelpDocument(BaseModel):
    title: str
    markdown: str


class DirectoryEntry(BaseModel):
    name: str
    path: str
    kind: Literal["directory", "file"]


class DirectoryListing(BaseModel):
    currentPath: str
    parentPath: str | None
    entries: list[DirectoryEntry]


class DatasetUploadResult(BaseModel):
    dataset_path: str
    imported_count: int
    skipped_count: int = 0
    message: str | None = None


@dataclass
class SqaRuntimeEngine:
    name: str
    target: str
    inferencer: SqaGrpcInferencer | None = None
    error: str | None = None


class SqaAssessor:
    def __init__(self, request: EvaluationRequest) -> None:
        self.concurrency = request.sqa_inference_concurrency
        engine_configs = _sqa_engine_configs(request)
        self.enabled = bool(engine_configs)
        self.semaphore = (
            threading.Semaphore(self.concurrency)
            if self.enabled and self.concurrency > 0
            else None
        )
        self.engines: list[SqaRuntimeEngine] = []
        if not self.enabled:
            return
        for name, target in engine_configs:
            try:
                inferencer = SqaGrpcInferencer(
                    target=target,
                    sample_rate=request.sample_rate,
                    request_timeout_seconds=request.request_timeout_seconds,
                    connect_timeout_seconds=request.connect_timeout_seconds,
                )
                self.engines.append(
                    SqaRuntimeEngine(name=name, target=target, inferencer=inferencer)
                )
            except Exception as exc:  # noqa: BLE001 - SQA 不应拖垮主评估
                logger.exception(
                    "SQA 引擎初始化失败，将记录为空分: name=%s target=%s",
                    name,
                    target,
                )
                self.engines.append(
                    SqaRuntimeEngine(name=name, target=target, error=str(exc))
                )

    def assess(self, audio_array: np.ndarray) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        if self.concurrency <= 0 or len(self.engines) <= 1:
            return [self._assess_engine(engine, audio_array) for engine in self.engines]

        max_workers = min(self.concurrency, len(self.engines))
        results: list[tuple[int, dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._assess_engine, engine, audio_array): index
                for index, engine in enumerate(self.engines)
            }
            for future in as_completed(futures):
                results.append((futures[future], future.result()))
        return [result for _, result in sorted(results, key=lambda item: item[0])]

    def close(self) -> None:
        for engine in self.engines:
            if engine.inferencer is not None:
                engine.inferencer.close()

    def _assess_engine(
        self,
        engine: SqaRuntimeEngine,
        audio_array: np.ndarray,
    ) -> dict[str, Any]:
        payload = {
            "engine_name": engine.name,
            "target": engine.target,
            "score": None,
            "error": engine.error,
        }
        if engine.inferencer is None:
            return payload

        try:
            if self.semaphore is None:
                result = engine.inferencer.infer(audio_array)
            else:
                with self.semaphore:
                    result = engine.inferencer.infer(audio_array)
            payload["score"] = result.score
            payload["error"] = None
        except Exception as exc:  # noqa: BLE001 - 单个 SQA 分数失败不影响主评估
            logger.exception(
                "SQA 评估失败: name=%s target=%s",
                engine.name,
                engine.target,
            )
            payload["error"] = str(exc)
        return payload


def _sqa_engine_configs(request: EvaluationRequest) -> list[tuple[str, str]]:
    configs: list[tuple[str, str]] = []
    if request.enable_mos:
        configs.append(("MOS", request.mos_target))
    if request.enable_snr:
        configs.append(("SNR", request.snr_target))
    return configs


@dataclass
class EvaluationJob:
    job_id: str
    request: EvaluationRequest
    status: JobStatus = "queued"
    latest_progress: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    sample_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    asr_inference_rows: list[dict[str, Any]] = field(default_factory=list)
    denoise_report_samples: list[dict[str, Any]] = field(default_factory=list)
    keyword_report_samples: list[dict[str, Any]] = field(default_factory=list)
    lid_report_samples: list[dict[str, Any]] = field(default_factory=list)
    vad_metric_rows: list[dict[str, Any]] = field(default_factory=list)
    vad_report_samples: list[dict[str, Any]] = field(default_factory=list)
    temp_dir: Path | None = None
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

ALLOWED_UPLOAD_SUFFIXES = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".json",
    ".jsonl",
    ".csv",
    ".parquet",
    ".txt",
}


@app.get("/")
def index() -> dict[str, str]:
    return {"name": "Prama ASR Evaluation Service", "status": "ok"}


@app.get("/api/help", response_model=HelpDocument)
def get_help_document() -> HelpDocument:
    help_path = Path(__file__).resolve().parents[1] / "help" / "datasets.md"
    if not help_path.exists():
        raise HTTPException(status_code=404, detail="帮助文档不存在")
    return HelpDocument(
        title="数据集与评估指标说明",
        markdown=help_path.read_text(encoding="utf-8"),
    )


@app.get("/api/files/directories", response_model=DirectoryListing)
def list_directory(
    path: str | None = Query(None, description="要浏览的工作目录内路径"),
) -> DirectoryListing:
    root = _workspace_root()
    current_path = _resolve_under_root(path, root=root)
    if not current_path.exists() or not current_path.is_dir():
        raise HTTPException(status_code=404, detail=f"目录不存在: {path or root}")

    entries = [
        DirectoryEntry(
            name=item.name,
            path=_display_path(item),
            kind="directory" if item.is_dir() else "file",
        )
        for item in sorted(
            current_path.iterdir(),
            key=lambda candidate: (not candidate.is_dir(), candidate.name.lower()),
        )
        if not item.name.startswith(".")
    ]
    parent_path = None
    if current_path != root:
        parent_path = _display_path(current_path.parent)
    return DirectoryListing(
        currentPath=_display_path(current_path),
        parentPath=parent_path,
        entries=entries,
    )


@app.post("/api/datasets/upload", response_model=DatasetUploadResult)
async def upload_dataset(
    files: list[UploadFile] = File(..., description="要上传的数据集文件"),
) -> DatasetUploadResult:
    if not files:
        raise HTTPException(status_code=400, detail="没有收到上传文件")

    upload_root = _upload_root()
    upload_root.mkdir(parents=True, exist_ok=True)
    prepared_files: list[tuple[UploadFile, Path]] = []
    skipped_count = 0

    for upload_file in files:
        raw_name = upload_file.filename or ""
        relative_path = _safe_relative_upload_path(raw_name)
        if relative_path is None:
            skipped_count += 1
            continue
        prepared_files.append((upload_file, relative_path))

    if not prepared_files:
        for upload_file in files:
            await upload_file.close()
        raise HTTPException(status_code=400, detail="没有可导入的支持文件")

    upload_plan = _build_upload_plan([path for _, path in prepared_files])
    upload_dir = _allocate_upload_directory(
        upload_root,
        preferred_name=upload_plan.root_directory_name,
    )
    imported_count = 0

    try:
        for upload_file, relative_path in prepared_files:
            target_relative_path = upload_plan.to_target_relative_path(relative_path)
            target_path = upload_dir / target_relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("wb") as output:
                shutil.copyfileobj(upload_file.file, output)
            imported_count += 1
    finally:
        for upload_file in files:
            await upload_file.close()

    logger.info(
        "数据集文件已上传: path=%s imported=%s skipped=%s",
        upload_dir,
        imported_count,
        skipped_count,
    )
    return DatasetUploadResult(
        dataset_path=_display_path(upload_dir),
        imported_count=imported_count,
        skipped_count=skipped_count,
        message=f"已上传 {imported_count} 个文件",
    )


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


@app.post("/api/engines/connectivity", response_model=EngineConnectivityResult)
def test_engine_connectivity(
    request: EngineConnectivityRequest,
) -> EngineConnectivityResult:
    target = request.target.strip()
    if not target:
        raise HTTPException(status_code=400, detail="引擎地址不能为空")
    channel = create_insecure_channel(target)
    try:
        grpc.channel_ready_future(channel).result(timeout=request.timeout_seconds)
    except Exception as exc:  # noqa: BLE001 - 返回连通性错误详情给前端
        return EngineConnectivityResult(
            ok=False,
            target=target,
            message=str(exc) or "连接失败",
        )
    finally:
        channel.close()
    return EngineConnectivityResult(ok=True, target=target, message="连接成功")


@app.get("/api/evaluations/{job_id}")
def get_evaluation(job_id: str) -> dict[str, Any]:
    return _get_job(job_id).snapshot()


@app.get("/api/evaluations/{job_id}/samples/{sample_id}/audio")
def get_sample_audio(job_id: str, sample_id: str) -> FileResponse:
    job = _get_job(job_id)
    with job.lock:
        record = job.sample_records.get(sample_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"样本不存在: {sample_id}")

    audio_path = Path(str(record.get("audio_path") or ""))
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=404, detail=f"音频文件不存在: {sample_id}")

    return FileResponse(
        audio_path,
        media_type=_audio_media_type(audio_path),
        filename=audio_path.name,
    )


@app.get("/api/evaluations/{job_id}/samples/{sample_id}/denoised-audio")
def get_sample_denoised_audio(job_id: str, sample_id: str) -> FileResponse:
    job = _get_job(job_id)
    with job.lock:
        record = job.sample_records.get(sample_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"样本不存在: {sample_id}")

    audio_path = Path(str(record.get("denoised_audio_path") or ""))
    if not audio_path.exists() or not audio_path.is_file():
        raise HTTPException(status_code=404, detail=f"SE 音频不存在: {sample_id}")

    return FileResponse(
        audio_path,
        media_type=_audio_media_type(audio_path),
        filename=audio_path.name,
    )


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


def _file_root() -> Path:
    return _workspace_root()


def _upload_root() -> Path:
    configured_upload_root = os.environ.get("PRAMA_UPLOAD_ROOT")
    if configured_upload_root:
        return Path(configured_upload_root).resolve()
    return _workspace_root()


def _workspace_root() -> Path:
    return Path(
        os.environ.get("PRAMA_WORKDIR")
        or os.environ.get("PRAMA_FILE_ROOT")
        or "data-bin"
    ).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_under_root(path: str | None, *, root: Path) -> Path:
    root = root.resolve()
    target = root if not path else Path(path).expanduser().resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise HTTPException(status_code=403, detail="路径不在允许访问的目录内") from error
    return target


def _safe_relative_upload_path(raw_name: str) -> Path | None:
    cleaned = raw_name.replace("\\", "/").lstrip("/")
    if not cleaned:
        return None
    relative_path = Path(cleaned)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        return None
    if relative_path.suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
        return None
    safe_parts = [
        "".join(
            char if char.isalnum() or char in ("-", "_", ".", " ") else "_"
            for char in part
        ).strip()
        for part in relative_path.parts
    ]
    safe_parts = [part for part in safe_parts if part]
    if not safe_parts:
        return None
    return Path(*safe_parts)


@dataclass(frozen=True)
class UploadPlan:
    root_directory_name: str
    strip_top_level_directory: bool = False

    def to_target_relative_path(self, relative_path: Path) -> Path:
        if self.strip_top_level_directory:
            return Path(*relative_path.parts[1:])
        return relative_path


def _build_upload_plan(relative_paths: list[Path]) -> UploadPlan:
    top_level_parts = {path.parts[0] for path in relative_paths if path.parts}
    should_preserve_directory = (
        len(top_level_parts) == 1
        and any(len(path.parts) > 1 for path in relative_paths)
    )
    if should_preserve_directory:
        return UploadPlan(
            root_directory_name=next(iter(top_level_parts)),
            strip_top_level_directory=True,
        )
    return UploadPlan(root_directory_name="upload")


def _allocate_upload_directory(upload_root: Path, *, preferred_name: str) -> Path:
    safe_name = _safe_directory_name(preferred_name) or "upload"
    candidate = upload_root / safe_name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate

    suffix = uuid4().hex[:8]
    candidate = upload_root / f"{safe_name}-{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _safe_directory_name(value: str) -> str:
    sanitized = "".join(
        char if char.isalnum() or char in ("-", "_", ".", " ") else "_"
        for char in value
    ).strip()
    return sanitized


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
        if request.task == "lid":
            _run_lid_evaluation(job)
            return
        if request.task == "denoise":
            _run_denoise_evaluation(job)
            return
        if request.task == "keyword":
            _run_keyword_evaluation(job)
            return

        sqa_assessor = SqaAssessor(request)
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
        inference_start_time = time.perf_counter()
        _register_asr_sample_records(job, dataset, sqa_assessor=sqa_assessor)
        total = len(dataset)
        evaluated = 0
        progress_lock = threading.Lock()
        inference_rows: list[dict[str, Any]] = []

        def publish_partial_infer_result(
            result: EvaluationPartialInferenceResult,
        ) -> None:
            if result.is_final:
                return
            with progress_lock:
                current_evaluated = evaluated
            payload = {
                "status": "running",
                "tag": result.tag,
                "total": total,
                "processed": current_evaluated,
                "evaluated": current_evaluated,
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
            with progress_lock:
                evaluated += 1
                current_evaluated = evaluated
            sample_record = _get_sample_record_for_result(
                job,
                sample_id=result.id,
                index=current_evaluated,
            )
            row = {
                "id": result.id,
                "index": current_evaluated,
                "reference": result.reference,
                "hypothesis": result.hypothesis,
                "audio_url": sample_record.get("audio_url"),
                "duration_seconds": sample_record.get("duration_seconds"),
                "sqa_scores": sample_record.get("sqa_scores", []),
            }
            with progress_lock:
                inference_rows.append(row)
            payload = {
                "status": "running",
                "tag": result.tag,
                "total": total,
                "processed": current_evaluated,
                "evaluated": current_evaluated,
                "id": result.id,
                "current_id": result.id,
                "reference": result.reference,
                "hypothesis": result.hypothesis,
                "is_final": True,
                "audio_url": sample_record.get("audio_url"),
                "duration_seconds": sample_record.get("duration_seconds"),
                "sqa_scores": sample_record.get("sqa_scores", []),
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
            inference_concurrency=_task_inference_concurrency(request, "asr"),
        ) as evaluator:
            metrics = evaluator.iter_evaluate(
                inferencer,
                on_infer_result=publish_infer_result,
                on_partial_infer_result=publish_partial_infer_result,
            )
        processing_elapsed_seconds = time.perf_counter() - inference_start_time

        wer_report = _build_wer_report(inference_rows)
        cer_report = _build_cer_report(inference_rows)
        result = {
            **metrics,
            "wer_report": wer_report,
            "cer_report": cer_report,
            **_performance_payload(
                audio_duration_seconds=_sum_row_duration(inference_rows),
                processing_elapsed_seconds=processing_elapsed_seconds,
            ),
            **_sample_count_payload(
                included_count=len(inference_rows),
                total_count=len(inference_rows),
            ),
            **_sqa_summary_payload(inference_rows),
        }
        with job.lock:
            job.status = "completed"
            job.asr_inference_rows = inference_rows
            job.result = result
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
        if request.task == "asr" and "sqa_assessor" in locals():
            sqa_assessor.close()
        message_manager.close_job(job.job_id)


def _run_keyword_evaluation(job: EvaluationJob) -> None:
    request = job.request
    sqa_assessor = SqaAssessor(request)
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
    dataset = _load_keyword_dataset(
        Path(request.dataset_path),
        split=request.split,
        limit=request.limit,
        sample_rate=request.sample_rate,
    )
    total = len(dataset)
    report_samples: list[dict[str, Any]] = []
    evaluation_start_time = time.perf_counter()

    def evaluate_sample(index: int, sample: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(sample.get("id") or sample.get("utt_id") or index)
        keyword = str(sample["keyword"]).strip()
        expected_hit = _as_bool(sample["expected_hit"])
        audio_array, audio_sample_rate = _decode_sample_audio(sample["audio"])
        if audio_sample_rate != request.sample_rate:
            raise ValueError(
                f"样本采样率与关键词评估配置不一致: id={sample_id} "
                f"audio_sample_rate={audio_sample_rate} sample_rate={request.sample_rate}"
            )
        audio_array = _prepare_audio_for_export(audio_array)
        sample_record = _register_sample_record(
            job=job,
            sample=sample,
            sample_id=sample_id,
            index=index,
            audio_array=audio_array,
            sample_rate=audio_sample_rate,
        )
        transcript = _infer_final_transcript(inferencer, audio_array)
        match_text = _normalize_keyword_text(transcript)
        predicted_hit = _keyword_matches(transcript, keyword)
        correct = predicted_hit == expected_hit
        report_sample = {
            "id": sample_id,
            "index": index,
            "audio_url": sample_record.get("audio_url"),
            "duration_seconds": sample_record.get("duration_seconds"),
            "keyword": keyword,
            "expected_hit": expected_hit,
            "predicted_hit": predicted_hit,
            "correct": correct,
            "transcript": transcript,
            "match_text": match_text,
            "sqa_scores": sqa_assessor.assess(audio_array),
        }
        return {
            "report_sample": report_sample,
            "payload": {
                "status": "running",
                "tag": job.job_id,
                "total": total,
                "id": sample_id,
                "current_id": sample_id,
                "reference": f"{keyword}: {'hit' if expected_hit else 'no hit'}",
                "hypothesis": "hit" if predicted_hit else "no hit",
                "is_final": True,
                "result": report_sample,
                "audio_url": sample_record.get("audio_url"),
                "duration_seconds": sample_record.get("duration_seconds"),
                "sqa_scores": report_sample.get("sqa_scores", []),
            },
        }

    def publish_sample_result(item: dict[str, Any], processed: int) -> None:
        report_samples.append(item["report_sample"])
        payload = {
            **item["payload"],
            "processed": processed,
            "evaluated": processed,
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

    try:
        samples = list(enumerate(dataset, start=1))
        inference_concurrency = _task_inference_concurrency(request, "keyword")
        if inference_concurrency > 0 and samples:
            max_workers = min(inference_concurrency, len(samples))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(evaluate_sample, index, sample)
                    for index, sample in samples
                ]
                for processed, future in enumerate(as_completed(futures), start=1):
                    publish_sample_result(future.result(), processed)
        else:
            for index, sample in samples:
                publish_sample_result(evaluate_sample(index, sample), index)

        processing_elapsed_seconds = time.perf_counter() - evaluation_start_time
        report_samples.sort(key=lambda sample: int(sample.get("index") or 0))
        with job.lock:
            job.status = "completed"
            job.keyword_report_samples = report_samples
            job.result = {
                **_build_keyword_report(report_samples),
                **_performance_payload(
                    audio_duration_seconds=_sum_sample_duration(report_samples),
                    processing_elapsed_seconds=processing_elapsed_seconds,
                ),
                **_sample_count_payload(
                    included_count=len(report_samples),
                    total_count=len(report_samples),
                ),
                **_sqa_summary_payload(report_samples),
            }
        logger.info("关键词评估任务完成: job_id=%s", job.job_id)
    finally:
        sqa_assessor.close()
        inferencer.close()


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


def _load_keyword_dataset(
    dataset_path: Path,
    *,
    split: str,
    limit: int | None,
    sample_rate: int,
) -> Dataset:
    logger.info("加载关键词数据集: path=%s split=%s", dataset_path, split)
    if _is_audiofolder_dataset_dir(dataset_path):
        dataset = load_dataset("audiofolder", data_dir=str(dataset_path), split=split)
    else:
        dataset = load_dataset(str(dataset_path), split=split)

    missing = sorted({"audio", "keyword", "expected_hit"} - set(dataset.column_names))
    if missing:
        raise ValueError(f"关键词数据集缺少必要字段: {', '.join(missing)}")

    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    normalized_keywords = {
        _normalize_keyword_text(str(sample["keyword"]))
        for sample in dataset
        if str(sample["keyword"]).strip()
    }
    if not normalized_keywords:
        raise ValueError("关键词数据集必须提供非空 keyword")
    for sample in dataset:
        _as_bool(sample["expected_hit"])

    logger.info(
        "关键词数据集已加载: size=%s keyword_count=%s",
        len(dataset),
        len(normalized_keywords),
    )
    return dataset


def _infer_final_transcript(
    inferencer: AsrGrpcInferencer,
    audio_array: np.ndarray,
) -> str:
    latest_transcript = ""
    final_transcript = ""
    for transcript, is_final in inferencer.infer(audio_array):
        latest_transcript = transcript
        if is_final:
            final_transcript = transcript
    return final_transcript or latest_transcript


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "hit"}:
            return True
        if normalized in {"false", "0", "no", "n", "miss", "none"}:
            return False
    raise ValueError(f"expected_hit 必须是布尔值: {value!r}")


def _normalize_keyword_text(text: str) -> str:
    normalized = "".join(
        " " if unicodedata.category(char).startswith("P") else char.lower()
        for char in text
    )
    return " ".join(normalized.split())


def _keyword_matches(transcript: str, keyword: str) -> bool:
    normalized_transcript = _normalize_keyword_text(transcript)
    normalized_keyword = _normalize_keyword_text(keyword)
    if not normalized_keyword:
        return False
    if _is_word_keyword(normalized_keyword):
        transcript_tokens = normalized_transcript.split()
        keyword_tokens = normalized_keyword.split()
        if not keyword_tokens or len(keyword_tokens) > len(transcript_tokens):
            return False
        return any(
            transcript_tokens[index:index + len(keyword_tokens)] == keyword_tokens
            for index in range(len(transcript_tokens) - len(keyword_tokens) + 1)
        )
    return normalized_keyword in normalized_transcript


def _is_word_keyword(normalized_keyword: str) -> bool:
    return re.fullmatch(r"[a-z0-9]+(?: [a-z0-9]+)*", normalized_keyword) is not None


def remove_punctuation(text: str) -> str:
    without_punctuation = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )
    return " ".join(without_punctuation.split())


def _register_asr_sample_records(
    job: EvaluationJob,
    dataset: Dataset,
    *,
    sqa_assessor: SqaAssessor,
) -> None:
    request = job.request
    for index, sample in enumerate(dataset, start=1):
        sample_id = str(sample.get("id") or sample.get("utt_id") or index)
        audio_array, audio_sample_rate = _decode_sample_audio(sample["audio"])
        if audio_sample_rate != request.sample_rate:
            raise ValueError(
                f"样本采样率与 ASR 配置不一致: id={sample_id} "
                f"audio_sample_rate={audio_sample_rate} sample_rate={request.sample_rate}"
            )
        sample_record = _register_sample_record(
            job=job,
            sample=sample,
            sample_id=sample_id,
            index=index,
            audio_array=audio_array,
            sample_rate=audio_sample_rate,
        )
        sample_record["sqa_scores"] = sqa_assessor.assess(audio_array)


def _register_sample_record(
    *,
    job: EvaluationJob,
    sample: dict[str, Any],
    sample_id: str,
    index: int,
    audio_array: np.ndarray,
    sample_rate: int,
) -> dict[str, Any]:
    audio_array = _prepare_audio_for_export(audio_array)
    audio_path = _find_sample_audio_path(sample)
    if audio_path is None:
        audio_path = _write_job_audio_file(
            job=job,
            sample_id=sample_id,
            audio_array=audio_array,
            sample_rate=sample_rate,
        )
    duration_seconds = len(audio_array) / sample_rate if sample_rate > 0 else 0.0
    record = {
        "id": sample_id,
        "index": index,
        "audio_path": str(audio_path),
        "audio_url": _sample_audio_url(job.job_id, sample_id),
        "duration_seconds": duration_seconds,
    }
    with job.lock:
        job.sample_records[sample_id] = record
    return record


def _get_sample_record_for_result(
    job: EvaluationJob,
    *,
    sample_id: str,
    index: int,
) -> dict[str, Any]:
    record = job.sample_records.get(sample_id)
    if record is not None:
        return record
    return next(
        (
            candidate
            for candidate in job.sample_records.values()
            if candidate.get("index") == index
        ),
        {},
    )


def _decode_sample_audio(audio: Any) -> tuple[np.ndarray, int]:
    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        data = samples.data
        audio_array = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
        return np.asarray(audio_array), int(samples.sample_rate)
    if isinstance(audio, dict):
        return np.asarray(audio["array"]), int(audio["sampling_rate"])
    return np.asarray(audio), 0


def _prepare_audio_for_export(audio_array: np.ndarray) -> np.ndarray:
    array = np.asarray(audio_array)
    array = np.squeeze(array)
    if array.ndim == 2:
        channel_axis = 0 if array.shape[0] <= array.shape[1] else 1
        array = array.mean(axis=channel_axis)
    if array.ndim != 1:
        raise ValueError(f"不支持的音频维度: {array.shape}")
    return array.astype(np.float32, copy=False)


def _find_sample_audio_path(sample: dict[str, Any]) -> Path | None:
    audio = sample.get("audio")
    if isinstance(audio, dict):
        audio_path = audio.get("path")
        if isinstance(audio_path, str) and audio_path:
            path = Path(audio_path)
            if path.exists() and path.is_file():
                return path

    for key in ("path", "file_name"):
        value = sample.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            if path.exists() and path.is_file():
                return path
    return None


def _write_job_audio_file(
    *,
    job: EvaluationJob,
    sample_id: str,
    audio_array: np.ndarray,
    sample_rate: int,
    name_suffix: str = "",
) -> Path:
    with job.lock:
        if job.temp_dir is None:
            job.temp_dir = Path(tempfile.mkdtemp(prefix=f"prama-{job.job_id}-"))
        temp_dir = job.temp_dir
    safe_name = "".join(
        char if char.isalnum() or char in ("-", "_", ".") else "_"
        for char in sample_id
    )
    suffix = f"-{name_suffix}" if name_suffix else ""
    audio_path = temp_dir / f"{safe_name or 'sample'}{suffix}.wav"
    sf.write(audio_path, audio_array, sample_rate)
    return audio_path


def _sample_audio_url(job_id: str, sample_id: str) -> str:
    return f"/api/evaluations/{job_id}/samples/{quote(sample_id, safe='')}/audio"


def _sample_denoised_audio_url(job_id: str, sample_id: str) -> str:
    return (
        f"/api/evaluations/{job_id}/samples/"
        f"{quote(sample_id, safe='')}/denoised-audio"
    )


def _audio_media_type(audio_path: Path) -> str:
    suffix = audio_path.suffix.lower()
    if suffix == ".mp3":
        return "audio/mpeg"
    if suffix == ".flac":
        return "audio/flac"
    if suffix == ".ogg":
        return "audio/ogg"
    return "audio/wav"


def _run_vad_evaluation(job: EvaluationJob) -> None:
    request = job.request
    sqa_assessor = SqaAssessor(request)
    inferencer = VadGrpcInferencer(
        target=request.target,
        sample_rate=request.sample_rate,
        mask_frame_seconds=request.mask_frame_seconds,
        chunk_duration_seconds=request.chunk_duration_seconds,
        speech_padding_seconds=request.speech_padding_seconds,
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
    evaluation_start_time = time.perf_counter()

    def evaluate_sample(index: int, sample: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(sample.get("id") or sample.get("utt_id") or index)
        audio = sample["audio"]
        audio_array = _prepare_vad_audio(audio, sample_rate=request.sample_rate)
        sample_record = _register_sample_record(
            job=job,
            sample=sample,
            sample_id=sample_id,
            index=index,
            audio_array=audio_array,
            sample_rate=request.sample_rate,
        )
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
        report_sample = _build_vad_sample_report(
            sample_id=sample_id,
            index=index,
            reference_mask=reference_mask,
            prediction_mask=prediction_mask,
            frame_seconds=request.mask_frame_seconds,
            hit_threshold=request.hit_threshold,
            metrics=result_dict,
        )
        report_sample.update(
            {
                "audio_url": sample_record.get("audio_url"),
                "duration_seconds": sample_record.get(
                    "duration_seconds",
                    report_sample["duration_seconds"],
                ),
                "sqa_scores": sqa_assessor.assess(audio_array),
            }
        )
        return {
            "row": {"id": sample_id, "index": index, **result_dict},
            "report_sample": report_sample,
            "payload": {
                "status": "running",
                "tag": job.job_id,
                "total": total,
                "id": sample_id,
                "current_id": sample_id,
                "reference": _format_vad_segments(
                    reference_mask,
                    request.mask_frame_seconds,
                ),
                "hypothesis": _format_vad_segments(
                    prediction_mask,
                    request.mask_frame_seconds,
                ),
                "is_final": True,
                "result": result_dict,
                "audio_url": sample_record.get("audio_url"),
                "duration_seconds": sample_record.get("duration_seconds"),
                "sqa_scores": report_sample.get("sqa_scores", []),
            },
        }

    def publish_sample_result(item: dict[str, Any], processed: int) -> None:
        rows.append(item["row"])
        report_samples.append(item["report_sample"])
        payload = {
            **item["payload"],
            "processed": processed,
            "evaluated": processed,
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

    try:
        samples = list(enumerate(dataset, start=1))
        inference_concurrency = _task_inference_concurrency(request, "vad")
        if inference_concurrency > 0 and samples:
            max_workers = min(inference_concurrency, len(samples))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(evaluate_sample, index, sample)
                    for index, sample in samples
                ]
                for processed, future in enumerate(as_completed(futures), start=1):
                    publish_sample_result(future.result(), processed)
        else:
            for index, sample in samples:
                publish_sample_result(evaluate_sample(index, sample), index)

        processing_elapsed_seconds = time.perf_counter() - evaluation_start_time
        rows.sort(key=lambda row: int(row.get("index") or 0))
        report_samples.sort(key=lambda sample: int(sample.get("index") or 0))
        with job.lock:
            job.status = "completed"
            job.vad_metric_rows = rows
            job.vad_report_samples = report_samples
            job.result = {
                **_build_vad_report(rows, samples=report_samples),
                **_performance_payload(
                    audio_duration_seconds=_sum_sample_duration(report_samples),
                    processing_elapsed_seconds=processing_elapsed_seconds,
                ),
                **_sample_count_payload(
                    included_count=len(rows),
                    total_count=len(rows),
                ),
                **_sqa_summary_payload(report_samples),
            }
        logger.info("VAD 评估任务完成: job_id=%s", job.job_id)
    finally:
        sqa_assessor.close()
        inferencer.close()


def _run_lid_evaluation(job: EvaluationJob) -> None:
    request = job.request
    sqa_assessor = SqaAssessor(request)
    inferencer = LidGrpcInferencer(
        target=request.target,
        sample_rate=request.sample_rate,
        request_timeout_seconds=request.request_timeout_seconds,
        connect_timeout_seconds=request.connect_timeout_seconds,
    )
    dataset = _load_lid_dataset(
        Path(request.dataset_path),
        split=request.split,
        limit=request.limit,
        sample_rate=request.sample_rate,
    )
    total = len(dataset)
    report_samples: list[dict[str, Any]] = []
    evaluation_start_time = time.perf_counter()

    def evaluate_sample(index: int, sample: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(sample.get("id") or sample.get("utt_id") or index)
        reference_language = str(sample["language_id"])
        audio_array = _prepare_vad_audio(sample["audio"], sample_rate=request.sample_rate)
        sample_record = _register_sample_record(
            job=job,
            sample=sample,
            sample_id=sample_id,
            index=index,
            audio_array=audio_array,
            sample_rate=request.sample_rate,
        )
        prediction = inferencer.infer(audio_array)
        predicted_language = _lid_predicted_language(
            prediction.lang,
            prediction.score,
            request.lid_confidence_threshold,
        )
        correct = predicted_language == reference_language
        report_sample = {
            "id": sample_id,
            "index": index,
            "audio_url": sample_record.get("audio_url"),
            "duration_seconds": sample_record.get("duration_seconds"),
            "reference_language": reference_language,
            "predicted_language": predicted_language,
            "raw_language": prediction.lang,
            "confidence": prediction.score,
            "correct": correct,
            "sqa_scores": sqa_assessor.assess(audio_array),
        }
        return {
            "report_sample": report_sample,
            "payload": {
                "status": "running",
                "tag": job.job_id,
                "total": total,
                "id": sample_id,
                "current_id": sample_id,
                "reference": reference_language,
                "hypothesis": f"{predicted_language} ({prediction.score:.4f})",
                "is_final": True,
                "result": report_sample,
                "audio_url": sample_record.get("audio_url"),
                "duration_seconds": sample_record.get("duration_seconds"),
                "sqa_scores": report_sample.get("sqa_scores", []),
            },
        }

    def publish_sample_result(item: dict[str, Any], processed: int) -> None:
        report_samples.append(item["report_sample"])
        payload = {
            **item["payload"],
            "processed": processed,
            "evaluated": processed,
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

    try:
        samples = list(enumerate(dataset, start=1))
        inference_concurrency = _task_inference_concurrency(request, "lid")
        if inference_concurrency > 0 and samples:
            max_workers = min(inference_concurrency, len(samples))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(evaluate_sample, index, sample)
                    for index, sample in samples
                ]
                for processed, future in enumerate(as_completed(futures), start=1):
                    publish_sample_result(future.result(), processed)
        else:
            for index, sample in samples:
                publish_sample_result(evaluate_sample(index, sample), index)

        processing_elapsed_seconds = time.perf_counter() - evaluation_start_time
        report_samples.sort(key=lambda sample: int(sample.get("index") or 0))
        with job.lock:
            job.status = "completed"
            job.lid_report_samples = report_samples
            job.result = {
                **_build_lid_report(report_samples),
                **_performance_payload(
                    audio_duration_seconds=_sum_sample_duration(report_samples),
                    processing_elapsed_seconds=processing_elapsed_seconds,
                ),
                **_sample_count_payload(
                    included_count=len(report_samples),
                    total_count=len(report_samples),
                ),
                **_sqa_summary_payload(report_samples),
            }
        logger.info("LID 评估任务完成: job_id=%s", job.job_id)
    finally:
        sqa_assessor.close()
        inferencer.close()


def _run_denoise_evaluation(job: EvaluationJob) -> None:
    request = job.request
    sqa_assessor = SqaAssessor(request)
    inferencer = DenoiseGrpcInferencer(
        target=request.target,
        sample_rate=request.sample_rate,
        request_timeout_seconds=request.request_timeout_seconds,
        connect_timeout_seconds=request.connect_timeout_seconds,
    )
    dataset = _load_denoise_dataset(
        Path(request.dataset_path),
        split=request.split,
        limit=request.limit,
        sample_rate=request.sample_rate,
    )
    total = len(dataset)
    report_samples: list[dict[str, Any]] = []
    evaluation_start_time = time.perf_counter()

    def evaluate_sample(index: int, sample: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(sample.get("id") or sample.get("utt_id") or index)
        audio_array = _prepare_vad_audio(sample["audio"], sample_rate=request.sample_rate)
        sample_record = _register_sample_record(
            job=job,
            sample=sample,
            sample_id=sample_id,
            index=index,
            audio_array=audio_array,
            sample_rate=request.sample_rate,
        )
        original_scores = sqa_assessor.assess(audio_array)
        denoised_scores: list[dict[str, Any]] = []
        denoised_audio_url = None
        error = None

        try:
            denoised_audio = inferencer.infer(audio_array)
            denoised_audio = _prepare_audio_for_export(denoised_audio)
            denoised_audio_path = _write_job_audio_file(
                job=job,
                sample_id=sample_id,
                audio_array=denoised_audio,
                sample_rate=request.sample_rate,
                name_suffix="denoised",
            )
            denoised_audio_url = _sample_denoised_audio_url(job.job_id, sample_id)
            with job.lock:
                sample_record["denoised_audio_path"] = str(denoised_audio_path)
                sample_record["denoised_audio_url"] = denoised_audio_url
            denoised_scores = sqa_assessor.assess(denoised_audio)
        except Exception as exc:  # noqa: BLE001 - 单样本失败要进入报告
            logger.exception("SE 样本处理失败: job_id=%s id=%s", job.job_id, sample_id)
            error = str(exc)

        sample_payload = _build_denoise_sample_payload(
            sample_id=sample_id,
            index=index,
            audio_url=sample_record.get("audio_url"),
            denoised_audio_url=denoised_audio_url,
            duration_seconds=sample_record.get("duration_seconds"),
            original_scores=original_scores,
            denoised_scores=denoised_scores,
            error=error,
        )
        return {
            "report_sample": sample_payload,
            "payload": {
                "status": "running",
                "tag": job.job_id,
                "total": total,
                "id": sample_id,
                "current_id": sample_id,
                "reference": "原始音频",
                "hypothesis": "SE 完成" if error is None else "SE 失败",
                "is_final": True,
                "result": sample_payload,
                "audio_url": sample_record.get("audio_url"),
                "duration_seconds": sample_record.get("duration_seconds"),
            },
        }

    def publish_sample_result(item: dict[str, Any], processed: int) -> None:
        report_samples.append(item["report_sample"])
        payload = {
            **item["payload"],
            "processed": processed,
            "evaluated": processed,
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

    try:
        samples = list(enumerate(dataset, start=1))
        inference_concurrency = _task_inference_concurrency(request, "denoise")
        if inference_concurrency > 0 and samples:
            max_workers = min(inference_concurrency, len(samples))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(evaluate_sample, index, sample)
                    for index, sample in samples
                ]
                for processed, future in enumerate(as_completed(futures), start=1):
                    publish_sample_result(future.result(), processed)
        else:
            for index, sample in samples:
                publish_sample_result(evaluate_sample(index, sample), index)

        processing_elapsed_seconds = time.perf_counter() - evaluation_start_time
        report_samples.sort(key=lambda sample: int(sample.get("index") or 0))
        with job.lock:
            job.status = "completed"
            job.denoise_report_samples = report_samples
            job.result = {
                **_build_denoise_report(report_samples),
                **_performance_payload(
                    audio_duration_seconds=_sum_sample_duration(report_samples),
                    processing_elapsed_seconds=processing_elapsed_seconds,
                ),
                **_sample_count_payload(
                    included_count=len(report_samples),
                    total_count=len(report_samples),
                ),
            }
        logger.info("SE 评估任务完成: job_id=%s", job.job_id)
    finally:
        sqa_assessor.close()
        inferencer.close()


def _load_denoise_dataset(
    dataset_path: Path,
    *,
    split: str,
    limit: int | None,
    sample_rate: int,
) -> Dataset:
    logger.info("加载 SE 数据集: path=%s split=%s", dataset_path, split)
    if _is_audiofolder_dataset_dir(dataset_path):
        dataset = load_dataset("audiofolder", data_dir=str(dataset_path), split=split)
    else:
        dataset = load_dataset(str(dataset_path), split=split)

    if "audio" not in dataset.column_names:
        raise ValueError("SE 数据集缺少必要字段: audio")

    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    logger.info("SE 数据集已加载: size=%s", len(dataset))
    return dataset


def _load_lid_dataset(
    dataset_path: Path,
    *,
    split: str,
    limit: int | None,
    sample_rate: int,
) -> Dataset:
    logger.info("加载 LID 数据集: path=%s split=%s", dataset_path, split)
    if _is_audiofolder_dataset_dir(dataset_path):
        dataset = load_dataset("audiofolder", data_dir=str(dataset_path), split=split)
    else:
        dataset = load_dataset(str(dataset_path), split=split)

    missing = sorted({"audio", "language_id"} - set(dataset.column_names))
    if missing:
        raise ValueError(f"LID 数据集缺少必要字段: {', '.join(missing)}")

    dataset = dataset.cast_column("audio", Audio(sampling_rate=sample_rate))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    logger.info("LID 数据集已加载: size=%s", len(dataset))
    return dataset


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
            "sample_count": 0,
        }

    frame_average_keys = [
        "frame_accuracy",
        "frame_specificity",
        "frame_balanced_accuracy",
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
    reference_segment_weights = [
        float(row["reference_segment_count"]) + VAD_REPORT_WEIGHT_EPS
        for row in rows
    ]
    prediction_segment_weights = [
        float(row["prediction_segment_count"]) + VAD_REPORT_WEIGHT_EPS
        for row in rows
    ]
    frame_metrics.update(
        {
            "frame_recall": _weighted_vad_metric(
                rows,
                key="frame_recall",
                weights=reference_segment_weights,
            ),
            "frame_precision": _weighted_vad_metric(
                rows,
                key="frame_precision",
                weights=prediction_segment_weights,
            ),
            "frame_miss_rate": _weighted_vad_metric(
                rows,
                key="frame_miss_rate",
                weights=reference_segment_weights,
            ),
            "frame_false_alarm_rate": _weighted_vad_metric(
                rows,
                key="frame_false_alarm_rate",
                weights=prediction_segment_weights,
            ),
        }
    )
    frame_metrics["frame_f1"] = _f1_from_precision_recall(
        frame_metrics["frame_precision"],
        frame_metrics["frame_recall"],
    )
    frame_metrics.update({key: int(sum(row[key] for row in rows)) for key in frame_count_keys})
    segment_metrics = {
        "segment_recall": _weighted_vad_metric(
            rows,
            key="segment_recall",
            weights=reference_segment_weights,
        ),
        "segment_precision": _weighted_vad_metric(
            rows,
            key="segment_precision",
            weights=prediction_segment_weights,
        ),
        "segment_miss_rate": _weighted_vad_metric(
            rows,
            key="segment_miss_rate",
            weights=reference_segment_weights,
        ),
        "segment_false_alarm_rate": _weighted_vad_metric(
            rows,
            key="segment_false_alarm_rate",
            weights=prediction_segment_weights,
        ),
    }
    segment_metrics["segment_f1"] = _f1_from_precision_recall(
        segment_metrics["segment_precision"],
        segment_metrics["segment_recall"],
    )
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


def _weighted_vad_metric(
    rows: list[dict[str, Any]],
    *,
    key: str,
    weights: list[float],
) -> float:
    weight_sum = sum(weights)
    if weight_sum <= 0:
        return 0.0
    return float(
        sum(float(row[key]) * weight for row, weight in zip(rows, weights))
        / weight_sum
    )


def _f1_from_precision_recall(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _build_lid_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    sample_count = len(samples)
    language_totals: dict[str, int] = {}
    language_hits: dict[str, int] = {}
    predicted_totals: dict[str, int] = {}
    confusion_counts: dict[str, dict[str, int]] = {}
    known_correct_count = 0
    known_sample_count = 0
    overall_correct_count = 0
    unknown_false_accept_count = 0
    known_reject_count = 0

    for sample in samples:
        reference_language = str(sample.get("reference_language") or "")
        predicted_language = str(sample.get("predicted_language") or "")
        correct = predicted_language == reference_language
        known_reference = reference_language != LID_UNKNOWN_LANGUAGE

        language_totals[reference_language] = language_totals.get(reference_language, 0) + 1
        predicted_totals[predicted_language] = predicted_totals.get(predicted_language, 0) + 1
        confusion_counts.setdefault(reference_language, {})
        confusion_counts[reference_language][predicted_language] = (
            confusion_counts[reference_language].get(predicted_language, 0) + 1
        )
        if correct:
            language_hits[reference_language] = language_hits.get(reference_language, 0) + 1
            overall_correct_count += 1
        if known_reference:
            known_sample_count += 1
            if correct:
                known_correct_count += 1
            if predicted_language == LID_UNKNOWN_LANGUAGE:
                known_reject_count += 1
        elif predicted_language != LID_UNKNOWN_LANGUAGE:
            unknown_false_accept_count += 1

    reference_languages = _sort_lid_languages(language_totals)
    predicted_languages = _sort_lid_languages(
        {
            predicted_language: 1
            for predictions in confusion_counts.values()
            for predicted_language in predictions
        }
    )
    all_language_recalls = [
        {
            "language": language,
            "correct_count": language_hits.get(language, 0),
            "sample_count": language_totals[language],
            "predicted_count": predicted_totals.get(language, 0),
            "precision": _safe_divide_float(
                language_hits.get(language, 0),
                predicted_totals.get(language, 0),
            ),
            "recall": _safe_divide_float(
                language_hits.get(language, 0),
                language_totals[language],
            ),
        }
        for language in reference_languages
    ]
    known_language_recalls = [
        item
        for item in all_language_recalls
        if item["language"] != LID_UNKNOWN_LANGUAGE
    ]
    macro_recall = (
        sum(item["recall"] for item in known_language_recalls)
        / len(known_language_recalls)
        if known_language_recalls
        else 0.0
    )
    macro_precision = (
        sum(item["precision"] for item in known_language_recalls)
        / len(known_language_recalls)
        if known_language_recalls
        else 0.0
    )
    known_accuracy = _safe_divide_float(known_correct_count, known_sample_count)
    return {
        "accuracy": known_accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "known_accuracy": known_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "known_correct_count": known_correct_count,
        "known_sample_count": known_sample_count,
        "overall_correct_count": overall_correct_count,
        "unknown_false_accept_count": unknown_false_accept_count,
        "known_reject_count": known_reject_count,
        "lid_language_recalls": known_language_recalls,
        "lid_confusion_matrix": {
            "reference_languages": reference_languages,
            "predicted_languages": predicted_languages,
            "rows": [
                {
                    "reference_language": reference_language,
                    "total": language_totals[reference_language],
                    "counts": {
                        predicted_language: confusion_counts.get(
                            reference_language,
                            {},
                        ).get(predicted_language, 0)
                        for predicted_language in predicted_languages
                    },
                }
                for reference_language in reference_languages
            ],
        },
        "correct_count": overall_correct_count,
        "sample_count": sample_count,
        "lid_report": {"samples": samples},
    }


def _build_keyword_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    hit_count = sum(
        1
        for sample in samples
        if sample.get("expected_hit") is True and sample.get("predicted_hit") is True
    )
    miss_count = sum(
        1
        for sample in samples
        if sample.get("expected_hit") is True and sample.get("predicted_hit") is False
    )
    false_alarm_count = sum(
        1
        for sample in samples
        if sample.get("expected_hit") is False and sample.get("predicted_hit") is True
    )
    correct_reject_count = sum(
        1
        for sample in samples
        if sample.get("expected_hit") is False and sample.get("predicted_hit") is False
    )
    positive_sample_count = hit_count + miss_count
    negative_sample_count = false_alarm_count + correct_reject_count
    correct_count = hit_count + correct_reject_count
    precision = _safe_divide_float(hit_count, hit_count + false_alarm_count)
    recall = _safe_divide_float(hit_count, positive_sample_count)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    sample_count = len(samples)
    return {
        "accuracy": _safe_divide_float(correct_count, sample_count),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "false_alarm_count": false_alarm_count,
        "correct_reject_count": correct_reject_count,
        "positive_sample_count": positive_sample_count,
        "negative_sample_count": negative_sample_count,
        "correct_count": correct_count,
        "sample_count": sample_count,
        "keyword_report": {"samples": samples},
    }


def _sort_lid_languages(language_counts: dict[str, int]) -> list[str]:
    languages = sorted(language for language in language_counts if language)
    if LID_UNKNOWN_LANGUAGE in languages:
        languages = [
            language for language in languages if language != LID_UNKNOWN_LANGUAGE
        ] + [LID_UNKNOWN_LANGUAGE]
    return languages


def _build_denoise_sample_payload(
    *,
    sample_id: str,
    index: int,
    audio_url: str | None,
    denoised_audio_url: str | None,
    duration_seconds: Any,
    original_scores: list[dict[str, Any]],
    denoised_scores: list[dict[str, Any]],
    error: str | None,
) -> dict[str, Any]:
    original_snr = _score_for_engine(original_scores, "SNR")
    denoised_snr = _score_for_engine(denoised_scores, "SNR")
    original_mos = _score_for_engine(original_scores, "MOS")
    denoised_mos = _score_for_engine(denoised_scores, "MOS")
    return {
        "id": sample_id,
        "index": index,
        "audio_url": audio_url,
        "denoised_audio_url": denoised_audio_url,
        "duration_seconds": duration_seconds,
        "original_sqa_scores": original_scores,
        "denoised_sqa_scores": denoised_scores,
        "original_snr": original_snr,
        "denoised_snr": denoised_snr,
        "snr_delta": _score_delta(original_snr, denoised_snr),
        "original_mos": original_mos,
        "denoised_mos": denoised_mos,
        "mos_delta": _score_delta(original_mos, denoised_mos),
        "error": error,
    }


def _build_denoise_report(samples: list[dict[str, Any]]) -> dict[str, Any]:
    snr_deltas = _finite_values(sample.get("snr_delta") for sample in samples)
    mos_deltas = _finite_values(sample.get("mos_delta") for sample in samples)
    original_snrs = _finite_values(sample.get("original_snr") for sample in samples)
    denoised_snrs = _finite_values(sample.get("denoised_snr") for sample in samples)
    original_moss = _finite_values(sample.get("original_mos") for sample in samples)
    denoised_moss = _finite_values(sample.get("denoised_mos") for sample in samples)
    return {
        "mean_snr_delta": _mean_or_none(snr_deltas),
        "mean_mos_delta": _mean_or_none(mos_deltas),
        "mean_original_snr": _mean_or_none(original_snrs),
        "mean_denoised_snr": _mean_or_none(denoised_snrs),
        "mean_original_mos": _mean_or_none(original_moss),
        "mean_denoised_mos": _mean_or_none(denoised_moss),
        "sample_count": len(samples),
        "scored_snr_sample_count": len(snr_deltas),
        "scored_mos_sample_count": len(mos_deltas),
        "failed_sample_count": sum(1 for sample in samples if sample.get("error")),
        "denoise_report": {"samples": samples},
    }


def _build_asr_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"wer": 0.0, "cer": 0.0}
    frame = pd.DataFrame(
        [
            {
                "id": str(row["id"]),
                "reference": str(row["reference"]),
                "hypothesis": str(row["hypothesis"]),
            }
            for row in rows
        ]
    )
    return {
        "wer": get_wer_pd(frame),
        "cer": get_cer_pd(frame),
    }


def _sample_count_payload(
    *,
    included_count: int,
    total_count: int,
) -> dict[str, Any]:
    return {
        "included_sample_count": included_count,
        "total_sample_count": total_count,
    }


def _performance_payload(
    *,
    audio_duration_seconds: float,
    processing_elapsed_seconds: float,
) -> dict[str, float]:
    realtime_factor = _safe_divide_float(
        audio_duration_seconds,
        processing_elapsed_seconds,
    )
    return {
        "audio_duration_seconds": audio_duration_seconds,
        "processing_elapsed_seconds": processing_elapsed_seconds,
        "realtime_factor": realtime_factor,
    }


def _sum_row_duration(rows: list[dict[str, Any]]) -> float:
    return sum(
        _as_float(row.get("duration_seconds")) or 0.0
        for row in rows
    )


def _sum_sample_duration(samples: list[dict[str, Any]]) -> float:
    return sum(
        _as_float(sample.get("duration_seconds")) or 0.0
        for sample in samples
    )


def _sqa_summary_payload(samples: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _build_sqa_summary(samples)
    return {"sqa_summary": summary} if summary else {}


def _build_sqa_summary(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_by_engine: dict[tuple[str, str], dict[str, Any]] = {}
    for sample in samples:
        scores = sample.get("sqa_scores")
        if not isinstance(scores, list):
            continue
        for score_item in scores:
            if not isinstance(score_item, dict):
                continue
            engine_name = str(score_item.get("engine_name") or "")
            target = str(score_item.get("target") or "")
            if not engine_name:
                continue
            key = (engine_name, target)
            summary = summary_by_engine.setdefault(
                key,
                {
                    "engine_name": engine_name,
                    "target": target,
                    "scores": [],
                    "failed_count": 0,
                },
            )
            score = _as_float(score_item.get("score"))
            if score is None:
                summary["failed_count"] += 1
            else:
                summary["scores"].append(score)

    payload: list[dict[str, Any]] = []
    for summary in summary_by_engine.values():
        scores = summary["scores"]
        payload.append(
            {
                "engine_name": summary["engine_name"],
                "target": summary["target"],
                "mean_score": float(np.mean(scores)) if scores else None,
                "scored_count": len(scores),
                "failed_count": summary["failed_count"],
            }
        )
    return payload


def _score_for_engine(scores: list[dict[str, Any]], engine_name: str) -> float | None:
    for score_item in scores:
        if not isinstance(score_item, dict):
            continue
        if str(score_item.get("engine_name") or "").upper() != engine_name:
            continue
        return _as_float(score_item.get("score"))
    return None


def _score_delta(
    original_score: float | None,
    denoised_score: float | None,
) -> float | None:
    if original_score is None or denoised_score is None:
        return None
    return denoised_score - original_score


def _finite_values(values: Any) -> list[float]:
    return [
        value
        for value in (_as_float(item) for item in values)
        if value is not None
    ]


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _normalize_lid_language(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("_", "-")
    if normalized in {"en", "eng", "en-us", "en-gb", "english"}:
        return "en"
    if normalized in {
        "cn",
        "zh",
        "zho",
        "chi",
        "zh-cn",
        "zh-tw",
        "zh-hans",
        "zh-hant",
        "chinese",
    }:
        return "zh"
    return normalized


def _lid_predicted_language(
    raw_language: str,
    confidence: float,
    threshold: float,
) -> str:
    if confidence < threshold:
        return LID_UNKNOWN_LANGUAGE
    return str(raw_language)

def _task_inference_concurrency(
    request: EvaluationRequest,
    task: EvaluationTask,
) -> int:
    if task == "asr":
        return request.asr_inference_concurrency or request.inference_concurrency
    if task == "keyword":
        return request.asr_inference_concurrency or request.inference_concurrency
    if task == "vad":
        return request.vad_inference_concurrency or request.inference_concurrency
    if task == "lid":
        return request.lid_inference_concurrency or request.inference_concurrency
    if task == "denoise":
        return request.inference_concurrency
    return request.inference_concurrency


def _safe_divide_float(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


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


def _build_wer_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _build_asr_alignment_report(rows, metric_fn=get_wer)


def _build_cer_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _build_asr_alignment_report(rows, metric_fn=get_cer)


def _build_asr_alignment_report(
    rows: list[dict[str, Any]],
    *,
    metric_fn: Callable[[list[str], list[str], list[str]], Any],
) -> dict[str, Any]:
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

    alignment_result = metric_fn(
        [row["reference"] for row in rows],
        [row["hypothesis"] for row in rows],
        [row["id"] for row in rows],
    )
    summary = alignment_result.summary
    utterance_summaries = {
        group.name.strip("()"): group.counts for group in alignment_result.groups
    }
    row_by_id = {str(row["id"]): row for row in rows}
    row_by_index = {
        int(row["index"]): row
        for row in rows
        if isinstance(row.get("index"), int)
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
            _wer_utterance_to_payload(
                utterance=utterance,
                fallback_index=index,
                summary=utterance_summaries[utterance.id.strip("()")],
                row=row_by_id.get(utterance.id.strip("()"))
                or row_by_index.get(index)
                or {},
            )
            for index, utterance in enumerate(alignment_result.utterances, start=1)
        ],
    }


def _wer_utterance_to_payload(
    *,
    utterance: Any,
    fallback_index: int,
    summary: Any,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": utterance.id.strip("()"),
        "index": row.get("index", fallback_index),
        "audio_url": row.get("audio_url"),
        "duration_seconds": row.get("duration_seconds"),
        "sqa_scores": row.get("sqa_scores", []),
        "summary": _wer_counts_to_payload(summary),
        "tokens": [
            {
                "label": token.eval_label,
                "ref": token.ref_word,
                "hyp": token.hyp_word,
            }
            for token in utterance.tokens
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
