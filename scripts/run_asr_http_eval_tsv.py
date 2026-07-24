from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="调用本地 HTTP ASR 评估接口，并导出每条样本 WER TSV。"
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--target", default="192.168.0.222:50011")
    parser.add_argument("--dataset-path", default="data-bin/cdb-asr-test")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="data-bin/tmp/cdb-asr-test-wer.tsv")
    parser.add_argument(
        "--snapshot-json",
        type=Path,
        default=None,
        help="已有 /api/evaluations/{job_id} 快照 JSON；提供后只导出 TSV，不重新发起评估",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-reference-words", type=int, default=0)
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--connect-timeout-seconds", type=float, default=10.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()

    api_base = args.api_base.rstrip("/")
    dataset_path = Path(args.dataset_path)
    output_path = Path(args.output)

    audio_by_id = _read_audio_paths(dataset_path=dataset_path, split=args.split)
    if args.snapshot_json is not None:
        with args.snapshot_json.open("r", encoding="utf-8") as file:
            snapshot = json.load(file)
        _write_wer_tsv(
            snapshot=snapshot,
            audio_by_id=audio_by_id,
            output_path=output_path,
        )
        return

    job_id = _create_evaluation(
        api_base=api_base,
        payload={
            "task": "asr",
            "target": args.target,
            "dataset_path": str(dataset_path),
            "split": args.split,
            "limit": None,
            "language_code": "en-US",
            "sample_rate": args.sample_rate,
            "min_reference_words": args.min_reference_words,
            "hotwords": [],
            "hotword_bias": 0,
            "connect_timeout_seconds": args.connect_timeout_seconds,
            "request_timeout_seconds": args.request_timeout_seconds,
            "interim_results": True,
            "remove_punctuation": False,
            "mask_frame_seconds": 0.01,
            "chunk_duration_seconds": 0.1,
            "speech_padding_seconds": 0.0,
            "hit_threshold": 0.9,
            "streaming": False,
        },
    )
    snapshot = _wait_evaluation(
        api_base=api_base,
        job_id=job_id,
        poll_interval_seconds=args.poll_interval_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    _write_wer_tsv(snapshot=snapshot, audio_by_id=audio_by_id, output_path=output_path)


def _read_audio_paths(*, dataset_path: Path, split: str) -> dict[str, str]:
    metadata_path = dataset_path / split / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.jsonl 不存在: {metadata_path}")

    audio_by_id: dict[str, str] = {}
    with metadata_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            item = json.loads(line)
            sample_id = str(item["id"])
            file_name = str(item["file_name"])
            audio_path = str(dataset_path / split / file_name)
            audio_by_id[sample_id] = audio_path
            audio_by_id[sample_id.lower()] = audio_path
    return audio_by_id


def _create_evaluation(*, api_base: str, payload: dict[str, Any]) -> str:
    logger.info("创建 ASR 评估任务: dataset=%s", payload["dataset_path"])
    response = _request_json(
        f"{api_base}/api/evaluations",
        method="POST",
        payload=payload,
    )
    job_id = str(response["job_id"])
    logger.info("评估任务已创建: job_id=%s", job_id)
    return job_id


def _wait_evaluation(
    *,
    api_base: str,
    job_id: str,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_processed: int | None = None
    while True:
        snapshot = _request_json(f"{api_base}/api/evaluations/{job_id}")
        status = snapshot.get("status")
        progress = snapshot.get("progress") or {}
        processed = progress.get("processed")
        total = progress.get("total")
        if processed != last_processed:
            logger.info(
                "评估进度: job_id=%s status=%s processed=%s total=%s",
                job_id,
                status,
                processed,
                total,
            )
            last_processed = processed

        if status == "completed":
            logger.info("评估完成: job_id=%s", job_id)
            return snapshot
        if status == "failed":
            raise RuntimeError(f"评估失败: {snapshot.get('error')}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"等待评估超时: job_id={job_id}")
        time.sleep(poll_interval_seconds)


def _write_wer_tsv(
    *,
    snapshot: dict[str, Any],
    audio_by_id: dict[str, str],
    output_path: Path,
) -> None:
    result = snapshot.get("result") or {}
    report = result.get("wer_report") or {}
    utterances = report.get("utterances") or []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "audio_file", "wer", "reference", "hypothesis"])
        for utterance in utterances:
            sample_id = str(utterance.get("id", ""))
            summary = utterance.get("summary") or {}
            tokens = utterance.get("tokens") or []
            writer.writerow(
                [
                    sample_id,
                    audio_by_id.get(sample_id) or audio_by_id.get(sample_id.lower(), ""),
                    summary.get("wer", ""),
                    _tokens_to_text(tokens, "ref"),
                    _tokens_to_text(tokens, "hyp"),
                ]
            )

    logger.info("逐条 WER TSV 已写入: %s rows=%s", output_path, len(utterances))


def _tokens_to_text(tokens: list[dict[str, Any]], key: str) -> str:
    words = [
        str(token[key])
        for token in tokens
        if token.get(key) not in (None, "")
    ]
    return " ".join(words)


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {url}: {detail}") from exc


if __name__ == "__main__":
    main()
