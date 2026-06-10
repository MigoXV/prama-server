from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="为音频目录就地生成 SE 评估 audiofolder metadata.jsonl。"
    )
    parser.add_argument("directory", type=Path, help="音频目录")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在的 metadata.jsonl",
    )
    args = parser.parse_args()

    write_metadata(directory=args.directory, overwrite=args.overwrite)


def write_metadata(*, directory: Path, overwrite: bool) -> Path:
    directory = directory.resolve()
    if not directory.exists() or not directory.is_dir():
        raise NotADirectoryError(f"输入目录不存在: {directory}")

    metadata_path = directory / "metadata.jsonl"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"metadata.jsonl 已存在，请使用 --overwrite: {metadata_path}")

    audio_files = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.name != "metadata.jsonl"
        and path.suffix.lower() in AUDIO_SUFFIXES
    ]
    if not audio_files:
        raise FileNotFoundError(f"输入目录中没有找到支持的音频文件: {directory}")

    used_ids: set[str] = set()
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        for audio_path in audio_files:
            relative_path = audio_path.relative_to(directory)
            sample_id = _unique_id(_safe_id(relative_path.with_suffix("")), used_ids)
            used_ids.add(sample_id)
            metadata_file.write(
                json.dumps(
                    {
                        "file_name": str(relative_path).replace("\\", "/"),
                        "id": sample_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info("metadata.jsonl 已生成: files=%s path=%s", len(audio_files), metadata_path)
    return metadata_path


def _safe_id(path: Path) -> str:
    value = "_".join(path.parts)
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in value.strip()
    )
    return safe or "sample"


def _unique_id(sample_id: str, used_ids: set[str]) -> str:
    if sample_id not in used_ids:
        return sample_id
    index = 2
    while f"{sample_id}_{index}" in used_ids:
        index += 1
    return f"{sample_id}_{index}"


if __name__ == "__main__":
    main()
