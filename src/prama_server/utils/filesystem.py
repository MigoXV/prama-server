from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4


def reject_output_inside_source(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if output == source or (source.is_dir() and source in output.parents):
        raise ValueError(f"输出目录不能位于输入路径内部: source={source} output={output}")


@contextmanager
def staged_output_directory(
    output: Path,
    *,
    overwrite: bool,
) -> Iterator[Path]:
    output = output.resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"输出目录已存在，请使用 --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.staging-",
            dir=output.parent,
        )
    )
    try:
        yield staging
        _commit_staged_directory(staging, output, overwrite=overwrite)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _commit_staged_directory(
    staging: Path,
    output: Path,
    *,
    overwrite: bool,
) -> None:
    if not output.exists():
        staging.replace(output)
        return
    if not overwrite:
        raise FileExistsError(f"输出目录已存在，请使用 --overwrite: {output}")

    backup = output.with_name(f".{output.name}.backup-{uuid4().hex}")
    output.replace(backup)
    try:
        staging.replace(output)
    except BaseException:
        backup.replace(output)
        raise
    shutil.rmtree(backup)
