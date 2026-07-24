from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def seconds_to_time(value: float) -> str:
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"时间必须是非负有限值: {value}")
    total_milliseconds = int(round(value * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


def time_to_seconds(value: str) -> float:
    parts = value.strip().split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            result = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            minutes, seconds = parts
            result = int(minutes) * 60 + float(seconds)
        else:
            raise ValueError
    except ValueError as exc:
        raise ValueError(f"时间格式错误: {value}") from exc
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"时间必须是非负有限值: {value}")
    return result


def au_path_to_seconds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path, delimiter="\t")
    missing = sorted({"Start", "Duration"} - set(frame.columns))
    if missing:
        raise ValueError(f"Audition CSV 缺少字段: {', '.join(missing)}")
    starts = frame["Start"].apply(time_to_seconds).to_numpy(dtype=np.float64)
    durations = frame["Duration"].apply(time_to_seconds).to_numpy(dtype=np.float64)
    return starts, durations


def seconds_to_mask(
    starts: np.ndarray,
    durations: np.ndarray,
    *,
    length: int,
    sample_rate: int,
) -> np.ndarray:
    starts = np.asarray(starts, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    if starts.shape != durations.shape:
        raise ValueError(f"starts 与 durations 长度不一致: {starts.shape} != {durations.shape}")
    if starts.ndim != 1:
        raise ValueError(f"starts 与 durations 必须是一维数组: shape={starts.shape}")
    if length < 0 or sample_rate <= 0:
        raise ValueError(f"length 和 sample_rate 必须有效: length={length} sample_rate={sample_rate}")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(durations)):
        raise ValueError("starts 与 durations 必须是有限数值")
    if np.any(durations < 0):
        raise ValueError("durations 不能包含负数")

    mask = np.zeros(length, dtype=bool)
    for start, duration in zip(starts, durations, strict=True):
        end = start + duration
        start_index = max(0, min(length, int(round(start * sample_rate))))
        end_index = max(0, min(length, int(round(end * sample_rate))))
        if end_index > start_index:
            mask[start_index:end_index] = True
    return mask


def mask_to_seconds(
    mask: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.ndim != 1:
        raise ValueError(f"mask 必须是一维 bool 数组: shape={mask.shape} dtype={mask.dtype}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate 必须大于 0: {sample_rate}")
    if mask.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    padded = np.pad(mask.astype(np.int8), (1, 1), mode="constant")
    edges = np.diff(padded)
    starts = np.where(edges == 1)[0] / sample_rate
    ends = np.where(edges == -1)[0] / sample_rate
    return starts, ends - starts


def seconds_to_au_df(
    starts: np.ndarray,
    durations: np.ndarray,
) -> pd.DataFrame:
    starts = np.asarray(starts, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    if starts.shape != durations.shape:
        raise ValueError(f"starts 与 durations 长度不一致: {starts.shape} != {durations.shape}")
    return pd.DataFrame(
        {
            "Name": np.arange(len(starts)),
            "Start": [seconds_to_time(float(value)) for value in starts],
            "Duration": [seconds_to_time(float(value)) for value in durations],
            "Time Format": "decimal",
            "Type": "Cue",
            "Description": "",
        }
    )


def mask_to_au_df(mask: np.ndarray, sample_rate: int) -> pd.DataFrame:
    starts, durations = mask_to_seconds(mask, sample_rate)
    return seconds_to_au_df(starts, durations)


def au_path_to_mask(
    path: Path,
    *,
    length: int,
    sample_rate: int,
) -> np.ndarray:
    starts, durations = au_path_to_seconds(path)
    return seconds_to_mask(
        starts,
        durations,
        length=length,
        sample_rate=sample_rate,
    )


def au_df_to_mask(
    frame: pd.DataFrame,
    *,
    length: int,
    sample_rate: int,
) -> np.ndarray:
    starts = frame["Start"].apply(time_to_seconds).to_numpy(dtype=np.float64)
    durations = frame["Duration"].apply(time_to_seconds).to_numpy(dtype=np.float64)
    return seconds_to_mask(
        starts,
        durations,
        length=length,
        sample_rate=sample_rate,
    )
