from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd


def seconds_to_time(t: float) -> str:
    hours = int(t // 3600)
    t_remainder = t % 3600
    minutes = int(t_remainder // 60)
    t_remainder = t_remainder % 60
    seconds = int(t_remainder)
    milliseconds = (t_remainder - seconds) * 1000

    if hours > 0:
        time_str = f"{hours}:{minutes:02d}:{seconds:02d}.{milliseconds:03.0f}"
    else:
        time_str = f"{minutes}:{seconds:02d}.{milliseconds:03.0f}"

    return time_str


def time_to_seconds(time_str: str) -> float:
    """
    将时间字符串（格式为 H:MM:SS.sss 或 MM:SS.sss）转换为以秒为单位的浮点数。

    参数:
        time_str (str): 需要转换的时间字符串。

    返回:
        float: 转换后的时间（单位：秒）。
    """
    parts = time_str.split(":")
    if len(parts) == 3:  # 格式为 H:MM:SS.sss
        hours, minutes, seconds = parts
        total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:  # 格式为 MM:SS.sss
        minutes, seconds = parts
        total_seconds = int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"时间格式错误: {time_str}")
    return total_seconds


def au_path_to_seconds(path_obj: Path) -> Tuple[np.ndarray, np.ndarray]:
    # 读取 CSV 文件
    df = pd.read_csv(path_obj, delimiter="\t")
    # 将 'Start' 和 'Duration' 列中的时间字符串转换为秒
    start_seconds = df["Start"].apply(time_to_seconds).to_numpy()
    duration_seconds = df["Duration"].apply(time_to_seconds).to_numpy()

    return start_seconds, duration_seconds


def seconds_to_mask(
    starts: np.ndarray, durations: np.array, length: int, sr: int
) -> np.ndarray:
    """
    将开始时间和持续时间转换为布尔掩码。
    """
    ends = starts + durations
    # 将时间转为索引位置，四舍五入
    starts = np.round(starts * sr).astype(int)
    ends = np.round(ends * sr).astype(int)
    # 初始化结果数组
    dash_dot_mask = np.zeros(length, dtype=bool)
    # 如果csv文件中没有任何标签，则返回一个全为False的数组
    if len(starts) == 0:
        return dash_dot_mask
    # 设置mask中所有starts和ends之间的位置为True
    for start_index, end_index in zip(starts, ends):
        dash_dot_mask[start_index:end_index] = True

    return dash_dot_mask


def mask_to_seconds(mask: np.ndarray, sr: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    将1维布尔型NumPy数组转换为True值开始和结束的时间段。
    """
    # 计算布尔值的差分，识别变化点
    edges = np.diff(mask.astype(int))
    # 找到从False到True的上升沿索引
    rising_edges = np.where(edges == 1)[0] + 1
    # 找到从True到False的下降沿索引
    falling_edges = np.where(edges == -1)[0] + 1
    # 如果数组以True开始，补充起始时间0
    if mask[0]:
        rising_edges = np.insert(rising_edges, 0, 0)
    # 如果数组以True结束，补充结束时间为数组长度对应的时间
    if mask[-1]:
        falling_edges = np.append(falling_edges, len(mask))
    # 将索引转换为时间
    starts = rising_edges / sr
    ends = falling_edges / sr
    durations = ends - starts
    # 返回包含开始和结束时间的DataFrame
    return starts, durations


def seconds_to_au_df(starts: np.ndarray, durations: np.ndarray) -> pd.DataFrame:
    """
    将bool_array_to_times函数的输出转换为指定格式的DataFrame。

    参数:
    - time_df: 包含'start_times'和'end_times'列的DataFrame

    返回:
    - 包含指定格式列的DataFrame
    """
    # 创建DataFrame
    time_df = pd.DataFrame({"start_times": starts, "end_times": starts + durations})
    # 计算持续时间
    time_df["Duration"] = time_df["end_times"] - time_df["start_times"]
    # 格式化开始时间和持续时间
    time_df["Start"] = time_df["start_times"].apply(seconds_to_time)
    time_df["Duration"] = time_df["Duration"].apply(seconds_to_time)
    # 创建指定格式的DataFrame
    formatted_df = pd.DataFrame(
        {
            "Name": time_df.index,
            "Start": time_df["Start"],
            "Duration": time_df["Duration"],
            "Time Format": "decimal",
            "Type": "Cue",
            "Description": "",
        }
    )

    return formatted_df.reset_index(drop=True)


def mask_to_au_df(mask: np.ndarray, sr: float) -> pd.DataFrame:
    """
    将布尔掩码转换为 Audition CSV 文件的 DataFrame 格式。

    参数:
        mask (np.ndarray): 布尔掩码数组。
        sr (float): 采样率。

    返回:
        pd.DataFrame: Audition CSV 文件的 DataFrame 格式。
    """
    # 将布尔掩码转换为时间段
    starts, durations = mask_to_seconds(mask, sr)

    # 将时间段转换为 Audition CSV 文件的 DataFrame 格式
    au_df = seconds_to_au_df(starts, durations)

    return au_df


def au_path_to_mask(path_obj: Path, length: int, sr: int) -> np.ndarray:
    """
    将 Audition CSV 文件转换为布尔掩码。

    参数:
        path_obj (Path): Audition CSV 文件路径。
        length (int): 掩码长度。
        sr (int): 采样率。

    返回:
        np.ndarray: 布尔掩码。
    """
    # 将 Audition CSV 文件转换为时间段
    starts, durations = au_path_to_seconds(path_obj)

    # 将时间段转换为布尔掩码
    mask = seconds_to_mask(starts, durations, length, sr)

    return mask


def au_df_to_mask(au_df: pd.DataFrame, length: int, sr: int) -> np.ndarray:
    """
    将 Audition CSV 文件的 DataFrame 格式转换为布尔掩码。

    参数:
        au_df (pd.DataFrame): Audition CSV 文件的 DataFrame 格式。
        length (int): 掩码长度。
        sr (int): 采样率。
    返回:
        np.ndarray: 布尔掩码。
    """
    # 将 Audition CSV 文件的 DataFrame 格式转换为时间段
    starts = au_df["Start"].apply(time_to_seconds).to_numpy()
    durations = au_df["Duration"].apply(time_to_seconds).to_numpy()

    # 将时间段转换为布尔掩码
    mask = seconds_to_mask(starts, durations, length, sr)

    return mask
