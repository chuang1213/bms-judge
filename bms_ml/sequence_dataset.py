"""Sequence 窗口数据集。

把每张谱面的 N x 4 note sequence 切成固定长度窗口：
- 窗口 = 连续 W 个 event（按保存顺序，即按时间排序）；
- stride = W（不重叠）；不足 W 的尾部丢弃（每谱面最多丢 W-1 个 event）；
- 每个窗口的标签 = 所在谱面的 difficulty；
- 同一谱面的所有窗口永远在同一 split（无跨 split leakage）；
- 每列做 z-score 归一化（在 train 窗口上拟合，参数保存）。

注：训练时同一谱面的窗口是相关的（可视为一种增广），
评估时按谱面取窗口预测的均值，因此评估口径不受窗口相关性影响。
"""

from __future__ import annotations

import json
import os
import sys
from typing import List, Tuple

import numpy as np


def load_sequences(records, seq_root: str) -> dict:
    out = {}
    for r in records:
        out[r["sha256"]] = np.load(os.path.join(seq_root, r["sha256"] + ".npy"))
    return out


def build_windows(records, seq_root: str, window_len: int, stride: int,
                  label_of) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """返回 (windows [M, W, 4], chart_sha256 list, labels [M], stats)。"""
    windows, shas, labels = [], [], []
    per_chart = {}
    for r in records:
        seq = np.load(os.path.join(seq_root, r["sha256"] + ".npy"))
        n = seq.shape[0]
        n_win = 0
        for start in range(0, n - window_len + 1, stride):
            windows.append(seq[start:start + window_len])
            shas.append(r["sha256"])
            labels.append(label_of(r))
            n_win += 1
        per_chart[r["sha256"]] = n_win
    return (np.asarray(windows, dtype=np.float32),
            np.asarray(shas),
            np.asarray(labels, dtype=np.float32),
            per_chart)


def fit_standardize(windows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = windows.reshape(-1, windows.shape[-1]).mean(axis=0)
    std = windows.reshape(-1, windows.shape[-1]).std(axis=0)
    std[std == 0] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardize(windows: np.ndarray, mean, std) -> np.ndarray:
    return ((windows - mean) / std).astype(np.float32)


def build_all(records_by_split: dict, seq_root: str, window_len: int, stride: int,
              label_of, out_json: str) -> dict:
    """records_by_split: {"train": [...], "val": [...], "test": [...]}"""
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    result = {}
    stats = {}
    for name, records in records_by_split.items():
        win, shas, labels, per_chart = build_windows(
            records, seq_root, window_len, stride, label_of)
        result[name] = (win, shas, labels)
        stats[name] = {
            "charts": len(records),
            "windows": len(shas),
            "windows_per_chart": {
                "min": min(per_chart.values()) if per_chart else 0,
                "max": max(per_chart.values()) if per_chart else 0,
                "median": float(np.median(list(per_chart.values()))) if per_chart else 0,
            },
        }
    # z-score 只在 train 上拟合
    mean, std = fit_standardize(result["train"][0])
    for name in result:
        win, shas, labels = result[name]
        result[name] = (apply_standardize(win, mean, std), shas, labels)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "window_len": window_len,
            "stride": stride,
            "normalization": {"mean": mean.tolist(), "std": std.tolist()},
            "split_stats": stats,
        }, f, ensure_ascii=False, indent=2)
    return result
