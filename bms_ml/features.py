"""谱面 → 机器学习输入。

第一版保留两种表示：
A. 人工统计特征（MLP baseline 用）：chart-level 元数据的有序数组；
B. note event 序列（为以后 1D CNN / RNN / Transformer 准备），
   同时提供简单的 piano-roll 网格（为 sanity check 可视化/以后 CNN 准备）。

注意：这里不做几十个"我认为代表难度"的特征。A 的特征全部来自
timeline.ChartMeta 中"结构事实"（数量、时长、BPM/STOP、密度、和弦），
不包含任何人工难度公式。
"""

from __future__ import annotations

from typing import List

import numpy as np

from .timeline import ChartMeta, UnifiedChart


# 特征顺序固定（保证 dataset 与 scaler 一致）
FEATURE_NAMES: List[str] = [
    "total_notes",
    "ln_ratio",
    "duration_sec",
    "measures",
    "initial_bpm",
    "min_bpm",
    "max_bpm",
    "bpm_change_count",
    "stop_count",
    "stop_total_sec",
    "lane0_scratch",
    "lane1",
    "lane2",
    "lane3",
    "lane4",
    "lane5",
    "lane6",
    "lane7",
    "scratch_ratio",
    "avg_nps",
    "peak_nps_1s",
    "peak_measure_nps",
    "chord_count",
    "chord2_count",
    "chord3plus_count",
    "jack_count",
]


def build_stats_features(meta: ChartMeta) -> np.ndarray:
    """把 ChartMeta 转成定长 float32 向量。"""
    total = max(meta.total_notes, 1)
    c = meta.chord_size_hist
    chord3plus = sum(c[3:])
    values = [
        meta.total_notes,
        meta.ln_ratio,
        meta.duration_sec,
        meta.measures,
        meta.initial_bpm,
        meta.min_bpm,
        meta.max_bpm,
        meta.bpm_change_count,
        meta.stop_count,
        meta.stop_total_sec,
        meta.lane_counts[0],
        meta.lane_counts[1],
        meta.lane_counts[2],
        meta.lane_counts[3],
        meta.lane_counts[4],
        meta.lane_counts[5],
        meta.lane_counts[6],
        meta.lane_counts[7],
        meta.scratch_count / total,
        meta.avg_nps,
        meta.peak_nps_1s,
        meta.peak_measure_nps,
        meta.chord_count,
        c[2],
        chord3plus,
        meta.jack_count,
    ]
    return np.asarray(values, dtype=np.float32)


def build_note_sequence(chart: UnifiedChart) -> np.ndarray:
    """表示 B：每行 (time_sec, lane, type_code, duration_sec)。
    type_code: 0=normal, 1=ln。按时间排序。
    """
    rows = []
    for n in chart.notes:
        rows.append([n.time_sec, n.lane, 1 if n.note_type == "ln" else 0, n.duration_sec])
    rows.sort(key=lambda r: (r[0], r[1]))
    return np.asarray(rows, dtype=np.float32).reshape(-1, 4)


def build_piano_roll(chart: UnifiedChart, cell_sec: float = 0.25,
                     lanes: int = 8) -> np.ndarray:
    """简单 piano-roll 网格：T x lanes，0=空 1=普通 2=LN。
    仅用于 sanity 可视化与未来 CNN 实验。
    """
    dur = max(chart.meta.duration_sec, 1.0)
    t = int(np.ceil(dur / cell_sec))
    grid = np.zeros((t, lanes), dtype=np.int8)
    for n in chart.notes:
        col = min(n.lane, lanes - 1)
        i = int(n.time_sec / cell_sec)
        if i < t:
            grid[i, col] = 2 if n.note_type == "ln" else 1
        if n.note_type == "ln" and n.duration_sec > 0:
            j = int(n.end_time_sec / cell_sec)
            for k in range(i + 1, min(j + 1, t)):
                if grid[k, col] == 0:
                    grid[k, col] = 2
    return grid
