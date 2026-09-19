"""Representation v1 的固定时间网格数据工具。

窗口：4s，cell = 1/60s（T=240），lane 轴 8（0-6 keys 有序 + 7 scratch 平面）。
通道：
  C0 = onset 计数（cap 2，2 表示极快重复）；
  C1 = LN hold 标志（v1 最简 hold 表示，可由 include_hold=False 关闭做 ablation）。
所有时间为 window-relative（窗口内 0..4s）。
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import numpy as np

WINDOW_SEC = 4.0
CELL_SEC = 1.0 / 60.0
T = int(round(WINDOW_SEC / CELL_SEC))          # 240
N_LANES = 8                                     # 7 keys + scratch
KEY_LANES = 7
SCRATCH_LANE = 7
N_CH = 2                                        # onset, hold


class SeqCache:
    """按 sha256 缓存 sequence npy（LRU）。"""

    def __init__(self, seq_root: str, capacity: int = 1024):
        self.seq_root = seq_root
        self.capacity = capacity
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def __call__(self, sha: str) -> np.ndarray:
        if sha not in self._cache:
            if len(self._cache) >= self.capacity:
                self._cache.popitem(last=False)
            self._cache[sha] = np.load(os.path.join(self.seq_root, sha + ".npy"))
        return self._cache[sha]


def seq_grid(seq: np.ndarray, t0: float, include_hold: bool = True,
             window_sec: float = WINDOW_SEC) -> np.ndarray:
    """把 N×4 note sequence 转成一个窗口网格 [T, N_LANES, 2] float32。"""
    t1 = t0 + window_sec
    grid = np.zeros((T, N_LANES, N_CH), dtype=np.float32)
    ts = seq[:, 0]
    lanes = seq[:, 1]
    types = seq[:, 2]
    durs = seq[:, 3]

    onset = (ts >= t0) & (ts < t1)
    if onset.any():
        cells = np.clip(((ts[onset] - t0) / CELL_SEC).astype(np.int64), 0, T - 1)
        ls = lanes[onset].astype(np.int64)
        ok = (ls >= 0) & (ls < N_LANES)
        if ok.any():
            np.add.at(grid[:, :, 0], (cells[ok], ls[ok]), 1.0)
    grid[:, :, 0] = np.minimum(grid[:, :, 0], 2.0)

    if include_hold:
        hold = (types == 1) & (durs > 0) & ((ts + durs) > t0) & (ts < t1)
        if hold.any():
            hs = ts[hold]
            ls = lanes[hold].astype(np.int64)
            de = durs[hold]
            c0 = np.clip(((hs - t0) / CELL_SEC).astype(np.int64), 0, T - 1)
            c1 = np.clip(((hs + de - t0) / CELL_SEC).astype(np.int64), 0, T - 1)
            ok = (ls >= 0) & (ls < N_LANES)
            for c0i, c1i, li in zip(c0[ok], c1[ok], ls[ok]):
                grid[c0i:c1i + 1, li, 1] = 1.0
    return grid


def window_starts(duration: float, window_sec: float = WINDOW_SEC,
                  stride: Optional[float] = None) -> List[float]:
    stride = stride or window_sec
    if duration <= window_sec + 1e-9:
        return []
    return list(np.arange(0.0, duration - window_sec + 1e-9, stride))


def pick_starts(starts: Sequence[float], max_windows: Optional[int],
                rng: np.random.RandomState) -> List[float]:
    if max_windows is None or len(starts) <= max_windows:
        return list(starts)
    idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
    return [starts[i] for i in sorted(set(idx.tolist()))]


def window_items(records: Sequence[dict], seq_root: str,
                 max_windows: Optional[int] = None,
                 seed: int = 0) -> List[Tuple[str, float]]:
    """返回 [(sha256, window_start)]，每谱面窗口不重叠；train 可用 max_windows 采样。"""
    cache = SeqCache(seq_root)
    rng = np.random.RandomState(seed)
    items: List[Tuple[str, float]] = []
    for r in records:
        seq = cache(r["sha256"])
        if len(seq) == 0:
            continue
        dur = float(seq[-1, 0] - seq[0, 0])
        starts = window_starts(dur)
        if not starts:
            starts = [0.0]
        for s in pick_starts(starts, max_windows, rng):
            items.append((r["sha256"], float(s)))
    return items


def grids_for(items: Sequence[Tuple[str, float]], seq_root: str,
              include_hold: bool = True,
              verbose: bool = True) -> np.ndarray:
    """构建 [n, T, L, 2] uint8 网格数组（值域 0..2，省内存）。"""
    cache = SeqCache(seq_root)
    out = np.zeros((len(items), T, N_LANES, N_CH), dtype=np.uint8)
    for i, (sha, start) in enumerate(items):
        seq = cache(sha)
        out[i] = seq_grid(seq, start, include_hold=include_hold)
        if verbose and (i + 1) % 50000 == 0:
            print(f"  grids built: {i + 1}/{len(items)}")
    return out


# ---------------- 程序化变换（只用于 R4 探针与 ablation） ----------------

def permute_lanes_seq(seq: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """全局 lane 置换：只置换 key lanes（0..6），scratch（7）不动。"""
    out = seq.copy()
    lanes = out[:, 1].astype(np.int64)
    m = lanes < KEY_LANES
    out[m, 1] = perm[lanes[m]]
    return out


KEY_PERMS: List[np.ndarray] = [
    np.array([0, 1, 2, 3, 4, 5, 6]),
    np.array([6, 5, 4, 3, 2, 1, 0]),
    np.array([1, 0, 2, 3, 4, 5, 6]),
    np.array([0, 2, 1, 3, 4, 5, 6]),
    np.array([0, 1, 3, 2, 4, 5, 6]),
    np.array([2, 1, 0, 3, 4, 5, 6]),
    np.array([3, 2, 1, 0, 4, 5, 6]),
    np.array([0, 1, 2, 3, 6, 5, 4]),
]


def local_shuffle_window(seq: np.ndarray, t0: float, rng: np.random.RandomState,
                         window_sec: float = WINDOW_SEC) -> np.ndarray:
    """局部 lane shuffle：窗口内 onset 的 lane 值随机重排（保留 lane 频率与时间/密度）。"""
    out = seq.copy()
    t1 = t0 + window_sec
    m = (out[:, 0] >= t0) & (out[:, 0] < t1)
    lanes = out[m, 1].copy()
    rng.shuffle(lanes)
    out[m, 1] = lanes
    return out


def shift_seq(seq: np.ndarray, dt: float) -> np.ndarray:
    out = seq.copy()
    out[:, 0] += dt
    return out


# ---------------- density-only baseline（低阶统计） ----------------

def density_baseline_masks(grids: np.ndarray, masks_int: np.ndarray,
                           neighbor_cells: int = 60) -> np.ndarray:
    """只用局部 density/occupancy 预测被遮挡区域。

    masks_int: [n, T, L] int8（0=未遮挡，1=span，2=random）。
    返回 probs [n, T, L, 2]（onset 与 hold 的概率）。
    p_onset(cell, lane) = 0.6 * 邻域(±neighbor_cells)未遮挡 onset 率 + 0.4 * 窗口内该 lane 未遮挡 onset 率。
    p_hold(cell, lane)  = 窗口内该 lane 未遮挡 hold 率。
    """
    n = grids.shape[0]
    probs = np.zeros((n, T, N_LANES, N_CH), dtype=np.float32)
    onset_bin = (grids[:, :, :, 0] > 0).astype(np.float32)
    hold_bin = (grids[:, :, :, 1] > 0).astype(np.float32)
    unmasked = (masks_int == 0).astype(np.float32)   # [n, T, L]

    # 窗口内 per-lane 未遮挡率
    cnt = unmasked.sum(axis=1, keepdims=True)        # [n, 1, L]
    cnt = np.maximum(cnt, 1e-6)
    lane_onset_rate = (onset_bin * unmasked).sum(axis=1, keepdims=True) / cnt   # [n,1,L]
    lane_hold_rate = (hold_bin * unmasked).sum(axis=1, keepdims=True) / cnt

    # 邻域率：沿时间轴累积（只统计未遮挡 cell）
    uo = onset_bin * unmasked                        # [n,T,L]
    cum = np.concatenate([np.zeros((n, 1, N_LANES)), np.cumsum(uo, axis=1)], axis=1)
    k = neighbor_cells
    neigh_num = cum[:, 2 * k + 1:] - cum[:, :-2 * k - 1] if T > 2 * k else np.sum(uo, axis=1, keepdims=True).repeat(T, axis=1)
    uc = unmasked
    cumc = np.concatenate([np.zeros((n, 1, N_LANES)), np.cumsum(uc, axis=1)], axis=1)
    neigh_den = cumc[:, 2 * k + 1:] - cumc[:, :-2 * k - 1]
    # 边缘处理：T=240 > 120，所以中心窗口完整；直接赋值
    neigh_rate = neigh_num / np.maximum(neigh_den, 1e-6)   # [n, T-2k, L]
    neigh_full = np.zeros((n, T, N_LANES), dtype=np.float32)
    neigh_full[:, k:T - k] = neigh_rate
    # 边界：退化为窗口全局率
    neigh_full[:, :k] = lane_onset_rate[:, 0, :][:, None, :]
    neigh_full[:, T - k:] = lane_onset_rate[:, 0, :][:, None, :]

    probs[:, :, :, 0] = 0.6 * neigh_full + 0.4 * lane_onset_rate[:, 0, :][:, None, :]
    probs[:, :, :, 1] = np.broadcast_to(lane_hold_rate[:, 0, :][:, None, :], (n, T, N_LANES))
    return probs


def density_baseline_next(cur_grids: np.ndarray) -> np.ndarray:
    """T2 baseline：用当前窗口的低阶统计预测下一窗口。
    返回 probs [n, T, L, 2]：p_onset = 当前窗口 per-lane onset 率 + 全局密度延续项；
    p_hold = 当前窗口 per-lane hold 率。
    """
    n = cur_grids.shape[0]
    probs = np.zeros((n, T, N_LANES, N_CH), dtype=np.float32)
    onset_bin = (cur_grids[:, :, :, 0] > 0).astype(np.float32)
    hold_bin = (cur_grids[:, :, :, 1] > 0).astype(np.float32)
    lane_rate = onset_bin.sum(axis=1) / max(T, 1)          # [n,L]
    hold_rate = hold_bin.sum(axis=1) / max(T, 1)
    probs[:, :, :, 0] = lane_rate[:, None, :]
    probs[:, :, :, 1] = hold_rate[:, None, :]
    return np.clip(probs, 0.0, 1.0)
