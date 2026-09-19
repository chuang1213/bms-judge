"""Task A：局部几何关系预测（pretext task 候选）。

与 T1 masked reconstruction 的本质区别：
  T1 的损失按 cell 计算，绝大多数 cell 是空 —— 模型可以靠"预测空"拿低 loss；
  Task A 的监督信号只落在被遮挡的**真实 note** 上，并预测它与可见上下文的
  客观几何关系（lane distance / 移动方向 / 相对时间 / 是否同时），
  因此无法用"预测空"偷分，必须对具体 lane 位置与时间关系作出承诺。

Target 全部由原始 BMS 自动生成，无人工 skill 标签。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .grid_data import CELL_SEC, KEY_LANES, N_LANES, T

HEAD_NAMES = ["lane_dist", "direction", "dt_prev", "simult"]
N_CLASSES = {"lane_dist": 5, "direction": 4, "dt_prev": 6, "simult": 2}


def _dt_bucket(dt_sec: float) -> int:
    if dt_sec < 0.05:
        return 0
    if dt_sec < 0.15:
        return 1
    if dt_sec < 0.30:
        return 2
    if dt_sec < 0.60:
        return 3
    return 4


def extract_focal_targets(grid: np.ndarray, input_grid: np.ndarray,
                          masks_int: np.ndarray) -> dict:
    """从 (全量 target grid, 遮挡后 input grid, mask) 提取 focal notes 与关系 target。

    返回 dict：
      flat_idx  [n_f]  t*L + l
      cls       [n_f, 4]（lane_dist, direction, dt_prev, simult 的类别）
      in_span   [n_f] bool（focal 是否在连续 span 内）
      edge      [n_f] bool（span 内且距 span 边界 < 0.5s）
      span_frac [n_f] float（距 span 边界的归一化距离）
    """
    T_, L, _ = grid.shape
    visible = input_grid[:, :, 0] > 0
    focal = (masks_int != 0) & (grid[:, :, 0] > 0)
    ts, ls = np.where(focal)
    n = len(ts)
    out = {
        "flat_idx": (ts * L + ls).astype(np.int64),
        "cls": np.zeros((n, 4), dtype=np.int64),
        "in_span": (masks_int[ts, ls] == 1),
        "edge": np.zeros(n, dtype=bool),
        "span_frac": np.zeros(n, dtype=np.float32),
    }
    if n == 0:
        return out
    vis_any = visible.any(axis=1)
    vis_times = np.where(vis_any)[0]
    # simult：该 cell 是否属于 chord（同一时间 ≥2 lane 有 onset）
    chord_cell = grid[:, :, 0].sum(axis=1) >= 2
    out["cls"][:, 3] = chord_cell[ts].astype(np.int64)

    # span 边界（mask==1 的连续区间；所有 lane 相同，取 lane 0）
    span_flag = masks_int[:, 0] == 1
    if span_flag.any():
        span_t = np.where(span_flag)[0]
        s0, s1 = int(span_t[0]), int(span_t[-1])
        span_len = max(s1 - s0, 1)
        dist_to_edge = np.minimum(ts - s0, s1 - ts)
        out["edge"] = (out["in_span"]) & (dist_to_edge < 30)   # <0.5s 视为边界
        out["span_frac"] = (dist_to_edge / span_len).astype(np.float32)

    for i in range(n):
        t, l = int(ts[i]), int(ls[i])
        pos = np.searchsorted(vis_times, t, side="left") - 1
        if pos < 0:
            continue   # 无前序可见 note：保持 no-prev 类
        prev_t = int(vis_times[pos])
        lanes_at = np.where(visible[prev_t])[0]
        best = lanes_at[np.argmin(np.abs(lanes_at - l))]
        d_lane = int(abs(l - best))
        d_t = (t - prev_t) * CELL_SEC
        out["cls"][i, 0] = min(3, d_lane) if d_lane <= 3 else 3   # 3+ 归入 3
        out["cls"][i, 1] = int(np.sign(l - best)) + 1             # -1,0,+1 -> 0,1,2
        out["cls"][i, 2] = _dt_bucket(d_t)
    return out


def focal_stats_features(input_grid: np.ndarray, masks_int: np.ndarray,
                         t: int, l: int) -> np.ndarray:
    """statistics-only baseline 的 per-note 特征：只看可见上下文，不含被遮挡内容。"""
    visible = input_grid[:, :, 0] > 0
    feats = [
        float(visible[max(0, t - 30):t + 31].sum()),      # ±0.5s 可见 onset
        float(visible[max(0, t - 90):t + 91].sum()),      # ±1.5s
        float(visible[:t].sum()),                          # 之前总数
        float(visible[t + 1:].sum()),                      # 之后总数
        float(visible[max(0, t - 60):t, l].sum()),         # 同 lane 之前 1s
        float(visible[t + 1:t + 61, l].sum()),             # 同 lane 之后 1s
        float(visible[t, :].sum()),                        # 同 cell 其他 lane
        float(t) / T,
        float(visible.sum()) / (T * N_LANES),              # 窗口可见密度
    ]
    per_lane = visible.sum(axis=0).astype(np.float64) / max(T, 1)
    feats += per_lane.tolist()
    dens4 = np.array([visible[60 * k:60 * (k + 1)].sum() for k in range(4)],
                     dtype=np.float64) / 60.0
    feats += dens4.tolist()
    return np.asarray(feats, dtype=np.float32)


class TaskAModel(nn.Module):
    """Task A 模型：GridEncoder cell features + 4 个 1x1 conv head（每 cell 分类）。"""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        in_ch = 64 + 64
        self.heads = nn.ModuleDict({
            name: nn.Conv2d(in_ch, n, 1) for name, n in N_CLASSES.items()
        })

    def cell_logits(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.encoder.cell_features(x)
        return {name: head(feats) for name, head in self.heads.items()}

    def forward(self, x: torch.Tensor):
        return self.cell_logits(x)


class PooledOnlyTaskAModel(nn.Module):
    """Task A 的 pooled-only 版本：head 只能读 pooled latent + 查询位置。

    结构：input window → encoder → pooled latent → geometry prediction heads。
    head 输入 = concat(pooled[B,D], 查询位置 (t/T, l/(L-1)))。
    没有 per-cell feature map 旁路：模型若想答对，必须把结构信息写进 pooled 向量。
    """

    def __init__(self, encoder, pooled_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.encoder = encoder
        self.heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(pooled_dim + 2, hidden), nn.ReLU(),
                nn.Linear(hidden, n))
            for name, n in N_CLASSES.items()
        })

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.pool(x)

    def forward_queries(self, pooled: torch.Tensor, bpos: torch.Tensor,
                        t_norm: torch.Tensor, l_norm: torch.Tensor):
        q = torch.stack([t_norm, l_norm], dim=1).float()
        inp = torch.cat([pooled[bpos], q], dim=1)
        return {name: head(inp) for name, head in self.heads.items()}
