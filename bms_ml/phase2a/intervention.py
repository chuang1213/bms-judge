"""统计受控的局部 lane 置换（degree-preserving 2-switch）。

目标：只改变 note 在 lane 空间的排列关系，尽量保持所有低阶统计不变。

构造：对窗口内的 tap（type==0、key lanes 0-6）做随机 2-switch：
     (t1,l1), (t2,l2)  ->  (t1,l2), (t2,l1)
单次交换的性质：
  - 每个时间位置保持一个 note  → chord size 分布不变；
  - 每条 lane 的总数不变        → per-lane count 不变；
  - 时间、onset、duration 完全不变 → 局部 density / NPS 不变；
  - 只交换 lane 标签 → 只有空间排列改变。

排除项：
  - LN（type==1）不参与交换：LN 是长时程对象，移动其 lane 会改变 hold 通道的
    per-lane 统计，污染低阶统计控制；
  - scratch（lane==7）不参与交换：scratch 是独立机械通道，保持其时间位置，
    避免改变 scratch 的局部 density 统计。

注意：简单随机 per-note 置换会破坏 per-lane totals 与 chord size 分布；
per-position 置换会破坏 per-lane totals。2-switch 是保持两者同时成立的
最小组原语，因此本设计不是"简单随机 permutation"。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .grid_data import CELL_SEC, KEY_LANES, N_LANES, T, WINDOW_SEC


def swap_lanes_window(seq: np.ndarray, t0: float, rng: np.random.RandomState,
                      swaps_factor: float = 10.0,
                      window_sec: float = WINDOW_SEC) -> Tuple[np.ndarray, int]:
    """对窗口内 tap（key lanes）做随机 2-switch，返回 (变体 sequence, 成功交换次数)。"""
    t1 = t0 + window_sec
    m = (seq[:, 0] >= t0) & (seq[:, 0] < t1)
    idx = np.where(m)[0]
    out = seq.copy()
    taps = idx[(out[idx, 2] == 0) & (out[idx, 1] < KEY_LANES)]
    if len(taps) < 2:
        return out, 0
    # per-cell per-lane onset 计数（窗口内全部 note，含 LN-start；用于 cap 拒绝规则）
    times = out[idx, 0]
    cells = np.clip(((times - t0) / CELL_SEC).astype(np.int64), 0, T - 1)
    lanes = out[idx, 1].astype(np.int64)
    counts = np.zeros((T, N_LANES), dtype=np.int64)
    np.add.at(counts, (cells, lanes), 1)
    cell_of = {int(i): int(c) for i, c in zip(idx, cells)}
    n_swaps = max(1, int(round(swaps_factor * len(taps))))
    r = np.random.RandomState(rng.randint(0, 2 ** 31))
    done = 0
    for _ in range(n_swaps):
        i, j = r.choice(len(taps), 2, replace=False)
        i, j = int(taps[i]), int(taps[j])
        li, lj = int(out[i, 1]), int(out[j, 1])
        if li == lj:
            continue
        ci, cj = cell_of[i], cell_of[j]
        if ci == cj:
            continue   # 同 cell 交换是 no-op（排序后序列不变），跳过
        # 跨 cell 交换若会让目标 cell-lane 达到 3（超过网格 cap 2），拒绝：
        # 避免网格截断导致 per-lane 统计在模型输入层面漂移
        if counts[ci, lj] >= 2 or counts[cj, li] >= 2:
            continue
        counts[ci, li] -= 1
        counts[ci, lj] += 1
        counts[cj, lj] -= 1
        counts[cj, li] += 1
        out[i, 1], out[j, 1] = lj, li
        done += 1
    return out, done


def grid_marginals_equal(g1: np.ndarray, g2: np.ndarray) -> bool:
    """模型输入层面检查：per-position 计数与 per-lane totals（含两个通道）完全一致。"""
    return bool(np.array_equal(g1.sum(axis=1), g2.sum(axis=1)) and
                np.array_equal(g1.sum(axis=0), g2.sum(axis=0)))


def window_stats_seq(seq: np.ndarray, t0: float,
                     window_sec: float = WINDOW_SEC) -> Dict[str, np.ndarray]:
    """窗口内低阶统计 + 简单的空间聚合量（后者预期会被 intervention 改变）。"""
    t1 = t0 + window_sec
    rows = seq[(seq[:, 0] >= t0) & (seq[:, 0] < t1)]
    note_count = float(len(rows))
    per_lane = np.bincount(rows[:, 1].astype(np.int64),
                           minlength=N_LANES)[:N_LANES].astype(np.float64)
    if len(rows):
        cells = np.clip(((rows[:, 0] - t0) / CELL_SEC).astype(np.int64), 0, T - 1)
    else:
        cells = np.zeros(0, dtype=np.int64)
    per_pos = np.zeros(T, dtype=np.int64)
    np.add.at(per_pos, cells, 1)
    chord_hist = np.bincount(per_pos, minlength=9)[:9].astype(np.float64)
    density_profile = np.array([per_pos[60 * k:60 * (k + 1)].sum()
                                for k in range(4)], dtype=np.float64)
    nps = note_count / window_sec

    if len(rows) >= 2:
        lanes = rows[:, 1].astype(np.int64)
        dlane = np.abs(np.diff(lanes)).astype(np.float64)
        adjacent_frac = float((dlane == 1).mean())
        runs, cur, cur_len = [], None, 0
        for l in lanes:
            if l == cur:
                cur_len += 1
            else:
                if cur is not None:
                    runs.append(cur_len)
                cur, cur_len = l, 1
        runs.append(cur_len)
        max_run = float(max(runs))
        spans = []
        for c in range(T):
            ls = lanes[cells == c]
            if len(ls) >= 2:
                spans.append(float(ls.max() - ls.min()))
        span_mean = float(np.mean(spans)) if spans else 0.0
        span_std = float(np.std(spans)) if spans else 0.0
    else:
        adjacent_frac, max_run = 0.0, note_count
        span_mean = span_std = 0.0

    return {"note_count": note_count, "per_lane": per_lane,
            "chord_hist": chord_hist, "density_profile": density_profile,
            "nps": nps, "adjacent_frac": adjacent_frac, "max_run": max_run,
            "span_mean": span_mean, "span_std": span_std}


def tap_hamming(orig: np.ndarray, var: np.ndarray, t0: float,
                window_sec: float = WINDOW_SEC) -> float:
    """窗口内 tap 中 lane 改变的占比（0=没变）。"""
    t1 = t0 + window_sec
    m = ((orig[:, 0] >= t0) & (orig[:, 0] < t1) &
         (orig[:, 2] == 0) & (orig[:, 1] < KEY_LANES))
    n = int(m.sum())
    if n == 0:
        return 0.0
    return float((orig[m, 1] != var[m, 1]).mean())


def is_global_perm(orig: np.ndarray, var: np.ndarray, t0: float,
                   window_sec: float = WINDOW_SEC) -> bool:
    """变体是否为原窗口的全局 lane 置换（一致映射）——若是，则没有破坏结构。"""
    t1 = t0 + window_sec
    m = ((orig[:, 0] >= t0) & (orig[:, 0] < t1) &
         (orig[:, 2] == 0) & (orig[:, 1] < KEY_LANES))
    a = orig[m, 1].astype(np.int64)
    b = var[m, 1].astype(np.int64)
    if len(a) < 2:
        return False
    mapping: Dict[int, int] = {}
    for x, y in zip(a, b):
        if x in mapping and mapping[x] != y:
            return False
        mapping[x] = y
    return len(set(mapping.values())) == len(mapping)


def preserved_stats_diff(so: Dict[str, np.ndarray], sv: Dict[str, np.ndarray]) -> Dict[str, float]:
    """理论上应保持不变的量的最大绝对差（注意 per_lane/chord_hist/density 是数组）。"""
    out = {}
    for k in ("note_count", "nps"):
        out[k] = float(abs(so[k] - sv[k]))
    for k in ("per_lane", "chord_hist", "density_profile"):
        a, b = np.asarray(so[k]), np.asarray(sv[k])
        out[k] = float(np.max(np.abs(a - b))) if a.size else 0.0
    return out


def arrangement_delta(so: Dict[str, np.ndarray], sv: Dict[str, np.ndarray]) -> Dict[str, float]:
    """预期被 intervention 改变的空间聚合量（报告 confound 用）。"""
    return {
        "adjacent_frac_delta": float(abs(so["adjacent_frac"] - sv["adjacent_frac"])),
        "max_run_delta": float(abs(so["max_run"] - sv["max_run"])),
        "span_mean_delta": float(abs(so["span_mean"] - sv["span_mean"])),
        "span_std_delta": float(abs(so["span_std"] - sv["span_std"])),
    }
