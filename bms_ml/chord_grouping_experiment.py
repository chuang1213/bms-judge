"""Chord-grouping 表示实验。

问题：旧表示把同一个时间点的 chord 拆成多个 event（多行同 time），
本实验把它合并为"一个时间点一行"：

  新 event = [delta_t, scratch, k1, k2, ..., k7, has_ln]   （10 维）

模型不变：2 x Conv1d(kernel=5) → GAP → Linear（仅第一层 Linear 输入维变化）。
其余条件严格复用：Satellite / song split / z-score(train) / 训练流程 / seeds。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from .input_representation_experiment import (
    band_mae,
    chart_metrics,
    make_delta_rep,
    train_eval,
    zscore_train,
)
from .receptive_field_experiment import VariantCNN


NEW_DIM = 10
NEW_COLS = ["delta_t", "scratch", "k1", "k2", "k3", "k4", "k5", "k6", "k7", "has_ln"]


def build_timepoint_rep(seq: np.ndarray) -> np.ndarray:
    """按时间点合并：同一 time 的所有 note → 一行 [delta_t, lane presence..., has_ln]。

    输入 seq: [N, 4]（time, lane, type, duration）
    输出: [M, 10]（M = 不同时间点数），首行 delta_t 由后续统一填充 0。
    """
    ts = np.round(seq[:, 0], 3)
    groups = defaultdict(list)
    for t, lane, typ, dur in seq:
        groups[t].append((int(lane), int(typ), float(dur)))
    times = sorted(groups)
    rows = []
    for t in times:
        notes = groups[t]
        lanes = [n[0] for n in notes]
        present = [0] * 8
        for l in lanes:
            if 0 <= l <= 7:
                present[l] = 1
        has_ln = 1 if any(n[1] == 1 for n in notes) else 0
        rows.append(present + [has_ln])  # 先 lanes + has_ln，delta_t 随后算
    arr = np.asarray(rows, dtype=np.float32)  # [M, 9]
    dt = np.zeros((arr.shape[0], 1), dtype=np.float32)
    dt[1:, 0] = np.log1p(np.maximum(np.diff(times), 0.0))  # 与旧表示一致的 delta 归一化
    return np.concatenate([dt, arr], axis=1)  # [M, 10]


def build_new_windows(records, seq_root, window_len, stride):
    win, shas = [], []
    for r in records:
        seq = np.load(os.path.join(seq_root, r["sha256"] + ".npy"))
        tp = build_timepoint_rep(seq)
        n = tp.shape[0]
        for start in range(0, n - window_len + 1, stride):
            win.append(tp[start:start + window_len])
            shas.append(r["sha256"])
    return np.asarray(win, dtype=np.float32), np.asarray(shas)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "manifest.jsonl"))
    ap.add_argument("--analysis", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "analysis"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "analysis", "chord_grouping"))
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--print-charts", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    root = os.path.dirname(os.path.abspath(__file__))
    seq_root = os.path.join(root, "output", "corpus", "sequences")
    records = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    sat = [r for r in records if not r["quarantine"] and r["has_7k"] and not r["flags"]
           and any(l["table"] == "Satellite" and l["value"] is not None for l in r["labels"])]
    by_sha = {r["sha256"]: r for r in sat}
    split = json.load(open(os.path.join(args.analysis, "split.json"), encoding="utf-8"))
    charts = {k: [by_sha[s] for s in split[f"{k}_sha256"]] for k in ("train", "val", "test")}
    chart_labels = {r["sha256"]: next(l["value"] for l in r["labels"] if l["table"] == "Satellite")
                    for r in sat}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]

    # ---- 长度分布对比（train）----
    old_lens, new_lens = [], []
    for r in charts["train"]:
        seq = np.load(os.path.join(seq_root, r["sha256"] + ".npy"))
        old_lens.append(seq.shape[0])
        new_lens.append(build_timepoint_rep(seq).shape[0])
    old_lens.sort(); new_lens.sort()
    print("旧表示（note 数） vs 新表示（时间点数）：")
    for q in (50, 75, 90, 95, 99):
        print(f"  p{q}: {int(np.percentile(old_lens, q)):5d} -> {int(np.percentile(new_lens, q)):5d}")
    print(f"  max: {old_lens[-1]:5d} -> {new_lens[-1]:5d}   min: {old_lens[0]:5d} -> {new_lens[0]:5d}")
    print(f"  中位下降: {100*(1 - np.median(new_lens)/np.median(old_lens)):.0f}%")

    # ---- 真实数据打印 ----
    import random
    random.Random(11).shuffle(charts["test"])
    print("\n=== 旧 vs 新表示（真实窗口开头） ===")
    for r in charts["test"][: args.print_charts]:
        seq = np.load(os.path.join(seq_root, r["sha256"] + ".npy"))
        tp = build_timepoint_rep(seq)
        print(f"\nchart: {r['title']}  (sl={chart_labels[r['sha256']]})  "
              f"notes={seq.shape[0]} -> timepoints={tp.shape[0]}")
        print("旧表示：")
        for row in seq[:9]:
            print(f"   t={row[0]:6.3f} lane={int(row[1])} type={'normal' if row[2]==0 else 'LN'} "
                  f"dur={row[3]:.2f}")
        print("新表示（delta_t, scratch, k1..k7, has_ln）：")
        for row in tp[:6]:
            lane_str = "".join(str(i) for i, v in enumerate([row[1]] + list(row[2:9])) if v == 1)
            print(f"   dt={row[0]:6.3f} lanes={lane_str or '-'} has_ln={int(row[9])}")

    # ---- 两种表示的窗口 ----
    from .input_representation_experiment import build_raw_windows
    raw_old = {k: build_raw_windows(charts[k], seq_root, args.window, args.stride)
               for k in ("train", "val", "test")}
    raw_new = {k: build_new_windows(charts[k], seq_root, args.window, args.stride)
               for k in ("train", "val", "test")}
    for k in ("train", "val", "test"):
        print(f"{k}: old windows {len(raw_old[k][1])}  new windows {len(raw_new[k][1])}")

    # 标准化（各自 train 拟合）
    d_old = make_delta_rep(raw_old["train"][0])
    m_o, s_o = zscore_train(d_old)
    rep_old = {k: ((make_delta_rep(raw_old[k][0]) - m_o) / s_o).astype(np.float32)
               for k in ("train", "val", "test")}
    m_n, s_n = zscore_train(raw_new["train"][0])
    rep_new = {k: ((raw_new[k][0] - m_n) / s_n).astype(np.float32)
               for k in ("train", "val", "test")}

    def labels_of(raw):
        return {k: np.asarray([chart_labels[s] for s in raw[k][1]], np.float32)
                for k in ("train", "val", "test")}

    y_old, y_new = labels_of(raw_old), labels_of(raw_new)

    def run(rep, y, seq_dim, test_shas):
        mae_list, rmse_list, r2_list, bands_list = [], [], [], []
        for seed in seeds:
            model = VariantCNN(seq_dim=seq_dim, num_convs=2, kernel=5).to(device)
            pt = train_eval(model, rep["train"], y["train"], rep["val"], y["val"],
                            rep["test"], device, args, seed)
            cm, yt, yp = chart_metrics(pt, test_shas, chart_labels)
            mae_list.append(cm["mae"]); rmse_list.append(cm["rmse"])
            r2_list.append(cm["r2"]); bands_list.append(band_mae(yt, yp))
        return {
            "chart_mae": round(float(np.mean(mae_list)), 4),
            "chart_rmse": round(float(np.mean(rmse_list)), 4),
            "chart_r2": round(float(np.mean(r2_list)), 4),
            "seed_maes": [round(x, 4) for x in mae_list],
            "bands": {k: round(float(np.nanmean([b[k] for b in bands_list])), 4)
                      for k in bands_list[0]},
        }

    res_old = run(rep_old, y_old, seq_dim=4, test_shas=raw_old["test"][1])
    res_new = run(rep_new, y_new, seq_dim=NEW_DIM, test_shas=raw_new["test"][1])

    print("\n=== 结果（3 种子均值，2×k5 CNN） ===")
    print(f"旧表示（4 维 note）: chart MAE {res_old['chart_mae']}  RMSE {res_old['chart_rmse']}  "
          f"R2 {res_old['chart_r2']}")
    print(f"新表示（10 维时间点）: chart MAE {res_new['chart_mae']}  RMSE {res_new['chart_rmse']}  "
          f"R2 {res_new['chart_r2']}")
    print("bands 旧:", res_old["bands"])
    print("bands 新:", res_new["bands"])

    report = {
        "old_repr": res_old,
        "new_repr": res_new,
        "length_stats": {
            "old": {f"p{q}": int(np.percentile(old_lens, q)) for q in (50, 75, 90, 95, 99)}
                   | {"max": old_lens[-1]},
            "new": {f"p{q}": int(np.percentile(new_lens, q)) for q in (50, 75, 90, 95, 99)}
                   | {"max": new_lens[-1]},
            "median_shrink": round(100 * (1 - np.median(new_lens) / np.median(old_lens)), 1),
        },
        "new_cols": NEW_COLS,
        "window": args.window,
    }
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nsaved ->", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
