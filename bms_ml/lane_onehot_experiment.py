"""Lane one-hot 实验（只改 lane 编码，其余锁死）。

A: [log1p(delta_t) z, lane:int z, type z, duration z]        4 维
B: [log1p(delta_t) z, lane_one_hot(8, 不 z-score), type z, duration z]  11 维

同一份原始窗口；delta_t / type / duration 的归一化统计两版完全一致
（在 A 的变换矩阵上拟合，B 复用）。模型 2×Conv1d(k=5)，仅第一层 Linear 输入维变化。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from .input_representation_experiment import (
    band_mae,
    build_raw_windows,
    chart_metrics,
    make_delta_rep,
    train_eval,
    zscore_train,
)
from .receptive_field_experiment import VariantCNN


def lane_one_hot(lane_int):
    bits = [0] * 8
    bits[int(lane_int)] = 1
    return bits


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "manifest.jsonl"))
    ap.add_argument("--analysis", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "analysis"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "analysis", "lane_onehot"))
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0,1,2")
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

    raw = {k: build_raw_windows(charts[k], seq_root, args.window, args.stride)
           for k in ("train", "val", "test")}
    # 旧表示（delta_t 化）及其按列归一化
    d_old = make_delta_rep(raw["train"][0])   # [N, 512, 4]：log1p(dt), lane, type, dur
    col_mean, col_std = zscore_train(d_old)
    repA = {k: ((make_delta_rep(raw[k][0]) - col_mean) / col_std).astype(np.float32)
            for k in ("train", "val", "test")}

    # 新表示：delta_t/type/duration 复用同一统计；lane 换 one-hot（不 z-score）
    def to_onehot(windows):
        out = np.zeros((windows.shape[0], windows.shape[1], 11), dtype=np.float32)
        dt = make_delta_rep(windows)[:, :, 0]  # log1p(dt)
        out[:, :, 0] = dt
        for b in range(windows.shape[0]):
            for t in range(windows.shape[1]):
                out[b, t, 1 + int(windows[b, t, 1])] = 1.0
        out[:, :, 9] = windows[:, :, 2]   # type
        out[:, :, 10] = windows[:, :, 3]  # duration
        return out

    repB = {k: to_onehot(raw[k][0]) for k in ("train", "val", "test")}
    # 只对 delta_t/type/duration 列应用 A 的统计；one-hot 位保持 0/1
    z_map = {0: 0, 9: 2, 10: 3}  # one-hot 矩阵列 -> 旧矩阵列（统计来源）
    for k in repB:
        for j, src in z_map.items():
            repB[k][:, :, j] = (repB[k][:, :, j] - col_mean[src]) / col_std[src]

    y = {k: np.asarray([chart_labels[s] for s in raw[k][1]], np.float32)
         for k in ("train", "val", "test")}

    # ---- sanity：同一事件在两种编码下的样子 ----
    print("=== sanity: 同一 note 的两种编码（真实窗口） ===")
    seq0 = np.load(os.path.join(seq_root, raw["test"][1][0] + ".npy"))
    w0 = seq0[:4]
    old_dt = make_delta_rep(w0[None, :, :])[0]
    new0 = to_onehot(w0[None, :, :])[0]
    for i in range(min(3, w0.shape[0])):
        old = ((old_dt[i] - col_mean) / col_std)
        new = new0[i].copy()
        for j, src in z_map.items():
            new[j] = (new0[i, j] - col_mean[src]) / col_std[src]
        print(f"event {i}:")
        print(f"  旧 [dt,lane,type,dur]      : "
              f"[{old[0]:+.2f}, lane={int(w0[i,1])}, {old[2]:+.2f}, {old[3]:+.2f}]")
        oh = "".join(str(int(v)) for v in w0[i, 1] == np.arange(8))
        print(f"  新 [dt, 8×lane, type, dur] : "
              f"[{new[0]:+.2f}, {oh}, {new[9]:+.2f}, {new[10]:+.2f}]")

    # ---- 训练 A / B ----
    def run(rep, seq_dim):
        Xtr, Xva, Xte = rep["train"], rep["val"], rep["test"]
        mae_list, rmse_list, r2_list, bands_list = [], [], [], []
        t0 = time.time()
        for seed in seeds:
            model = VariantCNN(seq_dim=seq_dim, num_convs=2, kernel=5).to(device)
            pt = train_eval(model, Xtr, y["train"], Xva, y["val"], Xte, device, args, seed)
            cm, yt, yp = chart_metrics(pt, raw["test"][1], chart_labels)
            mae_list.append(cm["mae"]); rmse_list.append(cm["rmse"])
            r2_list.append(cm["r2"]); bands_list.append(band_mae(yt, yp))
        wall = time.time() - t0
        n_params = sum(p.numel() for p in VariantCNN(seq_dim=seq_dim,
                                                     num_convs=2, kernel=5).parameters())
        return {
            "params": n_params,
            "train_seconds": round(wall, 1),
            "chart_mae": round(float(np.mean(mae_list)), 4),
            "chart_rmse": round(float(np.mean(rmse_list)), 4),
            "chart_r2": round(float(np.mean(r2_list)), 4),
            "seed_maes": [round(x, 4) for x in mae_list],
            "bands": {k: round(float(np.nanmean([b[k] for b in bands_list])), 4)
                      for k in bands_list[0]},
        }

    res_a = run(repA, seq_dim=4)
    res_b = run(repB, seq_dim=11)

    print("\n=== 结果（3 种子均值，2×k5） ===")
    print(f"A int-lane : MAE {res_a['chart_mae']}  RMSE {res_a['chart_rmse']}  "
          f"R2 {res_a['chart_r2']}  params {res_a['params']}  time {res_a['train_seconds']}s")
    print(f"B one-hot  : MAE {res_b['chart_mae']}  RMSE {res_b['chart_rmse']}  "
          f"R2 {res_b['chart_r2']}  params {res_b['params']}  time {res_b['train_seconds']}s")
    print("bands A:", res_a["bands"])
    print("bands B:", res_b["bands"])

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"A_integer_lane": res_a, "B_onehot_lane": res_b,
                   "config": {"window": args.window, "seeds": seeds}},
                  f, ensure_ascii=False, indent=2)
    print("\nsaved ->", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
