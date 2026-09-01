"""感受野实验：固定 delta_t 表示，只改变 CNN 深度/核宽。

A = 2 x Conv1d(kernel=3)  （= 上一轮 baseline）
B = 4 x Conv1d(kernel=3)  （更深）
C = 2 x Conv1d(kernel=5)  （更宽）

其余条件完全一致：delta_t 表示、Satellite、同 song split、同训练流程/种子。
报告 chart MAE/RMSE/R²、分带 MAE、参数量、训练时间、GPU 显存、理论感受野。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from .input_representation_experiment import (
    band_mae,
    build_raw_windows,
    chart_metrics,
    make_delta_rep,
    train_eval,
    zscore_train,
)


class VariantCNN(nn.Module):
    """与 SequenceCNN 同构的参数化变体（仅 num_convs / kernel 不同）。"""

    def __init__(self, seq_dim=4, embed_dim=32, conv_channels=64,
                 num_convs=2, kernel=3):
        super().__init__()
        self.embed = nn.Linear(seq_dim, embed_dim)
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_ch = embed_dim if i == 0 else conv_channels
            self.convs.append(
                nn.Conv1d(in_ch, conv_channels, kernel, padding=kernel // 2))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(conv_channels, 1)

    def pool(self, x):
        x = self.embed(x).transpose(1, 2)
        for conv in self.convs:
            x = self.relu(conv(x))
        return x.mean(dim=2)

    def forward(self, x):
        return self.fc(self.pool(x))


def receptive_field(num_convs, kernel):
    return 1 + num_convs * (kernel - 1)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "manifest.jsonl"))
    ap.add_argument("--analysis", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "analysis"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "analysis", "rf_experiment"))
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
    raw_b = {k: make_delta_rep(w) for k, (w, _) in raw.items()}
    m_b, s_b = zscore_train(raw_b["train"])
    rep = {k: ((w - m_b) / s_b).astype(np.float32) for k, w in raw_b.items()}
    ytr = np.asarray([chart_labels[s] for s in raw["train"][1]], np.float32)
    yva = np.asarray([chart_labels[s] for s in raw["val"][1]], np.float32)
    yte = np.asarray([chart_labels[s] for s in raw["test"][1]], np.float32)
    Xtr, Xva, Xte = rep["train"], rep["val"], rep["test"]

    models = {
        "A_2xk3": dict(num_convs=2, kernel=3),
        "B_4xk3": dict(num_convs=4, kernel=3),
        "C_2xk5": dict(num_convs=2, kernel=5),
    }
    report = {"config": {"window": args.window, "seeds": seeds}, "models": {}}
    print(f"\n理论感受野（每层 padding 保持长度，RF = 1 + n*(k-1)）:")
    for name, cfg in models.items():
        rf = receptive_field(cfg["num_convs"], cfg["kernel"])
        print(f"  {name}: RF = {rf} events")
        report["models"][name] = {"receptive_field": rf}

    for name, cfg in models.items():
        model = VariantCNN(**cfg)
        n_params = sum(p.numel() for p in model.parameters())
        t0 = time.time()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        mae_list, r2_list, rmse_list, bands_list, win_mae = [], [], [], [], []
        for seed in seeds:
            cnn = VariantCNN(**cfg).to(device)
            pt = train_eval(cnn, Xtr, ytr, Xva, yva, Xte, device, args, seed)
            cm, yt_c, yp_c = chart_metrics(pt, raw["test"][1], chart_labels)
            mae_list.append(cm["mae"]); rmse_list.append(cm["rmse"])
            r2_list.append(cm["r2"]); bands_list.append(band_mae(yt_c, yp_c))
            win_mae.append(float(np.mean(np.abs(yte - pt))))
        wall = time.time() - t0
        peak_mem = (torch.cuda.max_memory_allocated() / 1e6) if device == "cuda" else None
        report["models"][name].update({
            "params": n_params,
            "train_seconds": round(wall, 1),
            "peak_gpu_mb": round(peak_mem, 1) if peak_mem else None,
            "chart_mae": round(float(np.mean(mae_list)), 4),
            "chart_rmse": round(float(np.mean(rmse_list)), 4),
            "chart_r2": round(float(np.mean(r2_list)), 4),
            "window_mae": round(float(np.mean(win_mae)), 4),
            "seed_maes": [round(x, 4) for x in mae_list],
            "bands": {k: round(float(np.nanmean([b[k] for b in bands_list])), 4)
                      for k in bands_list[0]},
        })
        r = report["models"][name]
        print(f"\n{name}: params={n_params}  RF={r['receptive_field']}  "
              f"time={r['train_seconds']}s  gpu={r['peak_gpu_mb']}MB")
        print(f"  chart MAE {r['chart_mae']}  RMSE {r['chart_rmse']}  R2 {r['chart_r2']}  "
              f"window MAE {r['window_mae']}")
        print("  bands:", r["bands"])

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nsaved ->", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
