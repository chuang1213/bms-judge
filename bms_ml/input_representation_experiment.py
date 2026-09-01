"""输入表示实验：absolute time vs delta_t（CNN 完全不变）。

步骤：
1. 打印真实窗口的 time/lane/delta_t/delta_lane（人可读）；
2. 变体 A：原始 [time, lane, type, duration]（沿用现有标准化）；
   变体 B：第一列换成 delta_t = 当前 time - 前一个 time（log1p + z-score），
   其余列不变；
3. 用同一个 SequenceCNN 训练 A 与 B（同 split/同 seeds），报告 chart MAE/RMSE/R²
   与各难度带 MAE；
4. sanity：辅助任务"预测窗口时长"（固定 512 event 的窗口 note 数是常数，
   所以用时长/密度作为基础局部统计的探针），检验 CNN 能否学到基本结构。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


LANE_LABELS = ["SC", "1", "2", "3", "4", "5", "6", "7"]


def build_raw_windows(records, seq_root, window_len, stride):
    win, shas = [], []
    for r in records:
        seq = np.load(os.path.join(seq_root, r["sha256"] + ".npy"))
        n = seq.shape[0]
        for start in range(0, n - window_len + 1, stride):
            win.append(seq[start:start + window_len])
            shas.append(r["sha256"])
    return np.asarray(win, dtype=np.float32), np.asarray(shas)


def make_delta_rep(windows):
    """第一列 → log1p(delta_t)；首 event 的 delta_t=0。其余列原样。"""
    out = windows.copy()
    t = out[:, :, 0]
    dt = np.zeros_like(t)
    dt[:, 1:] = t[:, 1:] - t[:, :-1]
    out[:, :, 0] = np.log1p(np.maximum(dt, 0.0))
    return out


def zscore_train(windows):
    m = windows.reshape(-1, windows.shape[-1]).mean(axis=0)
    s = windows.reshape(-1, windows.shape[-1]).std(axis=0)
    s[s == 0] = 1.0
    return m.astype(np.float32), s.astype(np.float32)


def chart_metrics(pred_w, shas, chart_labels):
    agg = defaultdict(list)
    for p, s in zip(pred_w, shas):
        agg[s].append(p)
    yt, yp = [], []
    for sha, ps in agg.items():
        yt.append(chart_labels[sha])
        yp.append(float(np.mean(ps)))
    yt, yp = np.asarray(yt), np.asarray(yp)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    r2 = 1.0 - np.sum((yt - yp) ** 2) / np.sum((yt - yt.mean()) ** 2)
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4),
            "n_charts": len(yt)}, yt, yp


def band_mae(yt, yp):
    out = {}
    for lo, hi in ((0, 2), (3, 5), (6, 8), (9, 12)):
        m = (yt >= lo) & (yt <= hi)
        out[f"sl{lo}-{hi}"] = round(float(np.mean(np.abs(yt[m] - yp[m]))), 4) if m.sum() else None
    return out


def train_eval(model, Xa, ya, Xv, yv, Xt, device, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya)),
        batch_size=args.batch_size, shuffle=True)
    best, best_state, patience = float("inf"), None, 0
    for _ in range(args.epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(Xv).to(device)).squeeze(-1).cpu().numpy()
        mae_v = float(np.mean(np.abs(yv - pv)))
        if mae_v < best - 1e-5:
            best, patience, best_state = mae_v, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pt = []
        for i in range(0, len(Xt), 512):
            b = torch.from_numpy(Xt[i:i + 512]).to(device)
            pt.append(model(b).squeeze(-1).cpu().numpy())
    return np.concatenate(pt)


class AuxModel(nn.Module):
    """辅助任务：复用 CNN 的 pool，预测标量（窗口时长）。"""

    def __init__(self, cnn, pool_dim):
        super().__init__()
        self.cnn = cnn
        self.head = nn.Linear(pool_dim, 1)

    def forward(self, x):
        return self.head(self.cnn.pool(x))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "manifest.jsonl"))
    ap.add_argument("--analysis", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "analysis"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "analysis", "rep_experiment"))
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

    raw = {k: build_raw_windows(charts[k], seq_root, args.window, args.stride)
           for k in ("train", "val", "test")}
    for k in ("train", "val", "test"):
        print(f"{k}: raw windows {raw[k][0].shape}")

    # ---- 1. 输入分析打印 ----
    random.Random(3).shuffle(charts["test"])
    print("\n=== 输入分析：真实窗口（前若干 event） ===")
    for r in charts["test"][: args.print_charts]:
        seq = np.load(os.path.join(seq_root, r["sha256"] + ".npy"))
        sl = chart_labels[r["sha256"]]
        print(f"\nchart: {r['title']}  (sl={sl})")
        print(f"{'time':>8s} {'lane':>5s} {'delta_t':>8s} {'delta_lane':>10s}  type")
        prev_t, prev_l = None, None
        for row in seq[:16]:
            t, lane, typ, dur = row
            dt = "—" if prev_t is None else f"{t - prev_t:+.3f}"
            dl = "—" if prev_l is None else f"{int(lane - prev_l):+d}"
            print(f"{t:8.3f} {LANE_LABELS[int(lane)]:>5s} {dt:>8s} {dl:>10s}  "
                  f"{'normal' if typ == 0 else 'LN'}")
            prev_t, prev_l = t, lane

    # ---- 2. 两种表示 ----
    # A：沿用现有标准化（absolute time z-score）
    prep = json.load(open(os.path.join(args.out, "..", "seq_model", "window_preprocessing.json"),
                          encoding="utf-8"))
    # 路径可能不存在于新目录，改用 analysis/seq_model
    m_a = np.asarray(prep["normalization"]["mean"], np.float32)
    s_a = np.asarray(prep["normalization"]["std"], np.float32)
    rep_a = {k: ((w - m_a) / s_a).astype(np.float32) for k, (w, _) in raw.items()}
    # B：delta_t
    raw_b = {k: make_delta_rep(w) for k, (w, _) in raw.items()}
    m_b, s_b = zscore_train(raw_b["train"])
    rep_b = {k: ((w - m_b) / s_b).astype(np.float32) for k, w in raw_b.items()}

    # 标签
    ytr = np.asarray([chart_labels[s] for s in raw["train"][1]], np.float32)
    yva = np.asarray([chart_labels[s] for s in raw["val"][1]], np.float32)
    yte = np.asarray([chart_labels[s] for s in raw["test"][1]], np.float32)
    # 辅助任务目标：窗口时长（原始秒数）
    dur_tr = raw["train"][0][:, -1, 0] - raw["train"][0][:, 0, 0]
    dur_va = raw["val"][0][:, -1, 0] - raw["val"][0][:, 0, 0]
    dur_te = raw["test"][0][:, -1, 0] - raw["test"][0][:, 0, 0]

    from .sequence_cnn import SequenceCNN

    def pool_dim_of():
        m = SequenceCNN()
        with torch.no_grad():
            return m.pool(torch.zeros(2, args.window, 4)).shape[-1]

    pdim = pool_dim_of()
    report = {"config": {"window": args.window, "seeds": seeds}, "repA": {}, "repB": {}}

    for name, rep in (("repA", rep_a), ("repB", rep_b)):
        Xtr, Xva, Xte = rep["train"], rep["val"], rep["test"]
        # 难度任务
        mae_list, r2_list, bands_list = [], [], []
        win_mae_list = []
        for seed in seeds:
            cnn = SequenceCNN().to(device)
            pt = train_eval(cnn, Xtr, ytr, Xva, yva, Xte, device, args, seed)
            cm, yt_c, yp_c = chart_metrics(pt, raw["test"][1], chart_labels)
            mae_list.append(cm["mae"])
            r2_list.append(cm["r2"])
            bands_list.append(band_mae(yt_c, yp_c))
            win_mae_list.append(float(np.mean(np.abs(yte - pt))))
        # 辅助任务
        aux_mae, aux_r2 = [], []
        for seed in seeds[:2]:
            cnn = SequenceCNN().to(device)
            aux = AuxModel(cnn, pdim).to(device)
            pa = train_eval(aux, Xtr, dur_tr, Xva, dur_va, Xte, device, args, seed)
            aux_mae.append(float(np.mean(np.abs(dur_te - pa))))
            ss = np.sum((dur_te - pa) ** 2)
            st = np.sum((dur_te - dur_te.mean()) ** 2)
            aux_r2.append(float(1 - ss / st))
        report[name] = {
            "difficulty_chart_mae": round(float(np.mean(mae_list)), 4),
            "difficulty_chart_r2": round(float(np.mean(r2_list)), 4),
            "difficulty_window_mae": round(float(np.mean(win_mae_list)), 4),
            "seed_maes": [round(x, 4) for x in mae_list],
            "bands": {k: round(float(np.nanmean([b[k] for b in bands_list])), 4)
                      for k in bands_list[0]},
            "aux_duration_mae": round(float(np.mean(aux_mae)), 4),
            "aux_duration_r2": round(float(np.mean(aux_r2)), 4),
            "duration_stats": {"mean": round(float(dur_te.mean()), 2),
                               "std": round(float(dur_te.std()), 2)},
        }
        print(f"\n{name}: difficulty chart MAE {report[name]['difficulty_chart_mae']}  "
              f"R2 {report[name]['difficulty_chart_r2']}  "
              f"window MAE {report[name]['difficulty_window_mae']}  "
              f"aux duration MAE {report[name]['aux_duration_mae']}  "
              f"R2 {report[name]['aux_duration_r2']}")
        print("  bands:", report[name]["bands"])

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nsaved ->", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
