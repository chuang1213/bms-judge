"""Sequence CNN 训练框架（模型类由你实现：sequence_cnn.py）。

实验（同一 song-level split，同 feature_analysis）：
  A. mean predictor（chart-level）
  B. 26-feature MLP（chart-level）
  C. Sequence CNN（窗口级预测 → 同一谱面窗口预测取均值 → chart-level）
  D. 26 features + Sequence CNN（CNN 的 pool 向量与 26 特征拼接）

用法:
  先完成 sequence_cnn.py 的 SequenceCNN，然后：
  python -m bms_ml.train_sequence_cnn --manifest ... --analysis ...
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

from .features import FEATURE_NAMES
from .model import MLPBaseline
from .sequence_dataset import build_all


class FusedModel(nn.Module):
    """D：CNN pool 特征 + 26 统计特征 → Linear。"""

    def __init__(self, seq_model, pool_dim: int, stats_dim: int):
        super().__init__()
        self.seq = seq_model
        self.head = nn.Linear(pool_dim + stats_dim, 1)

    def forward(self, x, stats):
        pooled = self.seq.pool(x)            # [B, pool_dim]
        return self.head(torch.cat([pooled, stats], dim=1))


def chart_metrics(pred_windows: np.ndarray, labels_windows: np.ndarray,
                  shas: np.ndarray, chart_labels: dict):
    """窗口预测 → 按谱面取均值 → chart-level 指标。"""
    agg = defaultdict(list)
    for p, s in zip(pred_windows, shas):
        agg[s].append(p)
    yt, yp = [], []
    for sha, ps in agg.items():
        yt.append(chart_labels[sha])
        yp.append(float(np.mean(ps)))
    yt = np.asarray(yt)
    yp = np.asarray(yp)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4),
            "n_charts": len(yt)}, yt, yp


def band_mae(yt, yp, bands=((0, 2), (3, 5), (6, 8), (9, 12))):
    out = {}
    for lo, hi in bands:
        m = (yt >= lo) & (yt <= hi)
        out[f"sl{lo}-{hi}"] = round(float(np.mean(np.abs(yt[m] - yp[m]))), 4) if m.sum() else None
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "manifest.jsonl"))
    ap.add_argument("--analysis", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "analysis"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "analysis", "seq_model"))
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
    sat = [
        r for r in records
        if not r["quarantine"] and r["has_7k"] and not r["flags"]
        and any(l["table"] == "Satellite" and l["value"] is not None for l in r["labels"])
    ]
    by_sha = {r["sha256"]: r for r in sat}
    split = json.load(open(os.path.join(args.analysis, "split.json"), encoding="utf-8"))
    norm = json.load(open(os.path.join(args.analysis, "normalization.json"), encoding="utf-8"))
    mean26 = np.asarray(norm["mean"], np.float32)
    std26 = np.asarray(norm["std"], np.float32)

    def label_of(r):
        return next(l["value"] for l in r["labels"] if l["table"] == "Satellite")

    charts_by_split = {
        k: [by_sha[s] for s in split[f"{k}_sha256"]] for k in ("train", "val", "test")
    }
    chart_labels = {r["sha256"]: label_of(r) for r in sat}

    # ---- 窗口 ----
    ws = build_all(
        charts_by_split, seq_root, args.window, args.stride, label_of,
        os.path.join(args.out, "window_preprocessing.json"))
    for k in ("train", "val", "test"):
        win, shas, labels = ws[k]
        print(f"{k}: windows={len(shas)}  shape={win.shape}")

    Xtr_w, _, ytr_w = ws["train"]
    Xva_w, _, yva_w = ws["val"]
    Xte_w, te_shas, yte_w = ws["test"]

    # 26 特征（每 chart 一份，窗口级复制）
    def stats_for(rows):
        X = np.stack([np.asarray(r["features"], dtype=np.float32) for r in rows])
        return (X - mean26) / std26

    Xtr_s, Xva_s, Xte_s = (stats_for(charts_by_split[k]) for k in ("train", "val", "test"))
    tr_sha = np.asarray([r["sha256"] for r in charts_by_split["train"]])
    va_sha = np.asarray([r["sha256"] for r in charts_by_split["val"]])
    te_sha = np.asarray([r["sha256"] for r in charts_by_split["test"]])

    def stats_for_windows(w_sha, Xs, s_sha):
        m = {s: i for i, s in enumerate(s_sha)}
        return Xs[[m[s] for s in w_sha]]

    Xtr_s_w = stats_for_windows(ws["train"][1], Xtr_s, tr_sha)
    Xva_s_w = stats_for_windows(ws["val"][1], Xva_s, va_sha)
    Xte_s_w = stats_for_windows(ws["test"][1], Xte_s, te_sha)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]

    # ---- A. mean predictor ----
    tr_mean = float(np.mean([label_of(r) for r in charts_by_split["train"]]))
    yt = np.asarray([chart_labels[s] for s in te_sha])
    a_mae = float(np.mean(np.abs(yt - tr_mean)))
    a_rmse = float(np.sqrt(np.mean((yt - tr_mean) ** 2)))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    a_r2 = 1.0 - np.sum((yt - tr_mean) ** 2) / ss_tot if ss_tot > 0 else float("nan")
    print(f"\nA mean predictor: chart MAE {a_mae:.4f}  RMSE {a_rmse:.4f}  R2 {a_r2:.4f}")

    def train_loop(model, Xa, ya, Xv, yv, seed, return_history=False):
        torch.manual_seed(seed)
        np.random.seed(seed)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.SmoothL1Loss()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya)),
            batch_size=args.batch_size, shuffle=True)
        best, best_state, patience = float("inf"), None, 0
        history = []
        for _ in range(args.epochs):
            model.train()
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                out = model(xb).squeeze(-1)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pv = model(torch.from_numpy(Xv).to(device)).squeeze(-1).cpu().numpy()
            mae_v = float(np.mean(np.abs(yv - pv)))
            history.append(mae_v)
            if mae_v < best - 1e-5:
                best, patience = mae_v, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= args.patience:
                    break
        model.load_state_dict(best_state)
        return (model, history) if return_history else model

    def predict(model, X):
        model.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(X), 512):
                b = torch.from_numpy(X[i:i + 512]).to(device)
                out.append(model(b).squeeze(-1).cpu().numpy())
        return np.concatenate(out)

    # ---- B. 26-feature MLP ----
    print("\nB: training 26-feature MLP ...")
    b_results = []
    for seed in seeds:
        model = MLPBaseline(input_dim=26).to(device)
        train_loop(model, Xtr_s, np.asarray([label_of(r) for r in charts_by_split["train"]]),
                   Xva_s, np.asarray([label_of(r) for r in charts_by_split["val"]]), seed)
        pt = predict(model, Xte_s)
        mae = float(np.mean(np.abs(yt - pt)))
        rmse = float(np.sqrt(np.mean((yt - pt) ** 2)))
        r2 = 1.0 - np.sum((yt - pt) ** 2) / ss_tot if ss_tot > 0 else float("nan")
        b_results.append({"mae": mae, "rmse": rmse, "r2": r2})
    b_mae = float(np.mean([m["mae"] for m in b_results]))
    print(f"B: chart MAE {b_mae:.4f} (seeds {b_results})")

    # ---- B_window 对照：26 特征 MLP 在同样的窗口标签上 ----
    print("\nB_window: 26-MLP on window labels ...")
    bw_results = []
    for seed in seeds:
        model = MLPBaseline(input_dim=26).to(device)
        train_loop(model, Xtr_s_w, ytr_w, Xva_s_w, yva_w, seed)
        pt_w = predict(model, Xte_s_w)
        m_w, yt_w, yp_w = chart_metrics(pt_w, yte_w, te_shas, chart_labels)
        bw_results.append({
            "window_mae": round(float(np.mean(np.abs(yte_w - pt_w))), 4),
            **{k: v for k, v in m_w.items()},
        })
    bw_avg = {k: round(float(np.mean([r[k] for r in bw_results])), 4)
              for k in ("window_mae", "mae", "rmse", "r2")}
    print(f"B_window: window MAE {bw_avg['window_mae']}  chart MAE {bw_avg['mae']}  "
          f"R2 {bw_avg['r2']}")

    # ---- C / D：需要用户实现的 SequenceCNN ----
    try:
        from .sequence_cnn import SequenceCNN
        _ = SequenceCNN()
    except NotImplementedError:
        print("\nSequenceCNN 尚未实现。请先完成 sequence_cnn.py 的 "
              "class SequenceCNN 与 forward()/pool()，再运行本脚本。")
        return
    except Exception as e:
        print(f"SequenceCNN 导入/初始化失败: {e}")
        return

    def train_window_model(model, Xa, ya, Xv, yv, seed, return_history=False):
        return train_loop(model, Xa, ya, Xv, yv, seed, return_history=return_history)

    def make_xy(Xs, ys, extra=None):
        if extra is None:
            return Xs, ys
        return np.hstack([Xs, extra]), ys

    print("\nC/D: training Sequence CNN ...")
    c_metrics, d_metrics = [], []
    c_band_list, d_band_list = [], []
    c_window_mae_list = []
    c_history = None
    for seed in seeds:
        # C
        cnn = SequenceCNN().to(device)
        cnn, hist = train_window_model(cnn, Xtr_w, ytr_w, Xva_w, yva_w, seed, return_history=True)
        if seed == seeds[0]:
            c_history = hist
            torch.save(cnn.state_dict(), os.path.join(args.out, "cnn_checkpoint.pt"))
        pt_w = predict(cnn, Xte_w)
        print(f"  C seed {seed}: pred min/max/mean/std = "
              f"{pt_w.min():.2f}/{pt_w.max():.2f}/{pt_w.mean():.2f}/{pt_w.std():.2f}")
        cm, yt_c, yp_c = chart_metrics(pt_w, yte_w, te_shas, chart_labels)
        c_metrics.append(cm)
        c_band_list.append(band_mae(yt_c, yp_c))
        c_window_mae_list.append(float(np.mean(np.abs(yte_w - pt_w))))
        # D
        with torch.no_grad():
            pool_dim = cnn.pool(torch.zeros(2, args.window, 4).to(device)).shape[-1]
        fused = FusedModel(cnn, pool_dim=pool_dim, stats_dim=26).to(device)
        fused_opt = torch.optim.Adam(fused.parameters(), lr=args.lr)
        fused_loss = nn.SmoothL1Loss()
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(Xtr_w), torch.from_numpy(Xtr_s_w), torch.from_numpy(ytr_w)),
            batch_size=args.batch_size, shuffle=True)
        best, best_state, patience = float("inf"), None, 0
        for _ in range(args.epochs):
            fused.train()
            for xb, sb, yb in dl:
                xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
                fused_opt.zero_grad()
                loss = fused_loss(fused(xb, sb).squeeze(-1), yb)
                loss.backward()
                fused_opt.step()
            fused.eval()
            with torch.no_grad():
                pv = []
                for i in range(0, len(Xva_w), 512):
                    b = torch.from_numpy(Xva_w[i:i + 512]).to(device)
                    s = torch.from_numpy(Xva_s_w[i:i + 512]).to(device)
                    pv.append(fused(b, s).squeeze(-1).cpu().numpy())
            mae_v = float(np.mean(np.abs(yva_w - np.concatenate(pv))))
            if mae_v < best - 1e-5:
                best, patience = mae_v, 0
                best_state = {k: v.clone() for k, v in fused.state_dict().items()}
            else:
                patience += 1
                if patience >= args.patience:
                    break
        fused.load_state_dict(best_state)
        fused.eval()
        pt_fw = []
        with torch.no_grad():
            for i in range(0, len(Xte_w), 512):
                b = torch.from_numpy(Xte_w[i:i + 512]).to(device)
                s = torch.from_numpy(Xte_s_w[i:i + 512]).to(device)
                pt_fw.append(fused(b, s).squeeze(-1).cpu().numpy())
        pt_fw = np.concatenate(pt_fw)
        dm, yt_d, yp_d = chart_metrics(pt_fw, yte_w, te_shas, chart_labels)
        d_metrics.append(dm)
        d_band_list.append(band_mae(yt_d, yp_d))

    def avg(metrics_list):
        return {k: round(float(np.mean([m[k] for m in metrics_list])), 4)
                for k in ("mae", "rmse", "r2")}

    c_avg = avg(c_metrics)
    d_avg = avg(d_metrics)
    print(f"C (sequence CNN): chart MAE {c_avg['mae']}  RMSE {c_avg['rmse']}  R2 {c_avg['r2']}  "
          f"window-level MAE {float(np.mean(c_window_mae_list)):.4f}")
    print(f"D (26 + seq):     chart MAE {d_avg['mae']}  RMSE {d_avg['rmse']}  R2 {d_avg['r2']}")

    # 参数量
    cnn = SequenceCNN()
    n_cnn = sum(p.numel() for p in cnn.parameters())
    n_b = sum(p.numel() for p in MLPBaseline(26).parameters())
    print(f"\nparams: B MLP={n_b}  C CNN={n_cnn}")

    def band_avg(band_list):
        keys = list(band_list[0])
        return {k: round(float(np.nanmean([b[k] for b in band_list])), 4) for k in keys}

    report = {
        "config": {"window": args.window, "stride": args.stride,
                   "epochs": args.epochs, "patience": args.patience,
                   "batch_size": args.batch_size, "lr": args.lr, "seeds": seeds},
        "A_mean": {"mae": round(a_mae, 4), "rmse": round(a_rmse, 4), "r2": round(a_r2, 4)},
        "B_mlp26": {k: round(float(np.mean([m[k] for m in b_results])), 4)
                    for k in ("mae", "rmse", "r2")},
        "C_cnn": c_avg,
        "C_window_mae": round(float(np.mean(c_window_mae_list)), 4),
        "B_window_control": bw_avg,
        "C_val_window_mae_history": [round(float(v), 3) for v in c_history],
        "D_fused": d_avg,
        "params": {"mlp26": n_b, "cnn": n_cnn},
    }
    # B 的分带
    b_bands = []
    for seed in seeds:
        model = MLPBaseline(26).to(device)
        train_loop(model, Xtr_s, np.asarray([label_of(r) for r in charts_by_split["train"]]),
                   Xva_s, np.asarray([label_of(r) for r in charts_by_split["val"]]), seed)
        pt = predict(model, Xte_s)
        b_bands.append(band_mae(yt, pt))
    report["bands_B"] = band_avg(b_bands)
    report["bands_C"] = band_avg(c_band_list)
    report["bands_D"] = band_avg(d_band_list)
    report["B_seed_maes"] = b_results
    report["C_seed_maes"] = [m["mae"] for m in c_metrics]
    report["D_seed_maes"] = [m["mae"] for m in d_metrics]

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nresults saved ->", os.path.join(args.out, "results.json"))
    print("bands_B:", report["bands_B"])
    print("bands_C:", report["bands_C"])
    print("bands_D:", report["bands_D"])


if __name__ == "__main__":
    main()
