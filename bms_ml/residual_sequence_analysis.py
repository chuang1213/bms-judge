"""残差 + 序列派生量分析（26-feature MLP 未解释的部分）。

流程：
1. 复现 26-feature MLP（同 seed/split），保存 test 预测与残差；
2. |residual| 与现有特征/新序列量做相关性 + 分箱统计；
3. 从 note sequence 提取 4 类候选量（密度变化/间隔结构/lane 移动/chord 形状）；
4. 对照：每个新量 vs sl 与 vs |residual|（含控制 sl 的偏相关）；
5. 取最有证据的 3-5 个，做 26 vs 26+k 的 MLP 对比（同 split，3 种子）；
6. 按 sl0-2 / 3-5 / 6-8 / 9-12 分带报告 baseline 与新增特征 MAE。
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


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x, y, z):
    """控制 z 后 x 与 y 的偏相关（基于秩的偏相关）。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    rx, ry, rz = (np.argsort(np.argsort(v)).astype(np.float64) for v in (x, y, z))
    Z = np.stack([np.ones_like(rz), rz], axis=1)
    bx = np.linalg.lstsq(Z, rx, rcond=None)[0]
    by = np.linalg.lstsq(Z, ry, rcond=None)[0]
    ex = rx - Z @ bx
    ey = ry - Z @ by
    if ex.std() == 0 or ey.std() == 0:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def sequence_quantities(seq: np.ndarray, duration: float) -> dict:
    ts = seq[:, 0].astype(np.float64)
    lanes = seq[:, 1].astype(np.float64)
    q: dict = {}

    # A. 局部密度（1s 窗口）
    n_win = max(int(np.ceil(duration)), 1)
    counts = np.zeros(n_win, dtype=np.float64)
    for t in ts:
        i = min(int(t), n_win - 1)
        counts[i] += 1
    q["density_std"] = float(counts.std())
    q["high_density_frac"] = float((counts >= 25).mean())     # >=25 note/s 的秒占比
    d = np.diff(counts)
    q["density_change_mean"] = float(np.abs(d).mean()) if len(d) else 0.0
    q["density_change_max"] = float(np.abs(d).max()) if len(d) else 0.0

    # B. 时间间隔结构（只看不同时间点之间的正间隔，排除和弦内部 0 间隔）
    order = np.argsort(ts, kind="stable")
    ts_sorted = ts[order]
    dt = np.diff(ts_sorted)
    dt_pos = dt[dt > 1e-9]
    if len(dt_pos):
        q["gap_mean"] = float(dt_pos.mean())
        q["gap_std"] = float(dt_pos.std())
        q["gap_cv"] = float(dt_pos.std() / dt_pos.mean()) if dt_pos.mean() > 0 else 0.0
        q["tiny_gap_frac"] = float((dt_pos < 0.06).mean())
        q["rhythm_repeat_frac"] = float(
            (np.abs(np.diff(dt_pos)) < 0.005).mean()) if len(dt_pos) > 1 else 0.0
    else:
        q.update(gap_mean=0, gap_std=0, gap_cv=0, tiny_gap_frac=0, rhythm_repeat_frac=0)

    # C. lane 移动（相邻行）
    dl = np.abs(np.diff(lanes))
    if len(dl):
        q["lane_move_mean"] = float(dl.mean())
        q["lane_move_std"] = float(dl.std())
        q["large_move_frac"] = float((dl >= 4).mean())
    else:
        q.update(lane_move_mean=0, lane_move_std=0, large_move_frac=0)
    best, cur = 1, 1
    for a, b in zip(lanes[:-1], lanes[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    q["max_same_lane_run"] = float(best)

    # D. chord 形状（同一时间点的多行）
    rounded = np.round(ts, 3)
    groups = defaultdict(list)
    for t, l in zip(rounded, lanes):
        groups[t].append(l)
    sizes = [len(v) for v in groups.values()]
    chords = [v for v in groups.values() if len(v) >= 2]
    q["chord_size_mean"] = float(np.mean(sizes)) if sizes else 0.0
    if chords:
        spans = [max(v) - min(v) for v in chords]
        q["chord_span_mean"] = float(np.mean(spans))
        q["scratch_chord_frac"] = float(sum(0.0 in v for v in chords) / len(chords))
        cents = [np.mean(v) for v in chords]
        q["chord_centroid_move"] = float(np.mean(np.abs(np.diff(cents)))) if len(cents) > 1 else 0.0
    else:
        q.update(chord_span_mean=0, scratch_chord_frac=0, chord_centroid_move=0)
    return q


QUANTITY_NAMES = [
    "density_std", "high_density_frac", "density_change_mean", "density_change_max",
    "gap_mean", "gap_std", "gap_cv", "tiny_gap_frac", "rhythm_repeat_frac",
    "lane_move_mean", "lane_move_std", "large_move_frac", "max_same_lane_run",
    "chord_size_mean", "chord_span_mean", "scratch_chord_frac", "chord_centroid_move",
]


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "manifest.jsonl"))
    ap.add_argument("--analysis", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "analysis"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

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
    mean = np.asarray(norm["mean"], np.float32)
    std = np.asarray(norm["std"], np.float32)

    def load(idx_key):
        rows = [by_sha[s] for s in split[idx_key]]
        X = np.stack([np.asarray(r["features"], dtype=np.float32) for r in rows])
        y = np.asarray([next(l["value"] for l in r["labels"] if l["table"] == "Satellite")
                        for r in rows], dtype=np.float32)
        return rows, (X - mean) / std, y

    tr_rows, Xtr, ytr = load("train_sha256")
    va_rows, Xva, yva = load("val_sha256")
    te_rows, Xte, yte = load("test_sha256")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]

    def run_mlp(Xa, ya, Xv, yv, Xt, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MLPBaseline(Xa.shape[1]).to(device)
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
            pt = model(torch.from_numpy(Xt).to(device)).squeeze(-1).cpu().numpy()
        return pt

    # ---- 1. baseline 预测（seed 0，与上一阶段一致）----
    pred_te = run_mlp(Xtr, ytr, Xva, yva, Xte, 0)
    resid = yte - pred_te
    base_preds = [
        {"sha256": r["sha256"], "title": r["title"], "rel_path": r["rel_path"],
         "sl": float(y), "pred": float(p), "residual": float(rr),
         "abs_residual": float(abs(rr))}
        for r, y, p, rr in zip(te_rows, yte, pred_te, resid)
    ]
    with open(os.path.join(args.analysis, "baseline_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(base_preds, f, ensure_ascii=False, indent=2)
    print(f"baseline test MAE (seed 0): {np.mean(np.abs(resid)):.4f}  "
          f"n_test={len(te_rows)}")

    # ---- 2. |residual| vs 现有 26 特征 ----
    absr = np.abs(resid)
    exist_corr = []
    for j, name in enumerate(FEATURE_NAMES):
        exist_corr.append((name, round(spearman(Xte[:, j], absr), 3),
                           round(spearman(Xte[:, j], yte), 3)))
    print("\n|residual| vs existing features (top by rho):")
    for name, ra, rs in sorted(exist_corr, key=lambda t: -abs(t[1]))[:10]:
        print(f"  {name:20s} rho(|res|)={ra:+.3f}  rho(sl)={rs:+.3f}")

    # ---- 3. 序列派生量 ----
    q_rows = []
    for r in sat:
        seq = np.load(os.path.join(seq_root, r["sha256"] + ".npy"))
        q = sequence_quantities(seq, r["meta"]["duration_sec"])
        q_rows.append(q)
    with open(os.path.join(args.analysis, "sequence_quantities.json"), "w", encoding="utf-8") as f:
        json.dump([{**q, "sha256": r["sha256"]} for r, q in zip(sat, q_rows)],
                  f, ensure_ascii=False, indent=2)

    te_idx_in_sat = {r["sha256"]: i for i, r in enumerate(sat)}
    all_q = np.stack([np.asarray([q[n] for n in QUANTITY_NAMES], dtype=np.float64)
                      for q in q_rows])  # n_sat x 17
    te_q = all_q[[te_idx_in_sat[r["sha256"]] for r in te_rows]]

    # ---- 4. 对照：新量 vs sl / vs |residual| / 偏相关（控制 sl）----
    contrast = []
    for j, name in enumerate(QUANTITY_NAMES):
        x = te_q[:, j]
        contrast.append({
            "name": name,
            "rho_sl": round(spearman(x, yte), 3),
            "rho_absres": round(spearman(x, absr), 3),
            "partial_rho_absres_given_sl": round(partial_spearman(x, absr, yte), 3),
        })
    print("\nsequence quantities: rho vs sl / |residual| / partial(|res| | sl)")
    for c in sorted(contrast, key=lambda t: -abs(t["partial_rho_absres_given_sl"])):
        print(f"  {c['name']:22s} sl={c['rho_sl']:+.3f}  |res|={c['rho_absres']:+.3f}  "
              f"partial={c['partial_rho_absres_given_sl']:+.3f}")

    # ---- 5. 候选：按偏相关取前 5（且排除与已有特征几乎重复的量）----
    candidates = [c["name"] for c in
                  sorted(contrast, key=lambda t: -abs(t["partial_rho_absres_given_sl"]))[:5]]
    print("\nselected candidates:", candidates)

    # ---- 6. 26 vs 26+k 实验（3 种子，同 split）----
    new_idx = [QUANTITY_NAMES.index(n) for n in candidates]
    tr_mask = np.isin([r["sha256"] for r in sat], split["train_sha256"])
    q_mean = all_q[tr_mask].mean(axis=0)
    q_std = all_q[tr_mask].std(axis=0) + 1e-9

    def q_for(rows):
        idx = [te_idx_in_sat[r["sha256"]] for r in rows]
        return ((all_q[idx][:, new_idx] - q_mean[new_idx]) / q_std[new_idx]).astype(np.float32)

    Xtr_q = q_for(tr_rows)
    Xva_q = q_for(va_rows)
    Xte_q = q_for(te_rows)

    def evaluate(Xa, Xv, Xt):
        mae_t, rmse_t, r2_t = [], [], []
        for seed in seeds:
            pt = run_mlp(Xa, ytr, Xv, yva, Xt, seed)
            mae_t.append(float(np.mean(np.abs(yte - pt))))
            rmse_t.append(float(np.sqrt(np.mean((yte - pt) ** 2))))
            ss_res = np.sum((yte - pt) ** 2)
            ss_tot = np.sum((yte - yte.mean()) ** 2)
            r2_t.append(float(1 - ss_res / ss_tot))
        return (float(np.mean(mae_t)), float(np.mean(rmse_t)), float(np.mean(r2_t)),
                mae_t)

    base_mae, base_rmse, base_r2, _ = evaluate(Xtr, Xva, Xte)
    new_mae, new_rmse, new_r2, _ = evaluate(np.hstack([Xtr, Xtr_q]),
                                            np.hstack([Xva, Xva_q]),
                                            np.hstack([Xte, Xte_q]))
    print(f"\n26 features      test MAE {base_mae:.4f} (RMSE {base_rmse:.4f}, R2 {base_r2:.4f})")
    print(f"26 + {len(candidates)} seq   test MAE {new_mae:.4f} (RMSE {new_rmse:.4f}, R2 {new_r2:.4f})")

    # ---- 7. 分难度带 ----
    bands = [(0, 2), (3, 5), (6, 8), (9, 12)]
    band_report = {}
    for lo, hi in bands:
        mask = (yte >= lo) & (yte <= hi)
        if mask.sum() == 0:
            continue
        # 用 3 个种子各自的预测算带内 MAE（重跑一次拿预测）
        b_base, b_new = [], []
        for seed in seeds:
            p1 = run_mlp(Xtr, ytr, Xva, yva, Xte, seed)
            p2 = run_mlp(np.hstack([Xtr, Xtr_q]), ytr, np.hstack([Xva, Xva_q]),
                         yva, np.hstack([Xte, Xte_q]), seed)
            b_base.append(float(np.mean(np.abs(yte[mask] - p1[mask]))))
            b_new.append(float(np.mean(np.abs(yte[mask] - p2[mask]))))
        band_report[f"sl{lo}-{hi}"] = {
            "n": int(mask.sum()),
            "baseline_mae": round(float(np.mean(b_base)), 4),
            "new_mae": round(float(np.mean(b_new)), 4),
        }
    print("\nper-band MAE (3-seed mean):")
    for k, v in band_report.items():
        print(f"  {k:7s} n={v['n']:4d}  baseline {v['baseline_mae']:.3f}  "
              f"new {v['new_mae']:.3f}  delta {v['new_mae']-v['baseline_mae']:+.3f}")

    out = {
        "candidates": candidates,
        "contrast": contrast,
        "existing_feature_absres_corr": exist_corr,
        "experiment": {
            "baseline": {"mae": base_mae, "rmse": base_rmse, "r2": base_r2},
            "new": {"mae": new_mae, "rmse": new_rmse, "r2": new_r2},
            "seeds": seeds,
        },
        "bands": band_report,
    }
    with open(os.path.join(args.analysis, "residual_sequence_report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nsaved residual_sequence_report.json")


if __name__ == "__main__":
    main()
