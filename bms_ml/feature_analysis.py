"""Satellite 特征分析：26 维结构化特征能解释 sl0-sl12 到什么程度。

严格 song-level split（70/15/15，同曲差分同侧）。
Baseline：mean predictor / 单变量分析 / 线性回归 / 简单 MLP。
附带：残差分析、特征分组消融、图表、可复现产物。

用法: python -m bms_ml.feature_analysis --manifest output/corpus/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .features import FEATURE_NAMES
from .model import MLPBaseline
from .split import song_group_key


# 特征分组（消融用）
FEATURE_GROUPS = {
    "A_basic_scale": ["total_notes", "duration_sec", "measures", "avg_nps"],
    "B_bpm_timing": ["initial_bpm", "min_bpm", "max_bpm", "bpm_change_count",
                     "stop_count", "stop_total_sec"],
    "C_chord_density": ["peak_nps_1s", "peak_measure_nps", "chord_count",
                        "chord2_count", "chord3plus_count", "jack_count"],
    "D_lane_distribution": ["lane0_scratch", "lane1", "lane2", "lane3", "lane4",
                            "lane5", "lane6", "lane7", "scratch_ratio"],
    "E_ln": ["ln_ratio"],
}


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return pearson(rx, ry)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "manifest.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "analysis"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ---- 数据装载 + 过滤 ----
    records = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    sat = [
        r for r in records
        if not r["quarantine"] and r["has_7k"]
        and any(l["table"] == "Satellite" and l["value"] is not None for l in r["labels"])
    ]
    flagged = [r for r in sat if r["flags"]]
    sat = [r for r in sat if not r["flags"]]
    print(f"Satellite labeled 7K: {len(sat) + len(flagged)} (excluded flagged: {len(flagged)})")

    X = np.stack([np.asarray(r["features"], dtype=np.float32) for r in sat])
    y = np.asarray([next(l["value"] for l in r["labels"] if l["table"] == "Satellite")
                    for r in sat], dtype=np.float32)

    # ---- song-level 3-way split ----
    groups = defaultdict(list)
    for i, r in enumerate(sat):
        groups[song_group_key(r)].append(i)
    gids = sorted(groups)
    random.Random(args.seed).shuffle(gids)
    n_val = int(round(len(gids) * args.val_ratio))
    n_test = int(round(len(gids) * args.test_ratio))
    val_g = set(gids[:n_val])
    test_g = set(gids[n_val:n_val + n_test])
    train_idx = [i for g in gids[n_val + n_test:] for i in groups[g]]
    val_idx = [i for g in val_g for i in groups[g]]
    test_idx = [i for g in test_g for i in groups[g]]
    print(f"songs: {len(gids)}  train {len(train_idx)} / val {len(val_idx)} / test {len(test_idx)}")

    # 可复现：split + schema + 归一化参数
    split_info = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "n_songs": len(gids),
        "train_sha256": [sat[i]["sha256"] for i in train_idx],
        "val_sha256": [sat[i]["sha256"] for i in val_idx],
        "test_sha256": [sat[i]["sha256"] for i in test_idx],
    }
    with open(os.path.join(args.out, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "features_schema.json"), "w", encoding="utf-8") as f:
        json.dump({"names": FEATURE_NAMES, "groups": FEATURE_GROUPS}, f,
                  ensure_ascii=False, indent=2)

    Xtr, Xva, Xte = X[train_idx], X[val_idx], X[test_idx]
    ytr, yva, yte = y[train_idx], y[val_idx], y[test_idx]
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0)
    std[std == 0] = 1.0
    Xs_tr = (Xtr - mean) / std
    Xs_va = (Xva - mean) / std
    Xs_te = (Xte - mean) / std
    with open(os.path.join(args.out, "normalization.json"), "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)

    results: dict = {"seed": args.seed, "n_charts": len(sat),
                     "n_excluded_flagged": len(flagged), "device": device}

    # ---- sl 直方图 ----
    bins = np.arange(-0.5, 13.5, 1)
    plt.figure(figsize=(9, 4))
    plt.hist(ytr, bins=bins, alpha=0.7, label=f"train ({len(ytr)})")
    plt.hist(yva, bins=bins, alpha=0.7, label=f"val ({len(yva)})")
    plt.hist(yte, bins=bins, alpha=0.7, label=f"test ({len(yte)})")
    plt.xticks(range(0, 13))
    plt.xlabel("Satellite level (sl)")
    plt.ylabel("charts")
    plt.legend()
    plt.title("sl histogram by split")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "sl_histogram.png"), dpi=120)
    plt.close()

    # ---- 1. mean predictor ----
    tr_mean = float(ytr.mean())
    mean_va = metrics(yva, np.full_like(yva, tr_mean))
    mean_te = metrics(yte, np.full_like(yte, tr_mean))
    results["mean_predictor"] = {"train_mean": tr_mean, "val": mean_va, "test": mean_te}
    print(f"\nmean predictor: test MAE {mean_te['mae']}  RMSE {mean_te['rmse']}  R2 {mean_te['r2']}")

    # ---- 2. 单变量分析 ----
    single = {}
    for j, name in enumerate(FEATURE_NAMES):
        x = X[:, j].astype(np.float64)
        yv = y.astype(np.float64)
        r = pearson(x, yv)
        rho = spearman(x, yv)
        z = np.abs((x - x.mean()) / (x.std() + 1e-9))
        outliers = int((z > 5).sum())
        zeros = float((x == 0).mean())
        # 各 sl 级均值/中位数
        lv_means, lv_meds, lv_counts = {}, {}, {}
        for lv in range(13):
            mask = yv == lv
            if mask.sum():
                lv_means[str(lv)] = round(float(x[mask].mean()), 3)
                lv_meds[str(lv)] = round(float(np.median(x[mask])), 3)
                lv_counts[str(lv)] = int(mask.sum())
        # 单调性：等级均值随 sl 是否单调
        order = [lv for lv in range(13) if str(lv) in lv_means]
        monotone = int(all(lv_means[str(a)] <= lv_means[str(b)] for a, b in zip(order, order[1:])))
        single[name] = {
            "pearson_r": round(r, 4), "spearman_rho": round(rho, 4),
            "mean": round(float(x.mean()), 3), "median": round(float(np.median(x)), 3),
            "min": round(float(x.min()), 3), "max": round(float(x.max()), 3),
            "pct_zero": zeros, "n_outliers_z5": outliers,
            "level_means": lv_means, "level_medians": lv_meds,
            "level_counts": lv_counts, "level_mean_monotone": monotone,
            "nonlinear_hint": abs(rho - r) > 0.15,
        }
        # 关系图：feature vs sl + 等级均值线
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.scatter(x, yv, s=4, alpha=0.25)
        lv_x = [float(k) for k in lv_means]
        lv_y = [lv_means[k] for k in lv_means]
        ax.plot(lv_x, lv_y, "r-", lw=2, label="level mean")
        ax.set_xlabel(name)
        ax.set_ylabel("sl")
        ax.set_title(f"{name}  r={r:.2f} rho={rho:.2f}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"feature_{j:02d}_{name}.png"), dpi=100)
        plt.close(fig)
    with open(os.path.join(args.out, "single_feature_report.json"), "w", encoding="utf-8") as f:
        json.dump(single, f, ensure_ascii=False, indent=2)
    results["single_feature_top"] = sorted(
        ((n, d["spearman_rho"], d["pearson_r"], d["nonlinear_hint"]) for n, d in single.items()),
        key=lambda t: -abs(t[1]),
    )[:10]
    print("\ntop features by |spearman|:")
    for n, rho, r, nl in results["single_feature_top"]:
        print(f"  {n:22s} rho={rho:+.3f}  pearson={r:+.3f}  nonlinear_hint={nl}")

    # ---- 3. 线性回归（OLS，标准化特征）----
    def fit_linear(Xa, ya):
        A = np.hstack([Xa, np.ones((Xa.shape[0], 1))])
        coef, *_ = np.linalg.lstsq(A, ya, rcond=None)
        return coef

    coef = fit_linear(Xs_tr, ytr)
    pred_va_l = Xs_va @ coef[:-1] + coef[-1]
    pred_te_l = Xs_te @ coef[:-1] + coef[-1]
    lin_va = metrics(yva, pred_va_l)
    lin_te = metrics(yte, pred_te_l)
    results["linear_regression"] = {
        "val": lin_va, "test": lin_te,
        "coefficients": {n: round(float(c), 4) for n, c in zip(FEATURE_NAMES, coef[:-1])},
        "intercept": round(float(coef[-1]), 4),
    }
    print(f"\nlinear: val MAE {lin_va['mae']}  test MAE {lin_te['mae']}  "
          f"test R2 {lin_te['r2']}")

    # ---- 4. MLP（26→64→32→1，早停）----
    def run_mlp(Xa, ya, Xv, yv, Xt, yt, tag):
        model = MLPBaseline(input_dim=Xa.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.SmoothL1Loss()
        ds = torch.utils.data.TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya))
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
        best_mae, best_state, patience = float("inf"), None, 0
        for epoch in range(args.epochs):
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
            if mae_v < best_mae - 1e-5:
                best_mae, patience = mae_v, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= args.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            pva = model(torch.from_numpy(Xv).to(device)).squeeze(-1).cpu().numpy()
            pte = model(torch.from_numpy(Xt).to(device)).squeeze(-1).cpu().numpy()
        return metrics(yv, pva), metrics(yt, pte), pte, epoch + 1

    mlp_va, mlp_te, pred_te_m, mlp_epochs = run_mlp(
        Xs_tr, ytr, Xs_va, yva, Xs_te, yte, "mlp")
    results["mlp"] = {"val": mlp_va, "test": mlp_te, "epochs_used": mlp_epochs,
                      "early_stopping_patience": args.patience}
    print(f"\nMLP({args.epochs} max, {mlp_epochs} used): val MAE {mlp_va['mae']}  "
          f"test MAE {mlp_te['mae']}  test R2 {mlp_te['r2']}")

    # ---- rounded accuracy ----
    for tag, p in (("linear", pred_te_l), ("mlp", pred_te_m)):
        acc = float(np.mean(np.round(p) == np.round(yte)))
        results[f"{tag}_exact_level_accuracy"] = round(acc, 4)

    # ---- pred vs truth / residual 图 ----
    plt.figure(figsize=(6, 5))
    plt.scatter(yte, pred_te_l, s=10, alpha=0.4, label="linear")
    plt.scatter(yte, pred_te_m, s=10, alpha=0.4, label="MLP")
    plt.plot([0, 12], [0, 12], "k--", lw=1)
    plt.xlabel("true sl"); plt.ylabel("predicted sl")
    plt.title("prediction vs ground truth (test)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "pred_vs_truth.png"), dpi=120)
    plt.close()

    res = yte - pred_te_m
    plt.figure(figsize=(6, 5))
    plt.scatter(yte, res, s=10, alpha=0.4)
    plt.axhline(0, color="k", lw=1)
    plt.xlabel("true sl"); plt.ylabel("residual (pred - true)")
    plt.title("MLP residual vs truth (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "residual_vs_truth.png"), dpi=120)
    plt.close()

    # ---- 残差 vs 关键特征 ----
    resid_feats = {
        "note_count": Xte[:, FEATURE_NAMES.index("total_notes")],
        "avg_nps": Xte[:, FEATURE_NAMES.index("avg_nps")],
        "duration": Xte[:, FEATURE_NAMES.index("duration_sec")],
        "max_bpm": Xte[:, FEATURE_NAMES.index("max_bpm")],
        "ln_ratio": Xte[:, FEATURE_NAMES.index("ln_ratio")],
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes[0, 0].scatter(yte, res, s=8, alpha=0.4)
    axes[0, 0].set_title("residual vs sl")
    for ax, (fn, fv) in zip(
        [axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]],
        resid_feats.items(),
    ):
        ax.scatter(fv, res, s=8, alpha=0.4)
        ax.set_title(f"residual vs {fn}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "residual_vs_features.png"), dpi=110)
    plt.close(fig)

    resid_report = {
        "per_sl_mean_abs_error": {
            str(lv): round(float(np.mean(np.abs(res[yte == lv]))), 3)
            for lv in range(13) if (yte == lv).any()
        },
        "abs_resid_corr": {
            fn: round(float(pearson(fv, np.abs(res))), 3)
            for fn, fv in resid_feats.items()
        },
    }
    with open(os.path.join(args.out, "residual_report.json"), "w", encoding="utf-8") as f:
        json.dump(resid_report, f, ensure_ascii=False, indent=2)
    results["residual"] = resid_report

    # ---- 5. 消融 ----
    ablation = {}
    for gname, names in FEATURE_GROUPS.items():
        idx = [FEATURE_NAMES.index(n) for n in names]
        c = fit_linear(Xs_tr[:, idx], ytr)
        pv = Xs_va[:, idx] @ c[:-1] + c[-1]
        pt = Xs_te[:, idx] @ c[:-1] + c[-1]
        ablation[gname] = {
            "features": names,
            "val": metrics(yva, pv), "test": metrics(yte, pt),
        }
    c_all = fit_linear(Xs_tr, ytr)
    ablation["ALL"] = {"features": FEATURE_NAMES,
                       "val": metrics(yva, Xs_va @ c_all[:-1] + c_all[-1]),
                       "test": metrics(yte, Xs_te @ c_all[:-1] + c_all[-1])}
    results["ablation_linear"] = ablation
    print("\nablation (linear, test MAE / R2):")
    for g, d in ablation.items():
        print(f"  {g:20s} {d['test']['mae']:.3f}  R2 {d['test']['r2']:.3f}")

    # 消融柱状图
    plt.figure(figsize=(8, 4))
    names = list(ablation)
    maes = [ablation[g]["test"]["mae"] for g in names]
    plt.bar(range(len(names)), maes, color="steelblue")
    plt.axhline(mean_te["mae"], color="red", ls="--", label="mean baseline")
    plt.xticks(range(len(names)), names, rotation=30)
    plt.ylabel("test MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "ablation.png"), dpi=110)
    plt.close()

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n== summary ==")
    print(f"mean predictor   test MAE {mean_te['mae']}")
    print(f"linear regression test MAE {lin_te['mae']}  R2 {lin_te['r2']}")
    print(f"MLP              test MAE {mlp_te['mae']}  R2 {mlp_te['r2']}")
    print(f"exact-level acc: linear {results['linear_exact_level_accuracy']}  "
          f"MLP {results['mlp_exact_level_accuracy']}")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
