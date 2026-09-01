"""Lane shuffle 破坏测试。
对每张 chart：整谱随机打乱 lane 列（delta_t/type/duration 不变，
每 chart 的 lane 出现频率不变），固定种子每 chart 一个乱序，val/test 可复现。
同一 CNN（2×k5）重训，与原始 sequence 对比。
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
    band_mae, build_raw_windows, chart_metrics, make_delta_rep, train_eval, zscore_train,
)
from .receptive_field_experiment import VariantCNN


def shuffle_lanes(seq: np.ndarray, seed: int) -> np.ndarray:
    out = seq.copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(out[:, 1])
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "manifest.jsonl"))
    ap.add_argument("--analysis", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "analysis"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "analysis", "lane_shuffle"))
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
    # ---- sanity：同一谱面 original vs shuffled ----
    print("=== sanity: original vs shuffled（delta_t 必须一致） ===")
    test_r = charts["test"][0]
    orig = np.load(os.path.join(seq_root, test_r["sha256"] + ".npy"))
    seed = int(test_r["sha256"][:8], 16) % (2**31)
    shuf = shuffle_lanes(orig, seed)
    print(f"chart: {test_r['title']}  (seed={seed})")
    print(f"{'i':>2s} {'dt_orig':>8s} {'lane_orig':>9s} | {'dt_shuf':>8s} {'lane_shuf':>9s}")
    for i in range(min(10, orig.shape[0])):
        dt_o = orig[i, 0] - (orig[i - 1, 0] if i else orig[0, 0])
        dt_s = shuf[i, 0] - (shuf[i - 1, 0] if i else shuf[0, 0])
        print(f"{i:2d} {dt_o:8.4f} {int(orig[i,1]):9d} | {dt_s:8.4f} {int(shuf[i,1]):9d}")
    print(f"lane 多重集一致: {sorted(orig[:,1].tolist()) == sorted(shuf[:,1].tolist())}")
    # ---- 构建 original / shuffled 窗口 ----
    raw_orig = {k: build_raw_windows(charts[k], seq_root, args.window, args.stride)
                for k in ("train", "val", "test")}
    raw_shuf = {}
    for k in ("train", "val", "test"):
        wins, shas = [], []
        for r in charts[k]:
            seq = np.load(os.path.join(seq_root, r["sha256"] + ".npy"))
            seq = shuffle_lanes(seq, int(r["sha256"][:8], 16) % (2**31))
            n = seq.shape[0]
            for start in range(0, n - args.window + 1, args.stride):
                wins.append(seq[start:start + args.window])
                shas.append(r["sha256"])
        raw_shuf[k] = (np.asarray(wins, dtype=np.float32), np.asarray(shas))

    def prepare(raw_):
        m, s = zscore_train(make_delta_rep(raw_["train"][0]))
        return {k: ((make_delta_rep(raw_[k][0]) - m) / s).astype(np.float32)
                for k in ("train", "val", "test")}

    rep_orig = prepare(raw_orig)
    rep_shuf = prepare(raw_shuf)
    y = {k: np.asarray([chart_labels[s] for s in raw_orig[k][1]], np.float32)
         for k in ("train", "val", "test")}

    def run(rep, raw_):
        Xtr, Xva, Xte = rep["train"], rep["val"], rep["test"]
        mae_list, rmse_list, r2_list, bands_list = [], [], [], []
        t0 = time.time()
        for seed in seeds:
            model = VariantCNN(seq_dim=4, num_convs=2, kernel=5).to(device)
            pt = train_eval(model, Xtr, y["train"], Xva, y["val"], Xte, device, args, seed)
            cm, yt, yp = chart_metrics(pt, raw_["test"][1], chart_labels)
            mae_list.append(cm["mae"]); rmse_list.append(cm["rmse"])
            r2_list.append(cm["r2"]); bands_list.append(band_mae(yt, yp))
        return {
            "train_seconds": round(time.time() - t0, 1),
            "chart_mae": round(float(np.mean(mae_list)), 4),
            "chart_rmse": round(float(np.mean(rmse_list)), 4),
            "chart_r2": round(float(np.mean(r2_list)), 4),
            "seed_maes": [round(x, 4) for x in mae_list],
            "bands": {k: round(float(np.nanmean([b[k] for b in bands_list])), 4)
                      for k in bands_list[0]},
        }

    res_orig = run(rep_orig, raw_orig)
    res_shuf = run(rep_shuf, raw_shuf)
    print("\n=== 结果（3 种子均值，2×k5） ===")
    print(f"original     : MAE {res_orig['chart_mae']}  RMSE {res_orig['chart_rmse']}  "
          f"R2 {res_orig['chart_r2']}  time {res_orig['train_seconds']}s")
    print(f"lane-shuffled: MAE {res_shuf['chart_mae']}  RMSE {res_shuf['chart_rmse']}  "
          f"R2 {res_shuf['chart_r2']}  time {res_shuf['train_seconds']}s")
    print("bands original :", res_orig["bands"])
    print("bands shuffled :", res_shuf["bands"])
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"original": res_orig, "lane_shuffled": res_shuf,
                   "config": {"window": args.window, "seeds": seeds,
                              "shuffle_seed_per_chart": "sha256[:8] % 2^31"}},
                  f, ensure_ascii=False, indent=2)
    print("\nsaved ->", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
