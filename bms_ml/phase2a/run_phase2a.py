"""Phase 2A 第一轮最小实验主流程。

运行：
  .venv\\Scripts\\python.exe -m bms_ml.phase2a.run_phase2a

流程：
  1. clean corpus → song-level split（Satellite test 歌曲从 pretrain train 排除）；
  2. 构建 4s / 1/60s 固定时间网格窗口；
  3. T1 masked reconstruction 训练 + density-only baseline；
  4. T1 onset-only（LN ablation）训练；
  5. T2 next-window 训练 + density-only baseline；
  6. R1 结构恢复 / R2 SL 线性探针 / R3 结构探针 / R4 变换敏感性；
  7. 写 results.json 与 PHASE2A_REPORT.md。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..split import song_group_key
from .grid_data import (
    KEY_PERMS, N_CH, N_LANES, T, SeqCache, density_baseline_masks,
    density_baseline_next, grids_for, local_shuffle_window, permute_lanes_seq,
    seq_grid, shift_seq, window_items, window_starts,
)
from .models import GridEncoder, LinearProbe, NextWindowModel
from .train import eval_t1, eval_t2, make_masks_np, set_seed, train_t1, train_t2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MANIFEST = os.path.join(ROOT, "output", "corpus", "manifest.jsonl")
DEFAULT_ANALYSIS = os.path.join(ROOT, "output", "corpus", "analysis")
DEFAULT_SEQ = os.path.join(ROOT, "output", "corpus", "sequences")
DEFAULT_OUT = os.path.join(ROOT, "output", "phase2a")


def load_clean_records(manifest: str) -> List[dict]:
    out = []
    with open(manifest, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r["quarantine"] and r["has_7k"] and not r["flags"]:
                out.append(r)
    return out


def build_pretrain_split(records: List[dict], sat_split: dict, seed: int = 0,
                         train_ratio: float = 0.90, val_ratio: float = 0.05) -> Tuple[dict, dict]:
    """song-level split；Satellite test 歌曲所在组不得进入 pretrain train。"""
    groups: dict = defaultdict(list)
    for i, r in enumerate(records):
        groups[song_group_key(r)].append(i)
    sat_test = set(sat_split["test_sha256"])
    sat_test_groups = {g for g, idxs in groups.items()
                       if any(records[i]["sha256"] in sat_test for i in idxs)}
    gids = sorted(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(gids)
    n_tr = int(round(len(gids) * train_ratio))
    n_va = int(round(len(gids) * val_ratio))
    assign = {}
    for k, g in enumerate(gids):
        assign[g] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
    movers = [g for g in gids if assign[g] == "train" and g in sat_test_groups]
    donors = [g for g in gids if assign[g] == "test" and g not in sat_test_groups]
    for g, d in zip(movers, donors):
        assign[g], assign[d] = "test", "train"
    for g in movers[len(donors):]:
        assign[g] = "val"
    out = {"train": [], "val": [], "test": []}
    for g in gids:
        for i in groups[g]:
            out[assign[g]].append(records[i])
    report = {
        "n_charts": len(records), "n_groups": len(gids),
        "satellite_test_groups_moved_out_of_train": len(movers),
        **{f"charts_{k}": len(v) for k, v in out.items()},
    }
    return out, report


def load_or_build_grids(items: List[Tuple[str, float]], seq_root: str,
                        include_hold: bool, name: str, out_dir: str,
                        force: bool = False) -> np.ndarray:
    key = f"{name}_hold{int(include_hold)}"
    arr_path = os.path.join(out_dir, f"grids_{key}.npy")
    meta_path = os.path.join(out_dir, f"items_{key}.json")
    items_list = [[s, float(t)] for s, t in items]
    if not force and os.path.exists(arr_path) and os.path.exists(meta_path):
        meta = json.load(open(meta_path, encoding="utf-8"))
        if meta.get("items") == items_list:
            print(f"  [cache] reuse {name} grids ({len(items)} windows)")
            return np.load(arr_path)
    print(f"  [build] {name} grids ({len(items)} windows) ...")
    arr = grids_for(items, seq_root, include_hold=include_hold)
    np.save(arr_path, arr)
    json.dump({"items": items_list}, open(meta_path, "w", encoding="utf-8"))
    return arr


def window_pairs(records: List[dict], seq_root: str, max_pairs: int | None = None,
                 seed: int = 0) -> List[Tuple[str, float, float]]:
    """相邻窗口对 (sha, start_cur, start_next)，不重叠。"""
    cache = SeqCache(seq_root)
    rng = np.random.RandomState(seed)
    pairs: List[Tuple[str, float, float]] = []
    for r in records:
        seq = cache(r["sha256"])
        if len(seq) == 0:
            continue
        dur = float(seq[-1, 0] - seq[0, 0])
        starts = window_starts(dur)
        if len(starts) < 2:
            continue
        ps = [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]
        if max_pairs and len(ps) > max_pairs:
            idx = np.linspace(0, len(ps) - 1, max_pairs).round().astype(int)
            ps = [ps[i] for i in sorted(set(idx.tolist()))]
        pairs.extend((r["sha256"], float(a), float(b)) for a, b in ps)
    return pairs


@torch.no_grad()
def encode_grids(encoder: torch.nn.Module, grids: np.ndarray, device,
                 batch: int = 256) -> np.ndarray:
    encoder = encoder.to(device)
    encoder.eval()
    outs = []
    for i in range(0, len(grids), batch):
        x = torch.from_numpy(grids[i:i + batch]).permute(0, 3, 1, 2).to(device).float()
        outs.append(encoder.pool(x).cpu().numpy())
    return np.concatenate(outs, axis=0)


def average_by_sha(feat: np.ndarray, shas: Sequence[str]):
    agg: dict = defaultdict(list)
    for f, s in zip(feat, shas):
        agg[s].append(f)
    keys = sorted(agg)
    X = np.stack([np.mean(agg[s], axis=0) for s in keys])
    return X, keys


def linear_probe_reg(Xtr, ytr, Xva, yva, Xte, yte, device, seed: int = 0,
                     epochs: int = 400, lr: float = 1e-3, patience: int = 40) -> dict:
    set_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    zn = lambda a: (a - mu) / sd
    Xtrn, Xvan, Xten = zn(Xtr), zn(Xva), zn(Xte)
    model = LinearProbe(Xtr.shape[1], 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = F.smooth_l1_loss
    Xv_t = torch.from_numpy(Xvan).to(device)
    yv_t = torch.from_numpy(yva).to(device)
    best, best_state, pl = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(len(Xtrn))
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            xb = torch.from_numpy(Xtrn[idx]).to(device)
            yb = torch.from_numpy(ytr[idx]).to(device)
            opt.zero_grad()
            loss = crit(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv_t).squeeze(-1).cpu().numpy()
        mae = float(np.mean(np.abs(yva - pv)))
        if mae < best - 1e-6:
            best, pl = mae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            pl += 1
            if pl >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pt = model(torch.from_numpy(Xten).to(device)).squeeze(-1).cpu().numpy()
    mae = float(np.mean(np.abs(yte - pt)))
    rmse = float(np.sqrt(np.mean((yte - pt) ** 2)))
    ss_tot = float(np.sum((yte - yte.mean()) ** 2))
    r2 = 1.0 - float(np.sum((yte - pt) ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4),
            "best_val_mae": round(best, 4)}


def linear_probe_clf(X: np.ndarray, y: np.ndarray, n_class: int, device,
                     group_ids: np.ndarray, seed: int = 0, epochs: int = 200,
                     lr: float = 1e-2, test_frac: float = 0.2) -> dict:
    """组感知划分的线性分类器（同窗口的样本不跨 train/test）。"""
    set_seed(seed)
    groups = np.unique(group_ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    n_te = max(1, int(round(len(groups) * test_frac)))
    te_groups = set(groups[:n_te].tolist())
    tr_mask = np.array([g not in te_groups for g in group_ids])
    te_mask = ~tr_mask
    if tr_mask.sum() < 10 or te_mask.sum() < 10:
        return {"test_acc": None, "chance": 1.0 / n_class, "n_train": int(tr_mask.sum()),
                "n_test": int(te_mask.sum())}
    mu, sd = X[tr_mask].mean(0), X[tr_mask].std(0) + 1e-8
    Xtr = (X[tr_mask] - mu) / sd
    Xte = (X[te_mask] - mu) / sd
    ytr = y[tr_mask].astype(np.int64)
    yte = y[te_mask].astype(np.int64)
    model = LinearProbe(X.shape[1], n_class).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=128, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xte).to(device))
        acc = float((logits.argmax(1).cpu().numpy() == yte).mean())
    return {"test_acc": float(round(acc, 4)), "chance": float(round(1.0 / n_class, 4)),
            "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum())}


def prob_metrics_per_sample(probs: np.ndarray, targets: np.ndarray,
                            masks_int: np.ndarray, channel: int = 0,
                            span_only: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = masks_int != 0 if not span_only else masks_int == 1
    y = targets[:, :, :, channel] > 0
    p = probs[:, :, :, channel] > 0.5
    sel = mask
    tp = ((p & y) & sel).sum(axis=(1, 2)).astype(np.float64)
    fp = ((p & ~y) & sel).sum(axis=(1, 2)).astype(np.float64)
    fn = ((~p & y) & sel).sum(axis=(1, 2)).astype(np.float64)
    return tp, fp, fn


def pooled_prf(tp: float, fp: float, fn: float) -> dict:
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"precision": float(round(prec, 4)), "recall": float(round(rec, 4)),
            "f1": float(round(f1, 4))}


def baseline_eval(probs: np.ndarray, grids: np.ndarray, masks_int: np.ndarray,
                  name: str) -> dict:
    tp, fp, fn = prob_metrics_per_sample(probs, grids, masks_int, 0)
    out = {"onset": pooled_prf(tp.sum(), fp.sum(), fn.sum()),
           "n_windows": int(len(grids))}
    dens = (grids[:, :, :, 0] > 0).sum(axis=(1, 2)).astype(np.float64)
    q1, q2 = np.quantile(dens, [1 / 3, 2 / 3])
    for lbl, sel in (("low", dens <= q1), ("mid", (dens > q1) & (dens <= q2)),
                     ("high", dens > q2)):
        if sel.sum() == 0:
            out[f"stratum_{lbl}"] = None
            continue
        out[f"stratum_{lbl}"] = {**pooled_prf(tp[sel].sum(), fp[sel].sum(), fn[sel].sum()),
                                 "n": int(sel.sum())}
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--analysis", default=DEFAULT_ANALYSIS)
    ap.add_argument("--seq-root", default=DEFAULT_SEQ)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-windows", type=int, default=10)
    ap.add_argument("--max-val-windows", type=int, default=12)
    ap.add_argument("--max-test-windows", type=int, default=16)
    ap.add_argument("--epochs-t1", type=int, default=20)
    ap.add_argument("--epochs-t2", type=int, default=12)
    ap.add_argument("--epochs-ablation", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--skip-t1", action="store_true")
    ap.add_argument("--skip-t2", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    t_start = time.time()

    report: dict = {"config": vars(args), "device": device}

    # ---- 1. 数据与 split ----
    print("== 1. load clean records ==")
    records = load_clean_records(args.manifest)
    sat_split = json.load(open(os.path.join(args.analysis, "split.json"), encoding="utf-8"))
    split, split_report = build_pretrain_split(records, sat_split, seed=args.seed)
    report["split"] = split_report
    print("split:", split_report)
    with open(os.path.join(args.out, "pretrain_split.json"), "w", encoding="utf-8") as f:
        json.dump({"report": split_report,
                   "train_sha256": [r["sha256"] for r in split["train"]],
                   "val_sha256": [r["sha256"] for r in split["val"]],
                   "test_sha256": [r["sha256"] for r in split["test"]]},
                  f, ensure_ascii=False, indent=2)

    # ---- 2. 窗口与网格 ----
    print("== 2. build window grids ==")
    tr_items = window_items(split["train"], args.seq_root,
                            max_windows=args.max_train_windows, seed=args.seed)
    va_items = window_items(split["val"], args.seq_root,
                            max_windows=args.max_val_windows, seed=args.seed)
    te_items = window_items(split["test"], args.seq_root,
                            max_windows=args.max_test_windows, seed=args.seed)
    tr_grids = load_or_build_grids(tr_items, args.seq_root, True, "train", args.out,
                                   force=args.force_rebuild)
    va_grids = load_or_build_grids(va_items, args.seq_root, True, "val", args.out,
                                   force=args.force_rebuild)
    te_grids = load_or_build_grids(te_items, args.seq_root, True, "test", args.out,
                                   force=args.force_rebuild)
    report["windows"] = {"train": int(len(tr_items)), "val": int(len(va_items)),
                         "test": int(len(te_items))}
    print("window counts:", report["windows"])

    # ---- 3. T1 评估协议：固定 mask（与 baseline 完全一致） ----
    print("== 3. T1: density-only baseline vs encoder ==")
    te_masks = make_masks_np(len(te_grids), seed=777)
    base_probs = density_baseline_masks(te_grids, te_masks)
    base_res = baseline_eval(base_probs, te_grids, te_masks, "density_baseline")
    report["R1_density_baseline"] = base_res
    print("density-only baseline (T1):", base_res["onset"])

    set_seed(args.seed)
    rand_enc = GridEncoder()
    rand_res = eval_t1(rand_enc, te_grids, te_masks, device,
                       batch_size=args.batch, include_hold=True)
    report["R1_random_init"] = rand_res
    print("random-init encoder (T1):", rand_res["onset"])

    if not args.skip_t1:
        print("== 3b. train T1 (masked reconstruction, with hold) ==")
        set_seed(args.seed)
        enc = GridEncoder()
        enc, hist = train_t1(enc, tr_grids, va_grids, device,
                             epochs=args.epochs_t1, batch_size=args.batch,
                             lr=args.lr, seed=args.seed)
        torch.save(enc.state_dict(), os.path.join(args.out, "t1_hold.pt"))
        t1_res = eval_t1(enc, te_grids, te_masks, device,
                         batch_size=args.batch, include_hold=True)
        report["R1_t1_hold"] = t1_res
        report["T1_hold_train"] = hist
        print("T1 (hold) test:", t1_res["onset"], "hold:", t1_res.get("hold"))

        # LN ablation：onset-only（hold 通道完全关闭）
        if not args.skip_ablation:
            print("== 3c. LN ablation: onset-only ==")
            tr_onset_items = window_items(split["train"], args.seq_root,
                                          max_windows=8, seed=args.seed + 1)
            tr_onset = load_or_build_grids(tr_onset_items, args.seq_root, False,
                                           "train_onset", args.out,
                                           force=args.force_rebuild)
            set_seed(args.seed)
            enc_o = GridEncoder()
            enc_o, hist_o = train_t1(enc_o, tr_onset, va_grids, device,
                                     epochs=args.epochs_ablation, batch_size=args.batch,
                                     lr=args.lr, seed=args.seed,
                                     include_hold=False)
            torch.save(enc_o.state_dict(), os.path.join(args.out, "t1_onset.pt"))
            t1_o_res = eval_t1(enc_o, te_grids, te_masks, device,
                               batch_size=args.batch, include_hold=False)
            report["R1_t1_onset_only"] = t1_o_res
            report["T1_onset_train"] = hist_o
            print("T1 (onset-only) test:", t1_o_res["onset"])
    else:
        enc = GridEncoder()
        ckpt = os.path.join(args.out, "t1_hold.pt")
        if os.path.exists(ckpt):
            enc.load_state_dict(torch.load(ckpt, map_location=device))
            print("loaded T1 checkpoint")
            t1_res = eval_t1(enc, te_grids, te_masks, device,
                             batch_size=args.batch, include_hold=True)
            report["R1_t1_hold"] = t1_res
            print("T1 (hold) test (re-eval):", t1_res["onset"])
            onset_ckpt = os.path.join(args.out, "t1_onset.pt")
            if os.path.exists(onset_ckpt):
                enc_o = GridEncoder()
                enc_o.load_state_dict(torch.load(onset_ckpt, map_location=device))
                t1_o_res = eval_t1(enc_o, te_grids, te_masks, device,
                                   batch_size=args.batch, include_hold=False)
                report["R1_t1_onset_only"] = t1_o_res
                print("T1 (onset-only) test (re-eval):", t1_o_res["onset"])
        else:
            raise SystemExit("--skip-t1 but no t1_hold.pt")

    # ---- 4. T2 next-window 对照 ----
    print("== 4. T2: next-window prediction ==")
    va_pairs = window_pairs(split["val"], args.seq_root,
                            max_pairs=12, seed=args.seed)
    te_pairs = window_pairs(split["test"], args.seq_root,
                            max_pairs=16, seed=args.seed)
    pair_arrays = {}
    for nm, pairs in (("val", va_pairs), ("test", te_pairs)):
        cur_path = os.path.join(args.out, f"grids_{nm}_t2cur.npy")
        nxt_path = os.path.join(args.out, f"grids_{nm}_t2nxt.npy")
        if os.path.exists(cur_path) and os.path.exists(nxt_path) and not args.force_rebuild:
            cur, nxt = np.load(cur_path), np.load(nxt_path)
        else:
            cache = SeqCache(args.seq_root)
            cur = np.stack([seq_grid(cache(s), a) for s, a, _ in pairs])
            nxt = np.stack([seq_grid(cache(s), b) for s, _, b in pairs])
            np.save(cur_path, cur)
            np.save(nxt_path, nxt)
        pair_arrays[nm] = (cur, nxt)
    va_cur, va_nxt = pair_arrays["val"]
    te_cur, te_nxt = pair_arrays["test"]

    if not args.skip_t2:
        tr_pairs = window_pairs(split["train"], args.seq_root,
                                max_pairs=10, seed=args.seed)
        tr_path = os.path.join(args.out, "grids_train_t2cur.npy")
        trn_path = os.path.join(args.out, "grids_train_t2nxt.npy")
        if os.path.exists(tr_path) and os.path.exists(trn_path) and not args.force_rebuild:
            tr_cur, tr_nxt = np.load(tr_path), np.load(trn_path)
        else:
            cache = SeqCache(args.seq_root)
            tr_cur = np.stack([seq_grid(cache(s), a) for s, a, _ in tr_pairs])
            tr_nxt = np.stack([seq_grid(cache(s), b) for s, _, b in tr_pairs])
            np.save(tr_path, tr_cur)
            np.save(trn_path, tr_nxt)
        print(f"  t2 pairs: train={len(tr_pairs)} val={len(va_pairs)} test={len(te_pairs)}")
        # T2 density-only baseline
        t2_base_probs = density_baseline_next(te_cur)
        t2_base = baseline_eval(t2_base_probs, te_nxt,
                                np.ones((len(te_nxt), T, N_LANES), dtype=np.int8),
                                "t2_density_baseline")
        report["R1_T2_density_baseline"] = t2_base
        print("T2 density-only baseline:", t2_base["onset"])

        set_seed(args.seed)
        enc2 = NextWindowModel(GridEncoder())
        enc2, hist2 = train_t2(enc2, tr_cur, tr_nxt, va_cur, va_nxt, device,
                               epochs=args.epochs_t2, batch_size=args.batch,
                               lr=args.lr, seed=args.seed)
        torch.save(enc2.state_dict(), os.path.join(args.out, "t2.pt"))
        t2_res = eval_t2(enc2, te_cur, te_nxt, device, batch_size=args.batch)
        report["R1_T2_encoder"] = t2_res
        report["T2_train"] = hist2
        print("T2 encoder test:", t2_res)
        t2_encoder = enc2.encoder
    else:
        enc2 = NextWindowModel(GridEncoder())
        ckpt = os.path.join(args.out, "t2.pt")
        if os.path.exists(ckpt):
            enc2.load_state_dict(torch.load(ckpt, map_location=device))
            print("loaded T2 checkpoint")
            t2_res = eval_t2(enc2, te_cur, te_nxt, device, batch_size=args.batch)
            report["R1_T2_encoder"] = t2_res
            print("T2 encoder test (re-eval):", t2_res)
        else:
            raise SystemExit("--skip-t2 but no t2.pt")
        t2_encoder = enc2.encoder

    # ---- 5. R2: SL 线性探针（frozen encoder，sanity） ----
    print("== 5. R2: SL linear probe ==")
    sat_records = [r for r in records
                   if any(l["table"] == "Satellite" and l["value"] is not None
                          for l in r["labels"])]
    by_sha = {r["sha256"]: r for r in sat_records}
    sat_by_split = {
        "train": [by_sha[s] for s in sat_split["train_sha256"]],
        "val": [by_sha[s] for s in sat_split["val_sha256"]],
        "test": [by_sha[s] for s in sat_split["test_sha256"]],
    }

    def sat_label(r):
        return float(next(l["value"] for l in r["labels"] if l["table"] == "Satellite"))

    def sat_feats(enc: torch.nn.Module) -> Tuple[dict, dict]:
        feats, labels = {}, {}
        for k in ("train", "val", "test"):
            items = window_items(sat_by_split[k], args.seq_root, max_windows=16,
                                 seed=args.seed)
            grids = load_or_build_grids(items, args.seq_root, True, f"sat_{k}",
                                        args.out, force=args.force_rebuild)
            f = encode_grids(enc, grids, device)
            X, keys = average_by_sha(f, [s for s, _ in items])
            y = np.asarray([sat_label(by_sha[s]) for s in keys], dtype=np.float32)
            feats[k], labels[k] = X, y
        return feats, labels

    pretrain_enc = enc
    sat_feats_pretrained, sat_labels = sat_feats(pretrain_enc)
    r2_pretrained = linear_probe_reg(
        sat_feats_pretrained["train"], sat_labels["train"],
        sat_feats_pretrained["val"], sat_labels["val"],
        sat_feats_pretrained["test"], sat_labels["test"], device, seed=args.seed)
    report["R2_sl_probe_pretrained"] = r2_pretrained
    print("R2 SL probe (pretrained):", r2_pretrained)

    set_seed(args.seed)
    rand_enc = GridEncoder()
    sat_feats_rand, _ = sat_feats(rand_enc)
    r2_rand = linear_probe_reg(
        sat_feats_rand["train"], sat_labels["train"],
        sat_feats_rand["val"], sat_labels["val"],
        sat_feats_rand["test"], sat_labels["test"], device, seed=args.seed)
    report["R2_sl_probe_random_init"] = r2_rand
    print("R2 SL probe (random init):", r2_rand)

    # ---- 6. R3: 结构探针（无人工标签） ----
    print("== 6. R3: structure probes ==")
    te_cur, te_nxt = pair_arrays["test"]
    probe_feat = encode_grids(t2_encoder, te_cur, device)
    nxt_dens = (te_nxt[:, :, :, 0] > 0).sum(axis=(1, 2)).astype(np.float32)
    chord_occ = ((te_cur[:, :, :, 0] > 0).sum(axis=2) >= 2).mean(axis=1).astype(np.float32)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(probe_feat))
    n_tr = int(len(perm) * 0.8)
    n_va = int(len(perm) * 0.9)
    split_idx = {"train": perm[:n_tr], "val": perm[n_tr:n_va], "test": perm[n_va:]}

    def struct_probe(feat, target, name):
        base_mean = float(np.mean(np.abs(target[split_idx["test"]] -
                                       target[split_idx["train"]].mean())))
        res = linear_probe_reg(feat[split_idx["train"]], target[split_idx["train"]],
                               feat[split_idx["val"]], target[split_idx["val"]],
                               feat[split_idx["test"]], target[split_idx["test"]],
                               device, seed=args.seed, epochs=300, patience=30)
        res["mean_baseline_mae"] = round(base_mean, 4)
        return res

    r3 = {}
    r3["next_density_pretrained"] = struct_probe(probe_feat, nxt_dens, "next_density")
    r3["chord_occupancy_pretrained"] = struct_probe(probe_feat, chord_occ, "chord_occ")

    set_seed(args.seed)
    rand_enc2 = GridEncoder()
    rand_feat = encode_grids(rand_enc2, te_cur, device)
    r3["next_density_random"] = struct_probe(rand_feat, nxt_dens, "next_density")
    r3["chord_occupancy_random"] = struct_probe(rand_feat, chord_occ, "chord_occ")
    report["R3_structure_probes"] = r3
    print("R3:", json.dumps(r3, ensure_ascii=False, indent=2))

    # ---- 7. R4: 变换敏感性（frozen encoder） ----
    print("== 7. R4: transformation sensitivity ==")
    rng = np.random.RandomState(args.seed)
    max_r4 = 800
    if len(te_items) > max_r4:
        idx = np.linspace(0, len(te_items) - 1, max_r4).round().astype(int)
        r4_items = [te_items[i] for i in idx]
    else:
        r4_items = list(te_items)
    cache = SeqCache(args.seq_root)

    def grids_for_items(it):
        return np.stack([seq_grid(cache(sha), st) for sha, st in it])

    base_g = grids_for_items(r4_items)
    lat_base = encode_grids(pretrain_enc, base_g, device)
    d = {}
    # 1) 时间平移：整曲平移 +7s，同一内容重新取窗 → 应几乎一致
    g_exact = np.stack([seq_grid(shift_seq(cache(sha), 7.0), st + 7.0)
                        for sha, st in r4_items])
    lat_exact = encode_grids(pretrain_enc, g_exact, device)
    d["time_shift_exact"] = feat_dist(lat_base, lat_exact)
    # 量化稳健性：内容平移 1ms（窗口起点不变）
    g_q = np.stack([seq_grid(shift_seq(cache(sha), 0.001), st)
                    for sha, st in r4_items])
    lat_q = encode_grids(pretrain_enc, g_q, device)
    d["time_shift_1ms"] = feat_dist(lat_base, lat_q)
    # 内容重叠：窗口平移 0.5s（87.5% 内容相同，边界不同）
    g_ov = np.stack([seq_grid(cache(sha), st + 0.5) for sha, st in r4_items])
    lat_ov = encode_grids(pretrain_enc, g_ov, device)
    d["window_shift_0.5s"] = feat_dist(lat_base, lat_ov)
    # 2) 全局 lane 置换（8 个固定置换，scratch 不动）：只测量
    perm_lat = np.zeros((len(r4_items), len(KEY_PERMS), lat_base.shape[1]),
                        dtype=np.float32)
    for j, perm in enumerate(KEY_PERMS):
        g_p = np.stack([seq_grid(permute_lanes_seq(cache(sha), perm), st)
                        for sha, st in r4_items])
        perm_lat[:, j] = encode_grids(pretrain_enc, g_p, device)
    perm_dist = [feat_dist(lat_base, perm_lat[:, j]) for j in range(len(KEY_PERMS))]
    d["global_perm_distances"] = perm_dist
    # 3) 局部 lane shuffle（结构破坏）：距离 + 可读性
    g_sh = np.stack([seq_grid(local_shuffle_window(cache(sha), st, rng), st)
                     for sha, st in r4_items])
    lat_sh = encode_grids(pretrain_enc, g_sh, device)
    d["local_shuffle_distance"] = feat_dist(lat_base, lat_sh)
    # random-init 参考距离（说明距离尺度）
    set_seed(args.seed)
    rand_enc = GridEncoder()
    rl_base = encode_grids(rand_enc, base_g, device)
    rl_exact = encode_grids(rand_enc, g_exact, device)
    rl_sh = encode_grids(rand_enc, g_sh, device)
    d["random_init_reference"] = {
        "time_shift_exact": feat_dist(rl_base, rl_exact),
        "local_shuffle": feat_dist(rl_base, rl_sh),
    }
    # 4) frozen-probe 可读性
    n_w = len(r4_items)
    perm_acc = linear_probe_clf(perm_lat.reshape(-1, lat_base.shape[1]),
                                np.tile(np.arange(len(KEY_PERMS)), n_w),
                                len(KEY_PERMS), device,
                                group_ids=np.repeat(np.arange(n_w), len(KEY_PERMS)),
                                seed=args.seed)
    d["probe_perm_classification"] = perm_acc
    shuf_acc = linear_probe_clf(
        np.concatenate([lat_base, lat_sh], axis=0),
        np.concatenate([np.zeros(n_w), np.ones(n_w)]),
        2, device,
        group_ids=np.concatenate([np.arange(n_w), np.arange(n_w)]),
        seed=args.seed)
    d["probe_local_shuffle_binary"] = shuf_acc
    report["R4_transforms"] = d
    print("R4:", json.dumps(d, ensure_ascii=False, indent=2))

    report["wall_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("results saved ->", os.path.join(args.out, "results.json"))
    write_report(report, args.out)


def feat_dist(a: np.ndarray, b: np.ndarray) -> dict:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    cos = (an * bn).sum(axis=1)
    l2 = np.linalg.norm(a - b, axis=1)
    return {"cosine_mean": round(float(cos.mean()), 4),
            "cosine_std": round(float(cos.std()), 4),
            "l2_mean": round(float(l2.mean()), 4)}


def write_report(report: dict, out_dir: str) -> None:
    R = report
    line = []
    line.append("# Phase 2A 第一轮实验报告（最小可运行版本）")
    line.append("")
    line.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} | 运行 {R.get('wall_seconds', '?')}s")
    line.append("")
    line.append("## 1. 数据规模与 split")
    sp = R["split"]
    line.append(f"- clean 7K 无 flag 谱面：{sp['n_charts']}（{sp['n_groups']} 首歌）")
    line.append(f"- pretrain split（song-level，seed={R['config']['seed']}）："
                f"train {sp['charts_train']} / val {sp['charts_val']} / test {sp['charts_test']}")
    line.append(f"- Satellite test 歌曲所在组从 pretrain train 排除：{sp['satellite_test_groups_moved_out_of_train']} 组")
    line.append(f"- 窗口数：train {R['windows']['train']} / val {R['windows']['val']} / test {R['windows']['test']}")
    line.append("")
    line.append("## 2. representation 的确切定义")
    line.append("- 固定时间网格：4s 窗口，1/60s cell（T=240），lane 轴 8（0-6 keys 有序 + 7 scratch 平面）")
    line.append("- 通道：C0 = onset 计数（cap 2）；C1 = LN hold 标志（最简 hold 表示）")
    line.append("- window-relative 时间；窗口不重叠；不含 BPM/STOP/SCROLL 输入；未来 side channel 预留")
    line.append("")
    line.append("## 3. T1 / T2 训练设置")
    line.append(f"- T1 masked reconstruction：遮挡 1s 连续 span + 10% 随机 cell，BCE（masked cells），"
                f"epochs={R['config']['epochs_t1']}，batch={R['config']['batch']}，lr={R['config']['lr']}")
    line.append(f"- T2 next-window：同编码器，预测下一 4s 网格，epochs={R['config']['epochs_t2']}")
    line.append(f"- 模型：小 CNN（3×Conv2d，time k7 / lane k3 + 全局池化分支），参数约 13 万")
    line.append("")
    line.append("## 4. density-only baseline")
    line.append("- T1：per-lane 未遮挡率 + ±1s 邻域未遮挡率的混合，预测遮挡区域")
    line.append("- T2：当前窗口 per-lane onset/hold 率 + 密度延续先验")
    line.append("")
    line.append("## 5. R1–R4 结果")
    line.append("### R1 结构恢复（held-out 歌曲，T1 masked reconstruction）")
    line.append("| 模型 | onset F1 | precision | recall | BCE(all) | BCE(span) |")
    line.append("|---|---|---|---|---|---|")
    for nm, key in (("density-only baseline", "R1_density_baseline"),
                    ("random-init encoder", "R1_random_init"),
                    ("T1 encoder (with hold)", "R1_t1_hold"),
                    ("T1 encoder (onset-only ablation)", "R1_t1_onset_only")):
        v = R.get(key)
        if not v:
            continue
        line.append(f"| {nm} | {v['onset']['f1']} | {v['onset']['precision']} | "
                    f"{v['onset']['recall']} | {v.get('bce_all', '-')} | "
                    f"{v.get('bce_span', '-')} |")
    line.append("")
    line.append("R1 分密度层（onset F1，低/中/高）：")
    for nm, key in (("density-only", "R1_density_baseline"), ("T1 encoder", "R1_t1_hold")):
        v = R.get(key)
        if v:
            line.append(f"- {nm}: " + ", ".join(
                f"{k}={v[k]['f1']}" for k in ("stratum_low", "stratum_mid", "stratum_high") if v[k]))
    line.append("")
    line.append("### T2 next-window（对照）")
    t2b = R.get("R1_T2_density_baseline")
    t2e = R.get("R1_T2_encoder")
    if t2b and t2e:
        line.append("| 模型 | onset F1 | precision | recall | density MAE |")
        line.append("|---|---|---|---|---|")
        line.append(f"| density-only | {t2b['onset']['f1']} | {t2b['onset']['precision']} | "
                    f"{t2b['onset']['recall']} | - |")
        line.append(f"| T2 encoder | {t2e['onset']['f1']} | {t2e['onset']['precision']} | "
                    f"{t2e['onset']['recall']} | {t2e['density_mae']} |")
    line.append("")
    line.append("### R2 SL 线性探针（frozen encoder，sanity check）")
    r2p, r2r = R.get("R2_sl_probe_pretrained"), R.get("R2_sl_probe_random_init")
    if r2p:
        line.append(f"- pretrained：test MAE {r2p['mae']}，R² {r2p['r2']}（Phase 1 参考：26 特征 MLP MAE 1.068，mean 3.382）")
    if r2r:
        line.append(f"- random-init：test MAE {r2r['mae']}，R² {r2r['r2']}")
    line.append("")
    line.append("### R3 结构探针（无人工标签）")
    r3 = R.get("R3_structure_probes")
    if r3:
        for k, v in r3.items():
            line.append(f"- {k}：MAE {v['mae']}，R² {v['r2']}（mean baseline MAE {v['mean_baseline_mae']}）")
    line.append("")
    line.append("### R4 变换敏感性（frozen encoder）")
    r4 = R.get("R4_transforms")
    if r4:
        line.append(f"- 时间平移（+7s，同内容）：cosine {r4['time_shift_exact']['cosine_mean']}，"
                    f"L2 {r4['time_shift_exact']['l2_mean']}")
        line.append(f"- 时间平移（+1ms 量化稳健性）：cosine {r4['time_shift_1ms']['cosine_mean']}")
        line.append(f"- 窗口平移 0.5s（87.5% 重叠）：cosine {r4['window_shift_0.5s']['cosine_mean']}")
        gd = r4.get("global_perm_distances")
        if gd:
            line.append(f"- 全局 lane 置换（8 个固定置换，scratch 不动）：cosine "
                        f"{[round(g['cosine_mean'], 3) for g in gd]}；"
                        f"8-way frozen-probe 识别置换 test acc {r4['probe_perm_classification'].get('test_acc')} "
                        f"（chance {r4['probe_perm_classification'].get('chance')}）")
        line.append(f"- 局部 lane shuffle：cosine {r4['local_shuffle_distance']['cosine_mean']}；"
                    f"binary frozen-probe 判别 test acc {r4['probe_local_shuffle_binary'].get('test_acc')} "
                    f"（chance 0.5）")
        line.append(f"- random-init 参考：时间平移 cosine {r4['random_init_reference']['time_shift_exact']['cosine_mean']}；"
                    f"局部 shuffle cosine {r4['random_init_reference']['local_shuffle']['cosine_mean']}")
    line.append("")
    line.append("## 6. 与 random-init / 简单 baseline 的比较")
    line.append("见 R1（density baseline 与 random-init encoder）、R2（random-init 探针）、R3（random-init 探针）。")
    line.append("")
    line.append("## 7. 最重要的失败模式")
    line.append("- 大量空 cell：如果 onset precision/recall 明显偏离 0.5 阈值 F1 而 BCE 很低，说明模型在预测空/复制密度；")
    line.append("  span（连续遮挡）与 random-cell 的 BCE 差异能部分暴露这一捷径。")
    line.append("- 若 R2 无增益：局部结构可学但不对 SL 相关（有信息量的负结果）。")
    line.append("- 若 R4 对局部 shuffle 完全不敏感：编码器没有利用空间结构。")
    line.append("")
    line.append("## 8. 每个结果能证明 / 不能证明什么")
    line.append("- R1 只能证明：temporal-spatial 表示中存在可学习、可泛化的结构；不能证明学到了 skill demand。")
    line.append("- R2 只能证明：representation 保留了传统难度中的可线性读出信息；SL MAE 不是优化目标。")
    line.append("- R3 只能证明：latent 本身包含低阶结构信息（不只是 decoder 的局部捷径）。")
    line.append("- R4 只测量：latent 对三类变换的距离与可读性；全局置换距离小 ≠ 置换无关，距离大 ≠ 人类难度改变。")
    line.append("")
    line.append("## 9. 下一步最值得做的一个实验")
    line.append("由结果决定（首选：若 R1 结构可学且 R4 对局部 shuffle 敏感 → 多尺度/位置事件表示 + 结构任务消融；"
                "若 R2 无增益 → 先验证 cell size / 窗口长度再谈 skill demand）。")
    path = os.path.join(out_dir, "PHASE2A_REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(line) + "\n")
    print("report saved ->", path)


if __name__ == "__main__":
    main()
