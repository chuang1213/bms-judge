"""Pooled-only Task A 小规模实验。

唯一变化：Task A 的预测 head 只能读 pooled latent（+ 查询位置），不能读 per-cell feature map。
监督定义、数据表示、encoder、数据切分与上一轮完全一致。

先做 sanity（可学性 / val 不崩 / random-init 对照 / 无 leak），
再做表示级测试：统计受控 2-switch probe + 几何关系 frozen linear probe。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from .grid_data import N_LANES, T, SeqCache, seq_grid, window_items
from .intervention import grid_marginals_equal, swap_lanes_window, window_stats_seq
from .models import GridEncoder
from .run_phase2a import (
    build_pretrain_split, encode_grids, linear_probe_clf, linear_probe_reg,
    load_clean_records,
)
from .run_task_a_experiment import (
    batch_focal, build_focal_feature_matrix, build_task_data, focal_arrays,
    linear_clf_train_test, sample_songs,
)
from .task_a import HEAD_NAMES, N_CLASSES, PooledOnlyTaskAModel
from .train import make_masks_np, set_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "output", "phase2a")


def eval_pooled_model(model, input_grids, targets, device, batch=64):
    model.eval()
    offsets, flat, cls, flags = focal_arrays(targets)
    n_w = len(input_grids)
    preds = np.zeros((len(flat), len(HEAD_NAMES)), dtype=np.int64)
    t_arr = (flat // N_LANES).astype(np.float32) / T
    l_arr = (flat % N_LANES).astype(np.float32) / (N_LANES - 1)
    with torch.no_grad():
        for k in range(0, n_w, batch):
            w_ids = np.arange(k, min(k + batch, n_w))
            xb = torch.from_numpy(input_grids[w_ids]).permute(0, 3, 1, 2).to(device).float()
            pooled = model.pooled(xb)
            for pos, w in enumerate(w_ids):
                s, e = int(offsets[w]), int(offsets[w + 1])
                bpos = torch.full((e - s,), pos, dtype=torch.long, device=device)
                tt = torch.from_numpy(t_arr[s:e]).to(device)
                ll = torch.from_numpy(l_arr[s:e]).to(device)
                logits = model.forward_queries(pooled, bpos, tt, ll)
                for h, name in enumerate(HEAD_NAMES):
                    preds[s:e, h] = logits[name].argmax(1).cpu().numpy()
    acc = {name: float((preds[:, h] == cls[:, h]).mean()) for h, name in enumerate(HEAD_NAMES)}
    in_span = flags["in_span"]
    edge = flags["edge"]
    mask_type = np.where(in_span & ~edge, 0, np.where(in_span & edge, 1, 2))
    strata = {}
    for lbl, m in (("span_interior", 0), ("span_edge", 1), ("random_cell", 2)):
        sel = mask_type == m
        if sel.sum() == 0:
            strata[lbl] = None
            continue
        strata[lbl] = {name: float((preds[sel, h] == cls[sel, h]).mean())
                       for h, name in enumerate(HEAD_NAMES)}
        strata[lbl]["n"] = int(sel.sum())
    return acc, strata


def geometry_frozen_probe(enc, input_grids, targets, device, seed=0):
    """frozen pooled → 每窗口 focal note 的 mean lane_dist / direction / dt_prev / simult 线性回归。"""
    offsets, flat, cls, _ = focal_arrays(targets)
    n_w = len(input_grids)
    feats = encode_grids(enc, input_grids, device)
    means = {name: np.zeros(n_w, dtype=np.float32) for name in HEAD_NAMES}
    for w in range(n_w):
        s, e = int(offsets[w]), int(offsets[w + 1])
        for h, name in enumerate(HEAD_NAMES):
            sel = cls[s:e, h]
            if name in ("lane_dist",):
                valid = sel < 4          # 排除 no-prev
            elif name == "direction":
                valid = sel < 3
            elif name == "dt_prev":
                valid = sel < 5
            else:
                valid = np.ones(len(sel), dtype=bool)
            if valid.sum() > 0:
                means[name][w] = float(sel[valid].mean())
            else:
                means[name][w] = 0.0
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_w)
    n_tr, n_va = int(n_w * 0.8), int(n_w * 0.9)
    idx = {"train": perm[:n_tr], "val": perm[n_tr:n_va], "test": perm[n_va:]}
    out = {}
    for name in HEAD_NAMES:
        y = means[name]
        base = float(np.mean(np.abs(y[idx["test"]] - y[idx["train"]].mean())))
        r = linear_probe_reg(feats[idx["train"]], y[idx["train"]],
                             feats[idx["val"]], y[idx["val"]],
                             feats[idx["test"]], y[idx["test"]],
                             device, seed=seed, epochs=300, patience=30)
        r["mean_baseline_mae"] = round(base, 4)
        out[name] = r
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(ROOT, "output", "corpus", "manifest.jsonl"))
    ap.add_argument("--analysis", default=os.path.join(ROOT, "output", "corpus", "analysis"))
    ap.add_argument("--seq-root", default=os.path.join(ROOT, "output", "corpus", "sequences"))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--t1-ckpt", default=os.path.join(DEFAULT_OUT, "t1_hold.pt"))
    ap.add_argument("--prev-report", default=os.path.join(DEFAULT_OUT, "task_a_report.json"))
    ap.add_argument("--n-train-songs", type=int, default=100)
    ap.add_argument("--n-val-songs", type=int, default=20)
    ap.add_argument("--n-test-songs", type=int, default=30)
    ap.add_argument("--max-windows", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    report: dict = {"config": vars(args), "device": device}

    print("== 1. data（与上一轮完全一致）==")
    records = load_clean_records(args.manifest)
    sat_split = json.load(open(os.path.join(args.analysis, "split.json"), encoding="utf-8"))
    split, _ = build_pretrain_split(records, sat_split, seed=0)
    tr_records = sample_songs(split["train"], args.n_train_songs, args.seed)
    va_records = sample_songs(split["train"], args.n_val_songs, args.seed + 7)
    te_records = sample_songs(split["test"], args.n_test_songs, args.seed + 13)
    tr_items, tr_g, tr_in, tr_masks, tr_tgt = build_task_data(tr_records, args.seq_root,
                                                              args.max_windows, 0)
    va_items, va_g, va_in, va_masks, va_tgt = build_task_data(va_records, args.seq_root,
                                                              args.max_windows, 1)
    te_items, te_g, te_in, te_masks, te_tgt = build_task_data(te_records, args.seq_root,
                                                              args.max_windows, 2)
    report["data"] = {
        "train_windows": len(tr_items), "val_windows": len(va_items),
        "test_windows": len(te_items),
        "train_focal": int(sum(len(t["flat_idx"]) for t in tr_tgt)),
        "test_focal": int(sum(len(t["flat_idx"]) for t in te_tgt)),
    }
    print(report["data"])

    print("== 2. statistics-only baseline（与上一轮相同）==")
    Xtr_f, ytr_f = build_focal_feature_matrix(tr_in, tr_masks, tr_tgt)
    Xte_f, yte_f = build_focal_feature_matrix(te_in, te_masks, te_tgt)
    stats_base = {}
    for h, name in enumerate(HEAD_NAMES):
        acc = linear_clf_train_test(Xtr_f, ytr_f[:, h], Xte_f, yte_f[:, h],
                                    N_CLASSES[name], device, seed=args.seed)
        stats_base[name] = round(float(acc), 4)
    report["task_stats_baseline"] = stats_base
    print("stats baseline:", stats_base)

    print("== 3. random-init pooled-only head ==")
    set_seed(args.seed)
    rand_model = PooledOnlyTaskAModel(GridEncoder()).to(device)
    rand_acc, _ = eval_pooled_model(rand_model, te_in, te_tgt, device, args.batch)
    report["task_random_init"] = {k: round(v, 4) for k, v in rand_acc.items()}
    print("random-init:", report["task_random_init"])

    print("== 4. train pooled-only Task A ==")
    set_seed(args.seed)
    model = PooledOnlyTaskAModel(GridEncoder()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr_offsets, tr_flat, tr_cls, _ = focal_arrays(tr_tgt)
    va_offsets, va_flat, va_cls, _ = focal_arrays(va_tgt)
    n_w = len(tr_in)
    best, best_state, patience = -float("inf"), None, 0
    leak_checked = False
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(n_w)
        tot, cnt = 0.0, 0
        for k in range(0, n_w, args.batch):
            w_ids = np.sort(perm[k:k + args.batch])
            xb = torch.from_numpy(tr_in[w_ids]).permute(0, 3, 1, 2).to(device).float()
            if not leak_checked:
                masks_t = torch.from_numpy(tr_masks[w_ids]).to(device)
                leak = float((xb * (masks_t != 0).unsqueeze(1).float()).abs().max().item())
                assert leak == 0.0, f"mask leak: {leak}"
                leak_checked = True
            pooled = model.pooled(xb)
            bpos, flat_b, cls_b = batch_focal(tr_offsets, tr_flat, tr_cls, w_ids)
            if len(bpos) == 0:
                continue
            bp = torch.from_numpy(bpos).to(device)
            tt = torch.from_numpy((flat_b // N_LANES).astype(np.float32) / T).to(device)
            ll = torch.from_numpy((flat_b % N_LANES).astype(np.float32)
                                  / (N_LANES - 1)).to(device)
            logits = model.forward_queries(pooled, bp, tt, ll)
            loss = 0.0
            for h, name in enumerate(HEAD_NAMES):
                loss = loss + F.cross_entropy(
                    logits[name], torch.from_numpy(cls_b[:, h]).to(device))
            loss = loss / len(HEAD_NAMES)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()) * len(bpos); cnt += len(bpos)
        model.eval()
        va_acc, _ = eval_pooled_model(model, va_in, va_tgt, device, args.batch)
        va_mean = float(np.mean(list(va_acc.values())))
        print(f"  epoch {epoch + 1}/{args.epochs}: train_loss {tot / max(cnt, 1):.4f} "
              f"val_mean_acc {va_mean:.4f} (best {best:.4f})", flush=True)
        if va_mean > best + 1e-5:
            best, patience = va_mean, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.out, "task_a_pooled.pt"))
    report["best_val_mean_acc"] = round(float(best), 4)

    print("== 5. pooled-only Task A test ==")
    test_acc, strata = eval_pooled_model(model, te_in, te_tgt, device, args.batch)
    report["task_trained"] = {k: round(v, 4) for k, v in test_acc.items()}
    report["task_strata"] = {
        k: ({kk: round(vv, 4) for kk, vv in v.items()} if v else None)
        for k, v in strata.items()
    }
    print("trained:", report["task_trained"])
    print("strata:", json.dumps(report["task_strata"], ensure_ascii=False))

    print("== 6. 表示级测试 ==")
    from .run_task_a_experiment import run_perm_probe
    probe = run_perm_probe(model.encoder, args, device)
    report["probe"] = probe
    print(json.dumps(probe, ensure_ascii=False, indent=2))

    geo = {}
    geo["pooled_only_pretrained"] = geometry_frozen_probe(model.encoder, te_in, te_tgt,
                                                          device, args.seed)
    set_seed(args.seed)
    geo["random_init"] = geometry_frozen_probe(GridEncoder(), te_in, te_tgt,
                                               device, args.seed)
    report["geometry_probe"] = geo
    print("geometry probe:", json.dumps(geo, ensure_ascii=False, indent=2))

    # 上一轮 per-cell Task A 对照（若存在）
    if os.path.exists(args.prev_report):
        prev = json.load(open(args.prev_report, encoding="utf-8"))
        report["prev_per_cell_taskA"] = {
            "task_trained": prev.get("task_trained"),
            "probe": prev.get("probe"),
        }

    report["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "task_a_pooled_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_md(report, os.path.join(os.path.dirname(ROOT), "PHASE2A_POOLED_ONLY_REPORT.md"))
    print("saved ->", os.path.join(args.out, "task_a_pooled_report.json"))


def write_md(R: dict, path: str) -> None:
    L = []
    L.append("# Phase 2A Pooled-only Task A 小规模实验报告")
    L.append("")
    L.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} | 运行 {R.get('wall_seconds', '?')}s")
    L.append("")
    L.append("## 0. 设计")
    L.append("input window → encoder → pooled latent → geometry prediction heads")
    L.append("head 输入 = concat(pooled, 查询位置 (t/T, l/7))；无 per-cell feature map 旁路。")
    L.append("监督定义、数据表示、encoder、数据切分与上一轮 per-cell Task A 完全一致。")
    L.append("")
    L.append("## 1. 数据")
    d = R.get("data", {})
    L.append(f"- train {d.get('train_windows')} 窗口 / {d.get('train_focal')} focal；"
             f"test {d.get('test_windows')} 窗口 / {d.get('test_focal')} focal；1 seed")
    L.append("")
    L.append("## 2. sanity：pooled-only 下 Task A 是否可学")
    if R.get("task_stats_baseline"):
        L.append(f"- statistics-only baseline：{R['task_stats_baseline']}")
    if R.get("task_random_init"):
        L.append(f"- random-init pooled-only head：{R['task_random_init']}")
    if R.get("task_trained"):
        L.append(f"- pooled-only trained：{R['task_trained']}")
    if R.get("best_val_mean_acc"):
        L.append(f"- best val mean acc：{R['best_val_mean_acc']}（未崩溃）")
    L.append("- mask leak 检查：训练首 batch 断言通过（被遮挡 cell 在输入中确实为 0）。")
    L.append("")
    L.append("## 3. 表示级测试 1：统计受控 2-switch probe")
    p = R.get("probe", {})
    for k in ("taskA_pretrained", "T1_pretrained", "random_init", "S0", "S1"):
        if p.get(k):
            L.append(f"- {k}：{p[k]}")
    L.append("")
    L.append("## 4. 表示级测试 2：几何关系 frozen linear probe（pooled → mean lane_dist/direction/dt/simult）")
    g = R.get("geometry_probe", {})
    for who, block in g.items():
        if block:
            L.append(f"- {who}：")
            for name, r in block.items():
                L.append(f"  - {name}: MAE {r['mae']} R² {r['r2']}（mean baseline MAE {r['mean_baseline_mae']}）")
    L.append("")
    L.append("## 5. 与上一轮 per-cell Task A 的对照")
    prev = R.get("prev_per_cell_taskA")
    if prev:
        L.append(f"- per-cell trained acc：{prev.get('task_trained')}")
        pp = prev.get("probe", {})
        if pp:
            L.append(f"- per-cell 2-switch probe：taskA {pp.get('taskA_pretrained', {}).get('acc_mean')} "
                     f"/ T1 {pp.get('T1_pretrained', {}).get('acc_mean')} "
                     f"/ random {pp.get('random_init', {}).get('acc_mean')}")
    L.append("")
    L.append("## 6. 结论（由数字决定）")
    L.append("核心判据：pretrained pooled 是否明显超过 random-init（random-init ≈ pretrained 是否仍成立）。")
    L.append("")
    L.append("## 7. 失败时的三个候选解释")
    L.append("- pooled representation 过度压缩（64 维不足以承载 per-note 几何）；")
    L.append("- Task A 仍可走统计捷径（density → lane-dist 分布），pooled 里学的只是统计；")
    L.append("- 4s 单窗口不是合适的 representation 粒度（几何关系需要更长时间上下文）。")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
