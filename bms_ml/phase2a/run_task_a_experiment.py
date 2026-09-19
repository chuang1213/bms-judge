"""Task A（局部几何关系预测）小规模实验。

问题：Task A 是否提供比 T1 masked reconstruction 更强的"结构学习压力"？

测量：
  1. 任务本身可学性（test 准确率 vs random-init head / statistics-only baseline）；
  2. frozen pooled 表示的排列敏感度（统计受控 2-switch 判别，对比 T1-pretrained / random-init）；
  3. 与 T1 的对照（同 encoder 架构、同 probe 协议）。

不做全量：几十到几百首歌、1 seed、小模型、少量 epoch。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..split import song_group_key
from .grid_data import N_LANES, T, SeqCache, seq_grid, window_items
from .intervention import grid_marginals_equal, swap_lanes_window, window_stats_seq
from .models import GridEncoder
from .run_phase2a import (
    build_pretrain_split, encode_grids, linear_probe_clf, load_clean_records,
)
from .task_a import HEAD_NAMES, N_CLASSES, TaskAModel, extract_focal_targets, focal_stats_features
from .train import make_masks_np, set_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "output", "phase2a")


def sample_songs(split_records, n_songs: int, seed: int):
    groups: dict = defaultdict(list)
    for r in split_records:
        groups[song_group_key(r)].append(r)
    gids = sorted(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(gids)
    out = []
    for g in gids[:n_songs]:
        out.extend(groups[g])
    return out


def build_task_data(records, seq_root, max_windows, mask_seed):
    items = window_items(records, seq_root, max_windows=max_windows, seed=mask_seed)
    cache = SeqCache(seq_root)
    grids = np.stack([seq_grid(cache(sha), st) for sha, st in items]).astype(np.uint8)
    masks = make_masks_np(len(grids), seed=mask_seed)
    masked_zeros = (masks == 0)[:, :, :, None].astype(np.uint8)
    input_grids = grids * masked_zeros
    targets = [extract_focal_targets(grids[i], input_grids[i], masks[i])
               for i in range(len(grids))]
    return items, grids, input_grids, masks, targets


def focal_arrays(targets):
    offsets = np.concatenate([[0], np.cumsum([len(t["flat_idx"]) for t in targets])]).astype(int)
    flat = np.concatenate([t["flat_idx"] for t in targets])
    cls = np.concatenate([t["cls"] for t in targets], axis=0)
    flags = {
        "in_span": np.concatenate([t["in_span"] for t in targets]),
        "edge": np.concatenate([t["edge"] for t in targets]),
        "span_frac": np.concatenate([t["span_frac"] for t in targets]),
    }
    return offsets, flat, cls, flags


def batch_focal(offsets, flat, cls, w_ids):
    n = 0
    for w in w_ids:
        n += offsets[w + 1] - offsets[w]
    bpos = np.zeros(n, dtype=np.int64)
    flat_b = np.zeros(n, dtype=np.int64)
    cls_b = np.zeros((n, len(HEAD_NAMES)), dtype=np.int64)
    p = 0
    for k, w in enumerate(w_ids):
        s, e = int(offsets[w]), int(offsets[w + 1])
        ln = e - s
        bpos[p:p + ln] = k
        flat_b[p:p + ln] = flat[s:e]
        cls_b[p:p + ln] = cls[s:e]
        p += ln
    return bpos, flat_b, cls_b


def eval_task_model(model, input_grids, targets, device, batch=64):
    model.eval()
    offsets, flat, cls, flags = focal_arrays(targets)
    n_w = len(input_grids)
    preds = np.zeros((len(flat), len(HEAD_NAMES)), dtype=np.int64)
    with torch.no_grad():
        for k in range(0, n_w, batch):
            w_ids = np.arange(k, min(k + batch, n_w))
            xb = torch.from_numpy(input_grids[w_ids]).permute(0, 3, 1, 2).to(device).float()
            logits = model.cell_logits(xb)
            for pos, w in enumerate(w_ids):
                s, e = int(offsets[w]), int(offsets[w + 1])
                for h, name in enumerate(HEAD_NAMES):
                    lg = logits[name][pos].reshape(N_CLASSES[name], T * N_LANES)
                    preds[s:e, h] = lg[:, flat[s:e]].argmax(0).cpu().numpy()
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


def linear_clf_train_test(Xtr, ytr, Xte, yte, n_class, device, seed=0,
                          epochs=200, lr=1e-2):
    set_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtrn = (Xtr - mu) / sd
    Xten = (Xte - mu) / sd
    model = nn.Linear(Xtr.shape[1], n_class).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t = torch.from_numpy
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(t(Xtrn), t(ytr)), batch_size=256, shuffle=True)
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
        logits = model(t(Xten).to(device))
        acc = float((logits.argmax(1).cpu().numpy() == yte).mean())
    return acc


def build_focal_feature_matrix(input_grids, masks, targets):
    X_list, y_list = [], []
    for i, tg in enumerate(targets):
        ts = tg["flat_idx"] // N_LANES
        ls = tg["flat_idx"] % N_LANES
        for t, l in zip(ts, ls):
            X_list.append(focal_stats_features(input_grids[i], masks[i], int(t), int(l)))
        y_list.append(tg["cls"])
    return np.stack(X_list), np.concatenate(y_list, axis=0)


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
    ap.add_argument("--n-train-songs", type=int, default=100)
    ap.add_argument("--n-val-songs", type=int, default=20)
    ap.add_argument("--n-test-songs", type=int, default=30)
    ap.add_argument("--max-windows", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    report: dict = {"config": vars(args), "device": device}

    print("== 1. data ==")
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

    print("== 2. statistics-only baseline ==")
    Xtr_f, ytr_f = build_focal_feature_matrix(tr_in, tr_masks, tr_tgt)
    Xte_f, yte_f = build_focal_feature_matrix(te_in, te_masks, te_tgt)
    stats_base = {}
    for h, name in enumerate(HEAD_NAMES):
        acc = linear_clf_train_test(Xtr_f, ytr_f[:, h], Xte_f, yte_f[:, h],
                                    N_CLASSES[name], device, seed=args.seed)
        stats_base[name] = round(float(acc), 4)
    report["task_stats_baseline"] = stats_base
    print("stats baseline:", stats_base)

    print("== 3. random-init head accuracy ==")
    set_seed(args.seed)
    rand_model = TaskAModel(GridEncoder()).to(device)
    rand_acc, _ = eval_task_model(rand_model, te_in, te_tgt, device, args.batch)
    report["task_random_init"] = {k: round(v, 4) for k, v in rand_acc.items()}
    print("random-init:", report["task_random_init"])

    print("== 4. train Task A ==")
    set_seed(args.seed)
    model = TaskAModel(GridEncoder()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tr_offsets, tr_flat, tr_cls, _ = focal_arrays(tr_tgt)
    va_offsets, va_flat, va_cls, _ = focal_arrays(va_tgt)
    n_w = len(tr_in)
    best, best_state, patience = -float("inf"), None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(n_w)
        tot, cnt = 0.0, 0
        for k in range(0, n_w, args.batch):
            w_ids = np.sort(perm[k:k + args.batch])
            xb = torch.from_numpy(tr_in[w_ids]).permute(0, 3, 1, 2).to(device).float()
            logits = model.cell_logits(xb)
            bpos, flat_b, cls_b = batch_focal(tr_offsets, tr_flat, tr_cls, w_ids)
            if len(bpos) == 0:
                continue
            loss = 0.0
            for h, name in enumerate(HEAD_NAMES):
                lg = logits[name].reshape(len(w_ids), N_CLASSES[name], T * N_LANES)
                sel = lg[bpos, :, flat_b]
                loss = loss + F.cross_entropy(sel, torch.from_numpy(cls_b[:, h]).to(device))
            loss = loss / len(HEAD_NAMES)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()) * len(bpos); cnt += len(bpos)
        model.eval()
        va_acc, _ = eval_task_model(model, va_in, va_tgt, device, args.batch)
        va_mean = float(np.mean(list(va_acc.values())))
        print(f"  epoch {epoch + 1}/{args.epochs}: train_loss {tot / max(cnt, 1):.4f} "
              f"val_mean_acc {va_mean:.4f} (best {best:.4f})", flush=True)
        if va_mean > best + 1e-5:
            best, patience = va_mean, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 4:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.out, "task_a.pt"))

    print("== 5. Task A test ==")
    test_acc, strata = eval_task_model(model, te_in, te_tgt, device, args.batch)
    report["task_trained"] = {k: round(v, 4) for k, v in test_acc.items()}
    report["task_strata"] = {
        k: ({kk: round(vv, 4) for kk, vv in v.items()} if v else None)
        for k, v in strata.items()
    }
    print("trained:", report["task_trained"])
    print("strata:", json.dumps(report["task_strata"], ensure_ascii=False))

    print("== 6. frozen probe: controlled permutation discrimination ==")
    probe = run_perm_probe(model.encoder, args, device)
    report["probe"] = probe
    print(json.dumps(probe, ensure_ascii=False, indent=2))

    report["wall_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.out, "task_a_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_md(report, os.path.join(os.path.dirname(ROOT), "PHASE2A_TASK_COMPARISON_REPORT.md"))
    print("saved ->", os.path.join(args.out, "task_a_report.json"))


def run_perm_probe(enc_a, args, device) -> dict:
    records = load_clean_records(args.manifest)
    sat_split = json.load(open(os.path.join(args.analysis, "split.json"), encoding="utf-8"))
    split, _ = build_pretrain_split(records, sat_split, seed=0)
    te_records = sample_songs(split["test"], 20, 99)
    items = window_items(te_records, args.seq_root, max_windows=12, seed=99)
    rng = np.random.RandomState(7)
    cache = SeqCache(args.seq_root)
    orig, var, accepted = [], [], []
    for sha, st in items:
        seq = cache(sha)
        g1 = seq_grid(seq, st)
        vseq, _ = swap_lanes_window(seq, st, rng)
        g2 = seq_grid(vseq, st)
        if grid_marginals_equal(g1, g2):
            orig.append(g1); var.append(g2)
            accepted.append((sha, st))
        if len(orig) >= 400:
            break
    orig = np.stack(orig); var = np.stack(var)
    n = len(orig)
    probe = {}
    encs = {
        "taskA_pretrained": enc_a,
        "T1_pretrained": GridEncoder(),
        "random_init": GridEncoder(),
    }
    set_seed(args.seed)
    encs["T1_pretrained"].load_state_dict(torch.load(args.t1_ckpt, map_location=device))
    set_seed(args.seed)
    for name, enc in encs.items():
        X = np.concatenate([encode_grids(enc, orig, device),
                            encode_grids(enc, var, device)], axis=0)
        y = np.concatenate([np.zeros(n), np.ones(n)])
        gids = np.concatenate([np.arange(n), np.arange(n)])
        accs = []
        for s in (0, 1, 2):
            r = linear_probe_clf(X, y, 2, device, gids, seed=s)
            accs.append(r["test_acc"])
        probe[name] = {"acc_mean": round(float(np.mean(accs)), 4),
                       "acc_min": round(float(np.min(accs)), 4),
                       "acc_max": round(float(np.max(accs)), 4),
                       "n_test": int(r["n_test"])}
    so = [window_stats_seq(cache(sha), st) for sha, st in accepted]
    sv = []
    for i in range(n):
        sha, st = accepted[i]
        vseq, _ = swap_lanes_window(cache(sha), st, np.random.RandomState(3 + i))
        sv.append(window_stats_seq(vseq, st))

    def vec(s, extra):
        base = np.concatenate([np.asarray([s["note_count"], s["nps"]]),
                               s["per_lane"], s["chord_hist"], s["density_profile"]])
        if extra:
            base = np.concatenate([base, [s["adjacent_frac"], s["max_run"],
                                          s["span_mean"], s["span_std"]]])
        return base.astype(np.float32)
    for label, extra in (("S0", False), ("S1", True)):
        X = np.concatenate([np.stack([vec(s, extra) for s in so]),
                            np.stack([vec(s, extra) for s in sv])], axis=0)
        y = np.concatenate([np.zeros(n), np.ones(n)])
        gids = np.concatenate([np.arange(n), np.arange(n)])
        r = linear_probe_clf(X, y, 2, device, gids, seed=0)
        probe[label] = {"acc": r["test_acc"]}
    return probe


def write_md(R: dict, path: str) -> None:
    L = []
    L.append("# Phase 2A 候选 pretext task 比较 + Task A 小规模实验")
    L.append("")
    L.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} | 运行 {R.get('wall_seconds', '?')}s")
    L.append("")
    L.append("## 1. 三个候选任务")
    L.append("### A. 局部几何关系预测（本轮实现）")
    L.append("- 输入：窗口网格 + 被遮挡 span/随机 cell（与 T1 相同）；")
    L.append("- target：被遮挡真实 note 与最近可见前序 note 的客观几何关系：")
    L.append("  lane distance（0/1/2/3+）、相对移动方向（-1/0/+1）、相对时间桶、是否 chord（simultaneity）——全部自动生成；")
    L.append("- 学到什么：必须对具体 lane 位置与时间关系作出承诺（不是'有没有 note'）；")
    L.append("- 最容易的 shortcut：类别先验 / 从 span 边界复制 / 密度→lane 距离分布；")
    L.append("- 支持'理解结构'的结果：test 准确率显著高于 stats baseline 与 random-init，且 interior 样本仍有收益；")
    L.append("- 只说明'任务可预测'的结果：准确率高但 frozen probe 无增益（decoder 型学习，表示未受益）。")
    L.append("")
    L.append("### B. 时间结构重排")
    L.append("- 输入：一个窗口切成 K 段并打乱；target：正确顺序（或 pairwise order）；")
    L.append("- 学到什么：段间 temporal 连续性 / 乐句结构；")
    L.append("- 退化风险：很多谱面的段顺序本身不唯一（低 ceiling）；模型可能只学'边界 gap 更平滑'的局部统计；")
    L.append("- 判定标准：与 A 相同，但先要排除 ambiguity（需要额外分析正确顺序是否可恢复）。")
    L.append("")
    L.append("### C. 结构一致性 / matching")
    L.append("- 输入：两个窗口，其中一个来自另一个的已知程序化变换；target：匹配/关系；")
    L.append("- 学到什么：跨窗口内容相似性；")
    L.append("- 风险：上一轮已证明变换判别在 random features 上就可读（83%），且我们还没有正确的不变性集合，"
             "容易把'输入像素可读'误当成'结构理解'。")
    L.append("")
    L.append("## 2. 推荐：A，理由")
    L.append("1. A 直接修复 T1 的已知失败模式：T1 损失被空 cell 主导（模型靠预测空拿低 loss，recall 仅 0.066）；"
             "A 的监督只落在真实 note 上，无法用'预测空'偷分；")
    L.append("2. target 是几何关系（lane distance / 方向 / 相对时间 / 同时性），迫使模型对位置作出承诺，"
             "且全部自动生成；")
    L.append("3. B 有 order-ambiguity 低 ceiling 风险，C 已被证明在随机特征上可读——A 是三者中"
             "'任务难、信号干净、可判别'平衡最好的。")
    L.append("")
    L.append("## 3. 小规模实验结果")
    d = R.get("data", {})
    L.append(f"- 数据：train {d.get('train_windows')} 窗口 / {d.get('train_focal')} focal notes；"
             f"test {d.get('test_windows')} 窗口 / {d.get('test_focal')} focal notes")
    L.append(f"- 任务准确率（test，4 heads 均值）：")
    if R.get("task_stats_baseline"):
        L.append(f"  - statistics-only baseline：{R['task_stats_baseline']}")
    if R.get("task_random_init"):
        L.append(f"  - random-init head：{R['task_random_init']}")
    if R.get("task_trained"):
        L.append(f"  - Task A trained：{R['task_trained']}")
    if R.get("task_strata"):
        L.append(f"  - 分层（interior/edge/random）：{R['task_strata']}")
    L.append("")
    L.append("## 4. 与 T1 的比较（frozen pooled 表示的排列敏感度）")
    p = R.get("probe", {})
    if p.get("taskA_pretrained"):
        L.append(f"- Task A pretrained：{p['taskA_pretrained']['acc_mean']}")
    if p.get("T1_pretrained"):
        L.append(f"- T1 pretrained：{p['T1_pretrained']['acc_mean']}")
    if p.get("random_init"):
        L.append(f"- random-init：{p['random_init']['acc_mean']}")
    if p.get("S0"):
        L.append(f"- S0（保持的统计）：{p['S0']['acc']}；S1（+空间聚合）：{p['S1']['acc']}")
    L.append("")
    L.append("## 5. 结果能证明什么 / 不能证明什么")
    L.append("- 能证明：如果 Task A 的 frozen probe 明显高于 T1 与 random-init，说明该任务确实给"
             "pooled 表示施加了比 T1 更强的结构压力；任务准确率高于 stats baseline 说明监督信号"
             "不能只用低阶统计解出。")
    L.append("- 不能证明：几何关系预测 ≠ skill demand；也不能把 probe 增益直接解释为'学会了 stair/jack'。")
    L.append("")
    L.append("## 6. 失败最可能的原因（如果失败）")
    L.append("- 类别先验主导（多数 focal 的 lane distance=1/2，方向=0）：训练陷入 prior，需先平衡采样；")
    L.append("- span 边界复制：模型只在边界样本上得分，interior 无增益；")
    L.append("- 头部吃 local features，pooled 表示仍然懒惰：需要把任务头改为只用 pooled 或加 pooled-only 探针；")
    L.append("- 样本太少 / epoch 太少，任务未收敛。")
    L.append("")
    L.append("## 7. 是否值得全量")
    L.append("（由数字决定：Task A 在 probe 上明显超过 T1/random 且任务本身可学 → 值得；否则停止并修设计。）")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
