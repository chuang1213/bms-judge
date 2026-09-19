"""统计受控干预实验：小规模 audit + frozen probe + statistics-only baseline。

流程（严格按门控）：
  1. 从 pretrain test（未见歌曲）抽样几十首歌；
  2. 对每个窗口生成 2-switch 变体（统计受控）；
  3. audit：验证理论不变量的实际差异为 0，并报告空间聚合量的变化；
  4. audit 通过后：frozen probe（pretrained vs random-init）+ statistics-only baseline；
  5. 写 intervention_report.json 与 PHASE2A_INTERVENTION_REPORT.md。

用法：
  .venv\\Scripts\\python.exe -u -m bms_ml.phase2a.run_intervention_probe
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

from ..split import song_group_key
from .grid_data import KEY_LANES, N_LANES, SeqCache, WINDOW_SEC, seq_grid, window_items
from .intervention import (
    arrangement_delta, grid_marginals_equal, is_global_perm,
    preserved_stats_diff, swap_lanes_window, tap_hamming, window_stats_seq,
)
from .models import GridEncoder
from .run_phase2a import (
    build_pretrain_split, encode_grids, linear_probe_clf, load_clean_records,
)
from .train import set_seed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "output", "phase2a")


def sample_test_records(records, sat_split, n_songs: int, seed: int):
    split, _ = build_pretrain_split(records, sat_split, seed=0)
    test = split["test"]
    groups: dict = defaultdict(list)
    for r in test:
        groups[song_group_key(r)].append(r)
    gids = sorted(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(gids)
    chosen = gids[:n_songs]
    out = []
    for g in chosen:
        out.extend(groups[g])
    return out


def audit_items(items, seq_root, seed: int, swaps_factor: float):
    cache = SeqCache(seq_root)
    rng = np.random.RandomState(seed)
    rows = []
    n_failed = 0
    for sha, start in items:
        seq = cache(sha)
        g_orig = seq_grid(seq, start)
        variant = seq
        done = 0
        for _attempt in range(4):
            variant, done = swap_lanes_window(seq, start, rng,
                                              swaps_factor=swaps_factor)
            g_var = seq_grid(variant, start)
            if grid_marginals_equal(g_orig, g_var):
                break
        else:
            n_failed += 1
            variant, done = seq.copy(), 0
            g_var = g_orig.copy()
        so = window_stats_seq(seq, start)
        sv = window_stats_seq(variant, start)
        rows.append({
            "sha": sha, "start": start,
            "swaps_done": done,
            "hamming": tap_hamming(seq, variant, start),
            "global_perm_like": is_global_perm(seq, variant, start),
            "preserved": preserved_stats_diff(so, sv),
            "changed": arrangement_delta(so, sv),
            "note_count": so["note_count"],
        })
    return rows, n_failed


def path_get(r, path):
    if isinstance(path, str):
        path = (path,)
    for k in path:
        r = r[k]
    return r


def max_of(rows, path):
    vals = [path_get(r, path) for r in rows]
    return float(max(vals)) if vals else 0.0


def mean_of(rows, path):
    vals = [path_get(r, path) for r in rows]
    return float(np.mean(vals)) if vals else 0.0


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
    ap.add_argument("--ckpt", default=os.path.join(DEFAULT_OUT, "t1_hold.pt"))
    ap.add_argument("--n-songs", type=int, default=80)
    ap.add_argument("--max-windows", type=int, default=12)
    ap.add_argument("--swaps-factor", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe-seeds", default="0,1,2")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="audit 未通过时仍继续 probe")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0_wall = time.time()

    report: dict = {"config": vars(args), "device": device}

    # ---- 1. 抽样（pretrain test 歌曲，未见） ----
    print("== 1. sample songs (held-out) ==")
    records = load_clean_records(args.manifest)
    sat_split = json.load(open(os.path.join(args.analysis, "split.json"), encoding="utf-8"))
    sample_records = sample_test_records(records, sat_split, args.n_songs, args.seed)
    items = window_items(sample_records, args.seq_root,
                         max_windows=args.max_windows, seed=args.seed)
    print(f"  songs={args.n_songs} charts={len(sample_records)} windows={len(items)}")
    report["sample"] = {"n_songs": args.n_songs, "n_charts": len(sample_records),
                        "n_windows": len(items)}

    # ---- 2. 生成变体 + audit ----
    print("== 2. intervention + audit ==")
    rows, n_failed = audit_items(items, args.seq_root, args.seed, args.swaps_factor)
    audit = {
        "n_windows": len(rows),
        "n_marginal_fail_after_retry": n_failed,
        "swaps_mean": round(mean_of(rows, "swaps_done"), 2),
        "hamming_mean": round(mean_of(rows, "hamming"), 4),
        "hamming_median": round(float(np.median([r["hamming"] for r in rows])), 4),
        "noop_frac": round(float(np.mean([r["hamming"] == 0 for r in rows])), 4),
        "global_perm_like_frac": round(float(np.mean([r["global_perm_like"] for r in rows])), 4),
        "preserved_max_diff": {k: max_of(rows, ("preserved", k)) for k in
                               ("note_count", "per_lane", "chord_hist",
                                "density_profile", "nps")},
        "changed_mean_abs": {k: mean_of(rows, ("changed", k)) for k in
                             ("adjacent_frac_delta", "max_run_delta",
                              "span_mean_delta", "span_std_delta")},
    }
    report["audit"] = audit
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    audit_pass = (
        n_failed == 0
        and all(v == 0.0 for v in audit["preserved_max_diff"].values())
        and audit["hamming_mean"] > 0.03
        and audit["global_perm_like_frac"] < 0.5
    )
    report["audit_pass"] = audit_pass
    print("audit verdict:", "PASS" if audit_pass else "FAIL")
    if not audit_pass and not args.force:
        print("audit 未通过：先修改 intervention，不进入 probe。")
        _finish(report, args.out)
        return

    if args.audit_only:
        print("--audit-only：停止。")
        _finish(report, args.out)
        return

    # ---- 3. 构建原始/变体网格，跑 probe ----
    print("== 3. frozen probes ==")
    cache = SeqCache(args.seq_root)
    orig_grids, var_grids = [], []
    so_list, sv_list = [], []
    rng = np.random.RandomState(args.seed + 1)
    for sha, start in items:
        seq = cache(sha)
        variant, _ = swap_lanes_window(seq, start, rng, swaps_factor=args.swaps_factor)
        orig_grids.append(seq_grid(seq, start))
        var_grids.append(seq_grid(variant, start))
        so_list.append(window_stats_seq(seq, start))
        sv_list.append(window_stats_seq(variant, start))
    orig_grids = np.stack(orig_grids)
    var_grids = np.stack(var_grids)
    n_w = len(orig_grids)

    probe_seeds = [int(s) for s in args.probe_seeds.split(",")]

    def run_binary_probe(X: np.ndarray, label: str) -> dict:
        y = np.concatenate([np.zeros(n_w), np.ones(n_w)])
        gids = np.concatenate([np.arange(n_w), np.arange(n_w)])
        accs = []
        for s in probe_seeds:
            r = linear_probe_clf(X, y, 2, device, gids, seed=s)
            accs.append(r["test_acc"])
        return {"label": label, "acc_mean": round(float(np.mean(accs)), 4),
                "acc_min": round(float(np.min(accs)), 4),
                "acc_max": round(float(np.max(accs)), 4),
                "chance": 0.5, "n_train": int(r["n_train"]), "n_test": int(r["n_test"])}

    probe = {}
    # pretrained encoder
    enc = GridEncoder()
    enc.load_state_dict(torch.load(args.ckpt, map_location=device))
    Xp = np.concatenate([encode_grids(enc, orig_grids, device),
                         encode_grids(enc, var_grids, device)], axis=0)
    probe["pretrained_encoder"] = run_binary_probe(Xp, "pretrained")
    print("pretrained:", probe["pretrained_encoder"])
    # random-init encoder
    set_seed(args.seed)
    rand_enc = GridEncoder()
    Xr = np.concatenate([encode_grids(rand_enc, orig_grids, device),
                         encode_grids(rand_enc, var_grids, device)], axis=0)
    probe["random_init_encoder"] = run_binary_probe(Xr, "random_init")
    print("random-init:", probe["random_init_encoder"])

    # ---- 4. statistics-only baseline ----
    print("== 4. statistics-only baseline ==")
    def stat_vec(s: dict) -> np.ndarray:
        return np.concatenate([
            np.asarray([s["note_count"], s["nps"]], dtype=np.float32),
            np.asarray(s["per_lane"], dtype=np.float32),
            np.asarray(s["chord_hist"], dtype=np.float32),
            np.asarray(s["density_profile"], dtype=np.float32),
        ])
    def stat_vec_s1(s: dict) -> np.ndarray:
        return np.concatenate([
            stat_vec(s),
            np.asarray([s["adjacent_frac"], s["max_run"],
                        s["span_mean"], s["span_std"]], dtype=np.float32),
        ])
    Xs0 = np.concatenate([np.stack([stat_vec(s) for s in so_list]),
                          np.stack([stat_vec(s) for s in sv_list])], axis=0)
    Xs1 = np.concatenate([np.stack([stat_vec_s1(s) for s in so_list]),
                          np.stack([stat_vec_s1(s) for s in sv_list])], axis=0)
    probe["statistics_only_S0"] = run_binary_probe(Xs0, "S0 preserved marginals")
    probe["statistics_plus_aggregates_S1"] = run_binary_probe(Xs1, "S1 + spatial aggregates")
    print("S0:", probe["statistics_only_S0"])
    print("S1:", probe["statistics_plus_aggregates_S1"])

    report["probe"] = probe
    report["wall_seconds"] = round(time.time() - t0_wall, 1)
    _finish(report, args.out)


def _finish(report: dict, out_dir: str) -> None:
    with open(os.path.join(out_dir, "intervention_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_md(report, os.path.join(out_dir, "PHASE2A_INTERVENTION_REPORT.md"))
    write_md(report, os.path.join(os.path.dirname(ROOT), "PHASE2A_INTERVENTION_REPORT.md"))
    print("report saved ->", os.path.join(out_dir, "intervention_report.json"))


def write_md(R: dict, path: str) -> None:
    a = R.get("audit", {})
    p = R.get("probe", {})
    L = []
    L.append("# Phase 2A 统计受控干预实验报告（小规模）")
    L.append("")
    L.append(f"> 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} | 运行 {R.get('wall_seconds', '?')}s")
    L.append("")
    L.append("## 1. intervention 的精确定义")
    L.append("- 窗口内 tap（type==0，key lanes 0-6）随机 2-switch：`(t1,l1),(t2,l2) -> (t1,l2),(t2,l1)`")
    L.append(f"- 每窗口交换次数 ≈ swaps_factor × tap 数（本实验 swaps_factor={R['config']['swaps_factor']}）")
    L.append("- LN（type==1）与 scratch（lane==7）不参与交换（理由：避免 hold 通道与 scratch 统计被改变）")
    L.append("- 不是简单随机 permutation：2-switch 是保持 chord size 与 per-lane totals 同时成立的最小组原语")
    L.append("")
    L.append("## 2. 理论上保持不变的统计量")
    L.append("- note 总数；每时间位置 note 数（chord size 分布）；每条 lane 总 note 数；")
    L.append("  时间顺序；onset/duration；局部 density（任何时间分桶）；NPS")
    L.append("")
    L.append("## 3. 实际 audit 结果")
    L.append(f"- 样本：{R['sample']['n_songs']} 首歌 / {R['sample']['n_charts']} 谱面 / "
             f"{R['sample']['n_windows']} 窗口（全部来自 pretrain test，未见歌曲）")
    L.append(f"- 交换数均值 {a.get('swaps_mean')}；hamming（tap lane 改变占比）均值 {a.get('hamming_mean')}，"
             f"中位 {a.get('hamming_median')}；no-op 窗口占比 {a.get('noop_frac')}")
    L.append(f"- 全局置换相似窗口占比 {a.get('global_perm_like_frac')}（接近 0 说明确实破坏了局部结构，"
             f"而非整体重贴标签）")
    L.append(f"- 理论不变量的最大绝对差：{a.get('preserved_max_diff')}")
    L.append(f"- 被改变的空间聚合量（confound 记录）：{a.get('changed_mean_abs')}")
    L.append(f"- audit 判定：{'PASS' if R.get('audit_pass') else 'FAIL'}")
    L.append("")
    L.append("## 4. 小规模 pretrained vs random-init")
    if p.get("pretrained_encoder"):
        L.append(f"- pretrained（Phase 2A T1-hold）：test acc {p['pretrained_encoder']['acc_mean']} "
                 f"({p['pretrained_encoder']['acc_min']}–{p['pretrained_encoder']['acc_max']}，chance 0.5)")
    if p.get("random_init_encoder"):
        L.append(f"- random-init：test acc {p['random_init_encoder']['acc_mean']} "
                 f"({p['random_init_encoder']['acc_min']}–{p['random_init_encoder']['acc_max']}，chance 0.5)")
    L.append("")
    L.append("## 5. statistics-only baseline")
    if p.get("statistics_only_S0"):
        L.append(f"- S0（只含理论上被保持的统计：note count / per-lane / chord-size hist / "
                 f"density profile / NPS）：test acc {p['statistics_only_S0']['acc_mean']}")
    if p.get("statistics_plus_aggregates_S1"):
        L.append(f"- S1（S0 + 简单空间聚合：相邻过渡率 / 同 lane 最长 run / chord span）："
                 f"test acc {p['statistics_plus_aggregates_S1']['acc_mean']}"
                 f"（这些聚合量本身属于'排列摘要'，预期 > chance，用于说明 intervention 确实改变了排列）")
    L.append("")
    L.append("## 6. 结果能证明什么")
    L.append("- 若 pretrained >> random-init 且 >> S0：frozen representation 对'统计受控的空间排列变化'敏感，"
             "说明编码器确实利用了点位模式（而非仅统计量）。")
    L.append("")
    L.append("## 7. 结果不能证明什么")
    L.append("- 区分原始/变体 ≠ 学到 skill demand；≠ 原谱比变体更难；≠ Framework 轴正确。")
    L.append("- S1 > chance 也不构成问题：它说明我们确实改变了排列，只是被人工聚合量捕获。")
    L.append("")
    L.append("## 8. 是否值得进入全量实验")
    L.append("（由数字决定：pretrained 明显高于 S0/random 且 audit 干净 → 值得；否则先修 intervention/表示。）")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
