"""理解谱面序列：打印并可视化已保存的 note sequence（N x 4）。

不涉及任何神经网络。目的：看懂"机器眼里的 BMS 谱面"。

用法: python -m bms_ml.understand_sequence --manifest output/corpus/manifest.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LANE_LABELS = ["SC", "1", "2", "3", "4", "5", "6", "7"]
LANE_FULL = ["scratch", "key1", "key2", "key3", "key4", "key5", "key6", "key7"]


def fmt_row(row):
    t, lane, typ, dur = row
    return (f"time={t:8.3f}, lane={int(lane)} ({LANE_LABELS[int(lane)]:>2s}), "
            f"type={'normal' if typ == 0 else 'LN'}"
            + (f", duration={dur:.3f}s" if typ == 1 else ""))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "manifest.jsonl"))
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "analysis", "sequence_learn"))
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))  # bms_ml/
    seq_root = os.path.join(root, "output", "corpus", "sequences")
    records = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    sat = [
        r for r in records
        if not r["quarantine"] and r["has_7k"] and not r["flags"]
        and any(l["table"] == "Satellite" and l["value"] is not None for l in r["labels"])
    ]
    random.Random(args.seed).shuffle(sat)
    chosen = sat[: args.n]
    os.makedirs(args.out, exist_ok=True)

    for idx, r in enumerate(chosen):
        sl = next(l["level"] for l in r["labels"] if l["table"] == "Satellite")
        seq_path = os.path.join(seq_root, r["sha256"] + ".npy")
        seq = np.load(seq_path)  # N x 4
        meta = r["meta"]
        print("=" * 76)
        print(f"chart {idx+1}: {r['title']}  (Satellite sl={sl})")
        print(f"  file: {r['rel_path']}")
        print(f"  总 note 数: {meta['total_notes']}  (序列行数 N = {seq.shape[0]})")
        print(f"  谱面时长: {meta['duration_sec']:.2f} s")

        # 前 30 个 event
        print(f"\n  前 {min(30, seq.shape[0])} 个 event：")
        for row in seq[:30]:
            print("   ", fmt_row(row))

        # 最长连续 event 数：相邻 event 间隔 <= 0.25s 视为连续段
        ts = seq[:, 0]
        gaps = np.diff(ts)
        best, cur = 1, 1
        for g in gaps:
            if g <= 0.25:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        # 同一时间点最多几个 note（和弦）
        _, counts = np.unique(np.round(ts, 3), return_counts=True)
        print(f"  最长连续 event 数（间隔<=0.25s 的连续段）: {best}")
        print(f"  同一时间点最多同时几个 note: {counts.max()}")
        print(f"  其中 LN 行数: {int((seq[:, 2] == 1).sum())}  "
              f"平均 NPS: {meta['avg_nps']:.2f}")

        # 简单可视化：横轴时间，纵轴 lane
        if idx < 2:
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.scatter(seq[:, 0], seq[:, 1], s=9, c="steelblue", alpha=0.7)
            ax.set_yticks(range(8))
            ax.set_yticklabels(LANE_LABELS)
            ax.set_xlabel("time (s)")
            ax.set_ylabel("lane")
            ax.set_title(f"{r['title']}  (sl={sl}, {seq.shape[0]} notes)")
            fig.tight_layout()
            png = os.path.join(args.out, f"chart{idx+1}_{r['sha256'][:8]}.png")
            fig.savefig(png, dpi=110)
            plt.close(fig)
            print(f"  scatter 图已保存: {png}")
    print("=" * 76)


if __name__ == "__main__":
    main()
