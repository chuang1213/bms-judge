"""数据 sanity check：随机抽取谱面，打印解析统计并输出 piano-roll 图。

用法: python -m bms_ml.sanity_check --manifest <path> --n 5 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
from PIL import Image, ImageDraw

from .features import build_piano_roll
from .parser import parse_bms_text, decode_bytes
from .timeline import build_timeline


LANE_LABELS = ["SC", "1", "2", "3", "4", "5", "6", "7"]


def render_piano_roll(grid: np.ndarray, path: str, cell_px: int = 3,
                      lane_px: int = 16) -> None:
    t, lanes = grid.shape
    w = lane_px * lanes + 24
    h = max(cell_px * t, 20)
    img = Image.new("RGB", (w, h), (20, 20, 28))
    d = ImageDraw.Draw(img)
    for i in range(t):
        for lane in range(lanes):
            v = grid[i, lane]
            if v == 0:
                continue
            color = (80, 200, 255) if v == 1 else (255, 180, 60)
            x0 = 20 + lane * lane_px + 1
            y0 = i * cell_px
            d.rectangle([x0, y0, x0 + lane_px - 3, y0 + max(cell_px - 1, 1)], fill=color)
    for lane, label in enumerate(LANE_LABELS[:lanes]):
        d.text((20 + lane * lane_px, 0), label, fill=(200, 200, 200))
    img.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "output", "manifest.jsonl"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "sanity"))
    args = ap.parse_args()

    records = [json.loads(line) for line in open(args.manifest, encoding="utf-8")]
    clean = [r for r in records if not r["quarantine"]]
    if not clean:
        print("no clean charts; show quarantined instead")
        clean = records
    random.seed(args.seed)
    sample = random.sample(clean, min(args.n, len(clean)))
    os.makedirs(args.out, exist_ok=True)

    for r in sample:
        print("=" * 70)
        print(f"title : {r['title']}")
        print(f"artist: {r['artist']}")
        print(f"path  : {r['rel_path']}")
        print(f"sha256: {r['sha256']}")
        print(f"md5   : {r['md5']}")
        print(f"quarantine: {r['quarantine']}  flags: {r['flags']}")
        m = r["meta"]
        print(f"notes={m['total_notes']} ln={m['ln_count']} duration={m['duration_sec']}s "
              f"measures={m['measures']}")
        print(f"bpm init={m['initial_bpm']} min={m['min_bpm']} max={m['max_bpm']} "
              f"changes={m['bpm_change_count']} stop={m['stop_count']}")
        print(f"avg_nps={m['avg_nps']} peak1s={m['peak_nps_1s']} "
              f"chords={m['chord_count']} jacks={m['jack_count']}")
        if r["labels"]:
            print("labels:", [(l["table"], l["level"]) for l in r["labels"]])
        # 重新解析并渲染
        with open(r["path"], "rb") as f:
            raw = f.read()
        uc = build_timeline(parse_bms_text(decode_bytes(raw), r["path"]))
        grid = build_piano_roll(uc, cell_sec=0.25)
        png = os.path.join(args.out, r["sha256"][:12] + ".png")
        render_piano_roll(grid, png)
        print(f"piano roll: {png} (grid {grid.shape})")
        print("first 5 notes:", [(round(n.time_sec, 2), n.lane, n.note_type) for n in uc.notes[:5]])
    print("=" * 70)


if __name__ == "__main__":
    main()
