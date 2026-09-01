"""Corpus 侦察：流式扫描目录，统计规模并落盘谱面路径清单。

只遍历文件系统元数据（不读文件内容），内存占用恒定。
输出到项目 output/corpus/ 下，不触碰原始数据。

用法: python -m bms_ml.recon_corpus --root <BMS目录> --out <项目输出目录>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter


CHART_EXTS = {".bms", ".bme", ".bml", ".bmx", ".bmson", ".pms", ".pmsone"}
ARCHIVE_EXTS = {".zip", ".7z", ".rar", ".lzh", ".tar", ".gz"}


def scan(root: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    ext_count = Counter()
    n_files = 0
    n_dirs = 0
    total_bytes = 0
    chart_paths = []
    archive_count = Counter()
    chart_bytes = 0
    t0 = time.time()

    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                entries = list(it)
        except OSError as e:
            print("  [scan warn]", d, e)
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    n_dirs += 1
                    stack.append(e.path)
                elif e.is_file(follow_symlinks=False):
                    n_files += 1
                    sz = e.stat().st_size
                    total_bytes += sz
                    ext = os.path.splitext(e.name)[1].lower()
                    ext_count[ext] += 1
                    if ext in CHART_EXTS:
                        chart_paths.append(e.path)
                        chart_bytes += sz
                    elif ext in ARCHIVE_EXTS:
                        archive_count[ext] += 1
            except OSError:
                continue
        if n_files % 20000 == 0 and n_files:
            print(f"  scanned {n_files} files, {n_dirs} dirs, "
                  f"charts so far {len(chart_paths)}  ({time.time()-t0:.0f}s)")

    # 顶层目录计数（供 package 概念参考）
    top_dirs = 0
    try:
        with os.scandir(root) as it:
            top_dirs = sum(1 for e in it if e.is_dir(follow_symlinks=False))
    except OSError:
        pass

    # 落盘
    with open(os.path.join(out_dir, "charts.txt"), "w", encoding="utf-8") as f:
        for p in chart_paths:
            f.write(p + "\n")

    summary = {
        "root": root,
        "files": n_files,
        "dirs": n_dirs,
        "top_level_dirs": top_dirs,
        "total_bytes": total_bytes,
        "chart_files": len(chart_paths),
        "chart_bytes": chart_bytes,
        "ext_counts": dict(sorted(ext_count.items(), key=lambda x: -x[1])[:60]),
        "archive_counts": dict(archive_count),
        "scan_seconds": round(time.time() - t0, 1),
        "charts_path_file": os.path.join(out_dir, "charts.txt"),
    }
    with open(os.path.join(out_dir, "scan_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\games\BMS")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "corpus"))
    args = ap.parse_args()

    print(f"scanning {args.root} ...")
    s = scan(args.root, args.out)
    print("=== scan summary ===")
    for k in ("files", "dirs", "top_level_dirs", "total_bytes",
              "chart_files", "chart_bytes", "scan_seconds"):
        v = s[k]
        if isinstance(v, int) and k.endswith("bytes"):
            v = f"{v/1e9:.2f} GB"
        print(f"  {k}: {v}")
    print("  chart exts:", {k: v for k, v in s["ext_counts"].items() if k in CHART_EXTS})
    print("  archives:", s["archive_counts"])


if __name__ == "__main__":
    main()
