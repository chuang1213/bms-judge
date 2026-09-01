"""Corpus 抽样解析：探测真实谱面中未被测试覆盖的格式与 parser bug。

分层抽样（.pms/.bmx 全量，其余按比例随机），逐文件解析并汇总：
issue/quarantine 代码分布、编码分布、异常字段、解析失败样本。
不写任何训练产物，只输出 probe_report.json。

用法: python -m bms_ml.probe_corpus --charts <charts.txt> --n <抽样数>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

from .parser import decode_bytes, parse_bms_text
from .timeline import build_timeline


def probe_one(path: str) -> dict:
    rec = {"path": path, "ok": True}
    t0 = time.time()
    try:
        with open(path, "rb") as f:
            raw = f.read()
        text = decode_bytes(raw)
        rc = parse_bms_text(text, path)
        uc = build_timeline(rc)
        rec["encoding"] = _detect_encoding(raw)
        rec["bytes"] = len(raw)
        rec["parse_ms"] = round((time.time() - t0) * 1000, 1)
        rec["issues"] = sorted({i.code for i in rc.issues} | {i.code for i in uc.issues})
        rec["quarantine"] = sorted({i.code for i in uc.quarantine})
        rec["player"] = rc.player
        rec["lntype"] = rc.lntype
        rec["control_flow"] = sorted(set(rc.control_flow))
        rec["notes"] = uc.meta.total_notes
        rec["duration"] = round(uc.meta.duration_sec, 2)
        rec["measures"] = uc.meta.measures
        rec["channels"] = sorted(uc.used_channels)[:40]
        rec["title"] = uc.title[:60]
        # 记录出现过的未知头命令（用于发现新格式）
        seen = rc.headers.get("_seen_headers", "")
        rec["unknown_headers"] = sorted(set(x for x in seen.split(",") if x))[:30]
    except Exception as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def _detect_encoding(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp932", "gb18030", "euc-kr"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "?"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "charts.txt"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "corpus"))
    args = ap.parse_args()

    all_paths = [l.strip() for l in open(args.charts, encoding="utf-8") if l.strip()]
    rng = random.Random(args.seed)
    by_ext = defaultdict(list)
    for p in all_paths:
        by_ext[os.path.splitext(p)[1].lower()].append(p)

    # 分层：新扩展全量，其余按比例
    sample = []
    for ext, paths in by_ext.items():
        if ext in (".pms", ".bmx"):
            sample.extend(paths)
        else:
            want = max(1, int(args.n * len(paths) / max(len(all_paths), 1)))
            sample.extend(rng.sample(paths, min(want, len(paths))))
    rng.shuffle(sample)
    sample = sample[: args.n]
    print(f"sampled {len(sample)} charts from {len(all_paths)}")

    results = []
    for i, p in enumerate(sample):
        r = probe_one(p)
        results.append(r)
        if (i + 1) % 100 == 0:
            print(f"  probed {i+1}/{len(sample)}")

    issue_counter = Counter()
    quarantine_counter = Counter()
    enc_counter = Counter()
    ext_counter = Counter()
    fails = []
    for r in results:
        ext_counter[os.path.splitext(r["path"])[1].lower()] += 1
        enc_counter[r.get("encoding", "?")] += 1
        for c in r.get("issues", []):
            issue_counter[c] += 1
        for c in r.get("quarantine", []):
            quarantine_counter[c] += 1
        if not r["ok"]:
            fails.append(r)

    report = {
        "sampled": len(results),
        "by_ext": dict(ext_counter),
        "encoding": dict(enc_counter),
        "issue_codes": dict(issue_counter),
        "quarantine_codes": dict(quarantine_counter),
        "failures": fails[:30],
        "unknown_headers_seen": sorted({
            h for r in results for h in r.get("unknown_headers", [])
        }),
        "control_flow_in_sample": sum(1 for r in results if r.get("control_flow")),
        "lntype_hist": dict(Counter(r.get("lntype") for r in results)),
        "player_hist": dict(Counter(r.get("player") for r in results)),
        "slow_parses": sorted(
            [{"path": r["path"], "ms": r["parse_ms"]} for r in results
             if r.get("parse_ms", 0) > 2000],
            key=lambda x: -x["ms"],
        )[:20],
        "big_files": sorted(
            [{"path": r["path"], "bytes": r["bytes"]} for r in results
             if r.get("bytes", 0) > 2_000_000],
            key=lambda x: -x["bytes"],
        )[:20],
    }
    with open(os.path.join(args.out, "probe_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== probe report ===")
    print("by ext:", report["by_ext"])
    print("encoding:", report["encoding"])
    print("issues:", report["issue_codes"] or "(none)")
    print("quarantine:", report["quarantine_codes"] or "(none)")
    print("failures:", len(fails))
    for f in fails[:5]:
        print("  FAIL", f["path"], f.get("error"))
    print("unknown headers:", report["unknown_headers_seen"][:40])
    print("lntype hist:", report["lntype_hist"], " player hist:", report["player_hist"])
    print("control flow in sample:", report["control_flow_in_sample"])


if __name__ == "__main__":
    main()
