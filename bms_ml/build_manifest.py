"""解析全部谱面 → manifest.jsonl + 数据报告 + 隔离报告。

manifest 每行是一条记录：
  path / rel_path / ext / md5 / sha256 / title / artist / player / rank /
  quarantine[reasons] / issues / flags / meta(ChartMeta) / features(A) /
  labels[{table, level, value}] / has_7k / note_sequence_path(B)

用法: python -m bms_ml.build_manifest --data-dir <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

from .features import build_note_sequence, build_stats_features
from .labels import DifficultyTable, parse_table
from .parser import RawChart, decode_bytes, parse_bms_text
from .split import song_group_key
from .timeline import UnifiedChart, build_timeline


CHART_EXTS = (".bms", ".bme", ".bml", ".bmx")


def parse_one(path: str, rel: str):
    with open(path, "rb") as f:
        raw_bytes = f.read()
    text = decode_bytes(raw_bytes)
    raw: RawChart = parse_bms_text(text, path)
    return raw, build_timeline(raw)


def chart_hashes(path: str):
    with open(path, "rb") as f:
        data = f.read()
    return hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=r"F:\Projects\bms judge\testbms")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output"))
    ap.add_argument("--tables-dir", default=None,
                    help="难度表数据目录（含 <name>_header.json / <name>_data.json）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seq_dir = os.path.join(args.out, "sequences")
    os.makedirs(seq_dir, exist_ok=True)

    tables: list[DifficultyTable] = []
    if args.tables_dir and os.path.isdir(args.tables_dir):
        for hf in sorted(glob.glob(os.path.join(args.tables_dir, "*_header.json"))):
            name = os.path.basename(hf).replace("_header.json", "")
            df = os.path.join(args.tables_dir, name + "_data.json")
            if os.path.exists(df):
                try:
                    tables.append(parse_table(hf, df, source=name))
                    print(f"loaded table: {name}")
                except Exception as e:
                    print(f"skip table {name}: {e}")

    files = sorted({
        f
        for ext in CHART_EXTS
        for f in glob.glob(os.path.join(args.data_dir, "**", "*" + ext), recursive=True)
    })
    print(f"charts: {len(files)}")

    records = []
    quarantine_counter = Counter()
    flag_counter = Counter()
    for path in files:
        rel = os.path.relpath(path, args.data_dir)
        md5, sha256 = chart_hashes(path)
        raw, uc = parse_one(path, rel)

        quarantine = [i.code for i in uc.quarantine]
        flags = []
        meta = uc.meta
        # 7K 判定：出现过 18/19 通道（含全 0 行），或 key6/key7 有 note
        has_7k = bool(uc.used_channels & {"18", "19"})
        if meta.lane_counts[6] + meta.lane_counts[7] > 0:
            has_7k = True
        if meta.total_notes == 0:
            quarantine.append("no_notes")
        elif not has_7k:
            quarantine.append("not_7k")

        if meta.negative_bpm:
            flags.append("negative_bpm")
        if meta.total_notes > 10000:
            flags.append("extreme_notes")
        if meta.duration_sec > 600:
            flags.append("long_chart")
        if meta.max_bpm > 600:
            flags.append("extreme_bpm")
        if meta.stop_count > 100:
            flags.append("extreme_stops")
        for q in quarantine:
            quarantine_counter[q] += 1
        for fl in flags:
            flag_counter[fl] += 1

        features = build_stats_features(meta).tolist()
        seq = build_note_sequence(uc)
        np.save(os.path.join(seq_dir, sha256 + ".npy"), seq)

        labels = []
        for t in tables:
            lv = t.lookup(md5, sha256)
            if lv is not None:
                val = t.level_value(lv)
                labels.append({
                    "table": t.name,
                    "level": lv,
                    "value": val,
                })

        meta = uc.meta
        records.append({
            "path": path,
            "rel_path": rel,
            "ext": os.path.splitext(path)[1].lower(),
            "md5": md5,
            "sha256": sha256,
            "title": uc.title,
            "artist": uc.artist,
            "genre": raw.headers.get("GENRE", ""),
            "player": uc.player,
            "player_mode": {1: "1P", 2: "2P", 3: "DP", 4: "BATTLE"}.get(uc.player, f"P{uc.player}"),
            "rank": uc.rank,
            "group_id": song_group_key({
                "title": uc.title, "artist": uc.artist, "rel_path": rel,
            }),
            "quarantine": sorted(set(quarantine)),
            "flags": flags,
            "issues": [{"code": i.code, "detail": i.detail[:120]} for i in uc.issues[:20]],
            "has_7k": has_7k,
            "features": features,
            "labels": labels,
            "note_sequence_path": os.path.join("sequences", sha256 + ".npy"),
            "meta": {
                "total_notes": meta.total_notes,
                "ln_count": meta.ln_count,
                "ln_ratio": round(meta.ln_ratio, 4),
                "duration_sec": round(meta.duration_sec, 3),
                "initial_bpm": meta.initial_bpm,
                "min_bpm": meta.min_bpm,
                "max_bpm": meta.max_bpm,
                "bpm_change_count": meta.bpm_change_count,
                "stop_count": meta.stop_count,
                "stop_total_sec": round(meta.stop_total_sec, 3),
                "measures": meta.measures,
                "avg_nps": round(meta.avg_nps, 3),
                "peak_nps_1s": meta.peak_nps_1s,
                "chord_count": meta.chord_count,
                "jack_count": meta.jack_count,
                "scratch_count": meta.scratch_count,
                "lane_counts": meta.lane_counts,
            },
        })

    manifest_path = os.path.join(args.out, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_quarantined = sum(1 for r in records if r["quarantine"])
    report = {
        "total_charts": len(records),
        "quarantined": n_quarantined,
        "clean_charts": len(records) - n_quarantined,
        "quarantine_by_reason": dict(quarantine_counter),
        "flag_by_reason": dict(flag_counter),
        "labeled": {
            t.name: sum(1 for r in records if any(l["table"] == t.name for l in r["labels"]))
            for t in tables
        },
        "meta_stats": {
            k: {
                "min": min(r["meta"][k] for r in records),
                "median": sorted(r["meta"][k] for r in records)[len(records) // 2],
                "max": max(r["meta"][k] for r in records),
            }
            for k in ("total_notes", "duration_sec", "avg_nps", "peak_nps_1s",
                      "min_bpm", "max_bpm", "stop_count", "ln_count")
        },
    }
    with open(os.path.join(args.out, "data_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    qrep = [
        {"path": r["rel_path"], "title": r["title"], "quarantine": r["quarantine"],
         "flags": r["flags"], "issues": r["issues"], "meta": r["meta"]}
        for r in records
    ]
    with open(os.path.join(args.out, "quarantine_report.json"), "w", encoding="utf-8") as f:
        json.dump(qrep, f, ensure_ascii=False, indent=2)

    print(f"manifest: {manifest_path}")
    print(f"quarantined: {n_quarantined} / {len(records)}")
    print("quarantine reasons:", dict(quarantine_counter))
    print("flags:", dict(flag_counter))
    print("labeled by table:", report["labeled"])


if __name__ == "__main__":
    main()
