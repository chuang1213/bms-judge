"""全量 corpus 解析 → manifest（并行、流式、不污染原始数据）。

用法: python -m bms_ml.build_corpus_manifest
      --charts output/corpus/charts.txt --out output/corpus --tables-dir output/tables

产物（都在 --out 下）：
  manifest.jsonl          每条谱面记录（字段同 build_manifest）
  quarantine_report.json  隔离明细
  data_report.json        计数摘要
  dedupe_report.json      重复 md5 统计
  sequences/<sha>.npy     干净 7K 谱面的 note sequence（表示 B）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .features import build_note_sequence, build_stats_features
from .labels import DifficultyTable, parse_table
from .parser import decode_bytes, parse_bms_text
from .split import song_group_key
from .timeline import build_timeline


TABLES: list[DifficultyTable] = []
SEQ_DIR = ""


def _init_worker(tables_payload, seq_dir):
    global TABLES, SEQ_DIR
    TABLES = tables_payload
    SEQ_DIR = seq_dir


def _hash_stream(path):
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def _parse_one(path, md5, sha256):
    with open(path, "rb") as f:
        raw = f.read()
    text = decode_bytes(raw)
    rc = parse_bms_text(text, path)
    uc = build_timeline(rc)
    return rc, uc


def process_chunk(chunk: list):
    """处理一批路径，返回记录列表（worker 内执行）。"""
    out = []
    for path in chunk:
        t0 = time.time()
        rel = path
        ext = os.path.splitext(path)[1].lower()
        md5, sha256 = _hash_stream(path)
        rc, uc = _parse_one(path, md5, sha256)
        meta = uc.meta

        quarantine = [i.code for i in uc.quarantine]
        if ext == ".pms":
            quarantine.append("pms_9k_not_7k")
        elif ext == ".bmson":
            quarantine.append("bmson_unsupported")
        has_7k = bool(uc.used_channels & {"18", "19"})
        if meta.lane_counts[6] + meta.lane_counts[7] > 0:
            has_7k = True
        if meta.total_notes == 0:
            quarantine.append("no_notes")
        elif not has_7k:
            quarantine.append("not_7k")

        flags = []
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
        if meta.ln_ratio > 0.8:
            flags.append("extreme_ln")
        parse_ms = round((time.time() - t0) * 1000, 1)
        if parse_ms > 2000:
            flags.append("slow_parse")

        features = build_stats_features(meta).tolist()
        clean = not quarantine
        seq_path = None
        if clean:
            seq = build_note_sequence(uc)
            fn = os.path.join(SEQ_DIR, sha256 + ".npy")
            np.save(fn, seq)
            seq_path = os.path.join("sequences", sha256 + ".npy")

        labels = []
        for t in TABLES:
            lv = t.lookup(md5, sha256)
            if lv is not None:
                labels.append({"table": t.name, "level": lv,
                               "value": t.level_value(lv)})

        rec = {
            "path": path,
            "rel_path": rel,
            "ext": ext,
            "md5": md5,
            "sha256": sha256,
            "title": uc.title,
            "artist": uc.artist,
            "genre": rc.headers.get("GENRE", ""),
            "player": uc.player,
            "player_mode": {1: "1P", 2: "2P", 3: "DP", 4: "BATTLE"}.get(uc.player, f"P{uc.player}"),
            "rank": uc.rank,
            "group_id": song_group_key({"title": uc.title, "artist": uc.artist, "rel_path": rel}),
            "quarantine": sorted(set(quarantine)),
            "flags": flags,
            "issues": [{"code": i.code, "detail": i.detail[:120]} for i in uc.issues[:20]],
            "has_7k": has_7k,
            "parse_ms": parse_ms,
            "features": features,
            "labels": labels,
            "note_sequence_path": seq_path,
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
        }
        out.append(rec)
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "charts.txt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "corpus"))
    ap.add_argument("--tables-dir", default=os.path.join(os.path.dirname(__file__), "output", "tables"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="调试用：只处理前 N 个")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    global SEQ_DIR
    SEQ_DIR = os.path.join(args.out, "sequences")
    os.makedirs(SEQ_DIR, exist_ok=True)

    tables = []
    if args.tables_dir and os.path.isdir(args.tables_dir):
        import glob
        for hf in sorted(glob.glob(os.path.join(args.tables_dir, "*_header.json"))):
            name = os.path.basename(hf).replace("_header.json", "")
            df = os.path.join(args.tables_dir, name + "_data.json")
            if os.path.exists(df):
                try:
                    tables.append(parse_table(hf, df, source=name))
                except Exception as e:
                    print(f"skip table {name}: {e}")
    print(f"tables: {[t.name for t in tables]}")

    paths = [l.strip() for l in open(args.charts, encoding="utf-8") if l.strip()]
    if args.limit:
        paths = paths[: args.limit]
    print(f"charts to process: {len(paths)}")

    CHUNK = 400
    chunks = [paths[i:i + CHUNK] for i in range(0, len(paths), CHUNK)]
    t0 = time.time()
    n_done = 0
    quarantine_counter = Counter()
    flag_counter = Counter()
    md5_groups: dict = {}
    manifest_path = os.path.join(args.out, "manifest.jsonl")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 initializer=_init_worker,
                                 initargs=(tables, SEQ_DIR)) as ex:
            for chunk_out in ex.map(process_chunk, chunks, chunksize=1):
                for rec in chunk_out:
                    mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    for q in rec["quarantine"]:
                        quarantine_counter[q] += 1
                    for fl in rec["flags"]:
                        flag_counter[fl] += 1
                    md5_groups.setdefault(rec["md5"], []).append(rec["rel_path"])
                n_done += len(chunk_out)
                if n_done % 4000 < CHUNK and n_done:
                    print(f"  {n_done}/{len(paths)}  ({time.time()-t0:.0f}s)")

    dup = {k: v for k, v in md5_groups.items() if len(v) > 1}
    with open(os.path.join(args.out, "dedupe_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "total_charts": len(paths),
            "unique_md5": len(md5_groups),
            "duplicate_groups": len(dup),
            "duplicate_chart_files": sum(len(v) - 1 for v in dup.values()),
            "examples": [{"md5": k, "paths": v[:4]} for k, v in sorted(dup.items(), key=lambda x: -len(x[1]))[:20]],
        }, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.out, "data_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "total_charts": len(paths),
            "quarantine_by_reason": dict(quarantine_counter),
            "flag_by_reason": dict(flag_counter),
        }, f, ensure_ascii=False, indent=2)

    print(f"done in {time.time()-t0:.0f}s")
    print("quarantine:", dict(quarantine_counter))
    print("flags:", dict(flag_counter))
    print("unique md5:", len(md5_groups), " duplicate groups:", len(dup))
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
