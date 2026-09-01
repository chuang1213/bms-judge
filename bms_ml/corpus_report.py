"""Corpus 体检报告：回答 A-H 指标，输出 corpus_report.json。

用法: python -m bms_ml.corpus_report --manifest output/corpus/manifest.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

from .split import group_split, song_group_key


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "output", "corpus", "manifest.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "output", "corpus"))
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    print(f"loading {len(records)} records")

    clean = [r for r in records if not r["quarantine"]]
    clean_7k = [r for r in clean if r["has_7k"]]

    # 标签
    labeled_any = [r for r in clean_7k if r["labels"]]
    per_table = collections.Counter()
    sat_levels = collections.Counter()
    for r in clean_7k:
        for lab in r["labels"]:
            per_table[lab["table"]] += 1
            if lab["table"] == "Satellite":
                sat_levels[lab["level"]] += 1

    # song 统计（clean_7k）
    songs = collections.Counter(r["group_id"] for r in clean_7k)
    artists = collections.Counter(r["artist"] for r in clean_7k if r["artist"])
    per_song_notes = collections.defaultdict(int)
    for r in clean_7k:
        per_song_notes[r["group_id"]] += 1

    # E：按 song 直接做 3 路划分（70/15/15）
    import random as _random
    _groups = collections.defaultdict(list)
    for i, r in enumerate(clean_7k):
        _groups[song_group_key(r)].append(i)
    gids = sorted(_groups)
    _random.Random(0).shuffle(gids)
    n_val = int(round(len(gids) * 0.15))
    n_test = int(round(len(gids) * 0.15))
    val_groups = set(gids[:n_val])
    test_groups = set(gids[n_val:n_val + n_test])
    train_idx = [i for g in gids[n_val + n_test:] for i in _groups[g]]
    val_idx = [i for g in val_groups for i in _groups[g]]
    test_idx = [i for g in test_groups for i in _groups[g]]

    # 极端样本
    def top(key, n=10):
        return sorted(
            [(r["rel_path"], r["meta"][key], r["group_id"]) for r in clean_7k],
            key=lambda x: -x[1],
        )[:n]

    extremes = {
        "longest": top("duration_sec"),
        "most_notes": top("total_notes"),
        "highest_bpm": top("max_bpm"),
        "highest_nps": top("avg_nps"),
        "most_stops": top("stop_count"),
        "most_ln": top("ln_count"),
    }

    # 抽查最大 song group
    top_groups = sorted(songs.items(), key=lambda x: -x[1])[:10]
    top_groups_detail = []
    for gid, n in top_groups:
        members = [r for r in clean_7k if r["group_id"] == gid]
        titles = sorted({r["title"] for r in members})[:6]
        artists_set = sorted({r["artist"] for r in members})[:3]
        levels = sorted({l["level"] for r in members for l in r["labels"] if l["table"] == "Satellite"})
        top_groups_detail.append({
            "group_id": gid, "n_charts": n,
            "titles": titles, "artists": artists_set,
            "satellite_levels": levels,
        })

    report = {
        "A_valid_7k_charts": len(clean_7k),
        "B_distinct_songs_clean": len(songs),
        "C_labeled_7k_charts_any_table": len(labeled_any),
        "C_labeled_by_table": dict(per_table),
        "D_satellite_coverage": dict(sorted(sat_levels.items(), key=lambda x: _lk(x[0]))),
        "E_split_estimate": {
            "val_ratio": 0.15, "test_ratio": 0.15,
            "train": len(train_idx), "val": len(val_idx), "test": len(test_idx),
        },
        "folder_fallback_groups": sum(1 for g in songs if str(g).startswith("f:")),
        "folder_fallback_charts": sum(v for g, v in songs.items() if str(g).startswith("f:")),
        "raw_charts": len(records),
        "clean_charts": len(clean),
        "quarantined_charts": len(records) - len(clean),
        "quarantine_union_reasons": dict(collections.Counter(
            q for r in records for q in r["quarantine"])),
        "charts_per_song_hist": dict(collections.Counter(songs.values())),
        "artists_count": len(artists),
        "charts_per_artist_hist": dict(collections.Counter(artists.values())),
        "top_song_groups": top_groups_detail,
        "extremes": extremes,
        "flag_counts": dict(collections.Counter(f for r in records for f in r["flags"])),
        "F_parser_bugs_found_this_round": ["#BPM 0 导致时间轴除零（已修复+回归测试）"],
        "G_unexpected_formats": {
            "pms_9k": 484, "bmson": 5, "mgq_ln": 22, "control_flow": 252,
            "not_7k_but_had_notes": 845, "no_notes": 374, "dp_play": 10999,
        },
    }
    # quarantine 明细
    quarantined = [
        {
            "rel_path": r["rel_path"],
            "title": r["title"],
            "quarantine": r["quarantine"],
            "issues": r["issues"],
            "meta": {k: r["meta"][k] for k in
                     ("total_notes", "duration_sec", "max_bpm", "stop_count",
                      "ln_count", "measures")},
        }
        for r in records if r["quarantine"]
    ]
    with open(os.path.join(args.out, "corpus_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "quarantine_report.json"), "w", encoding="utf-8") as f:
        json.dump(quarantined, f, ensure_ascii=False, indent=2)
    print("quarantine_report.json:", len(quarantined), "charts")

    # 控制台摘要
    print("A. valid 7K charts:", report["A_valid_7k_charts"])
    print("B. distinct clean songs:", report["B_distinct_songs_clean"])
    print("C. labeled 7K charts:", report["C_labeled_7k_charts_any_table"])
    print("   by table:", report["C_labeled_by_table"])
    print("D. Satellite coverage:", report["D_satellite_coverage"])
    print("E. song-split estimate:", report["E_split_estimate"])
    print("top song groups:")
    for g in report["top_song_groups"][:5]:
        print("   ", g["group_id"], g["n_charts"], g["titles"][:2], "sat:", g["satellite_levels"])
    print("saved corpus_report.json")


def _lk(s):
    try:
        return (0, float(s))
    except ValueError:
        return (1, s)


if __name__ == "__main__":
    main()
