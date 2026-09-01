"""Dataset audit：对 manifest 输出数据质量统计报告。

用法: python -m bms_ml.dataset_audit --manifest <path> [--level-table <表名>]

输出：JSON 报告 + 控制台摘要 + 难度等级直方图（ASCII）。
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

from .split import group_summary


def hist(values, bins, labels):
    counts = [0] * (len(bins) - 1)
    for v in values:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                counts[i] += 1
                break
    rows = []
    for i, c in enumerate(counts):
        bar = "#" * min(c, 60)
        rows.append(f"{labels[i]:>16s} {c:6d} {bar}")
    return "\n".join(rows)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(__file__), "output", "manifest.jsonl"))
    ap.add_argument("--level-table", default="",
                    help="难度直方图使用的表名；缺省自动选标签最多的表")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    clean = [r for r in records if not r["quarantine"]]

    # 基础计数
    report = {
        "total_charts": len(records),
        "clean_charts": len(clean),
        "quarantined_charts": len(records) - len(clean),
        "songs": len({r["group_id"] for r in records}),
        "clean_songs": len({r["group_id"] for r in clean}),
        "artists": len({r["artist"] for r in records if r["artist"]}),
        "quarantine_by_reason": dict(collections.Counter(
            q for r in records for q in r["quarantine"])),
        "warning_by_code": dict(collections.Counter(
            i["code"] for r in records for i in r["issues"])),
        "player_modes": dict(collections.Counter(r["player_mode"] for r in records)),
    }

    # 表统计
    table_levels = collections.defaultdict(collections.Counter)
    for r in records:
        for lab in r["labels"]:
            table_levels[lab["table"]][lab["level"]] += 1
    report["tables"] = {
        t: {"n_charts": sum(c.values()), "n_levels": len(c),
            "levels": dict(sorted(c.items(), key=lambda x: _level_key(x[0])))}
        for t, c in table_levels.items()
    }

    # 数值分布
    m = [r["meta"] for r in clean]
    report["distributions"] = {
        "bpm_init": {
            "min": min(x["initial_bpm"] for x in m),
            "median": sorted(x["initial_bpm"] for x in m)[len(m) // 2],
            "max": max(x["initial_bpm"] for x in m),
        },
        "bpm_max": {
            "min": min(x["max_bpm"] for x in m),
            "median": sorted(x["max_bpm"] for x in m)[len(m) // 2],
            "max": max(x["max_bpm"] for x in m),
        },
        "total_notes": {
            "min": min(x["total_notes"] for x in m),
            "median": sorted(x["total_notes"] for x in m)[len(m) // 2],
            "max": max(x["total_notes"] for x in m),
        },
        "duration_sec": {
            "min": min(x["duration_sec"] for x in m),
            "median": sorted(x["duration_sec"] for x in m)[len(m) // 2],
            "max": max(x["duration_sec"] for x in m),
        },
        "ln_ratio": {
            "min": min(x["ln_ratio"] for x in m),
            "median": sorted(x["ln_ratio"] for x in m)[len(m) // 2],
            "max": max(x["ln_ratio"] for x in m),
        },
        "scratch_ratio": {
            "min": min(x["scratch_count"] / max(x["total_notes"], 1) for x in m),
            "median": sorted(x["scratch_count"] / max(x["total_notes"], 1) for x in m)[len(m) // 2],
            "max": max(x["scratch_count"] / max(x["total_notes"], 1) for x in m),
        },
    }

    # 组（song）统计 + 每 song 难度范围
    groups = group_summary(records)
    report["group_stats"] = {
        "flagged_merges": groups["flagged_merges"],
        "charts_per_song": dict(collections.Counter(
            g["n_charts"] for g in groups["groups"].values())),
    }
    song_levels = collections.defaultdict(list)
    for r in clean:
        for lab in r["labels"]:
            if lab["value"] is not None:
                song_levels[(r["group_id"], lab["table"])].append(lab["value"])
    report["song_difficulty_range"] = {
        f"{g} | {t}": {"min": min(v), "max": max(v), "n": len(v)}
        for (g, t), v in sorted(song_levels.items())
    }

    with open(os.path.join(os.path.dirname(args.manifest), "audit_report.json"),
              "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ---- 控制台摘要 ----
    print(f"charts: {report['total_charts']}  clean: {report['clean_charts']}  "
          f"quarantined: {report['quarantined_charts']}")
    print(f"songs: {report['songs']} (clean: {report['clean_songs']})  artists: {report['artists']}")
    print("player modes:", report["player_modes"])
    print("quarantine:", report["quarantine_by_reason"])
    print("warnings:", report["warning_by_code"] or "(none)")
    for t, info in report["tables"].items():
        print(f"table {t}: {info['n_charts']} charts, {info['n_levels']} distinct levels")

    # 难度直方图
    table = args.level_table or max(report["tables"], key=lambda t: report["tables"][t]["n_charts"])
    if table in report["tables"]:
        print(f"\n== difficulty level histogram: {table} ==")
        for level, c in report["tables"][table]["levels"].items():
            print(f"  {level:>5s} {c:4d} {'#' * min(c, 60)}")

    print("\nflagged potential mis-merges (duration span>50% or notes>5x):",
          len(report["group_stats"]["flagged_merges"]))
    for fg in report["group_stats"]["flagged_merges"][:10]:
        print("  ", fg["group_id"], fg["n_charts"], fg["duration_min"], "-", fg["duration_max"])
    print("\naudit_report.json saved")


def _level_key(s: str):
    try:
        return (0, float(s))
    except ValueError:
        return (1, s)


if __name__ == "__main__":
    main()
