"""Phase 3 audit: inspect raw beatoraja save files (score.db / scorelog.db / scoredatalog.db).

Schema reference: lampghost/internal/entity/beatoraja.go (read-only reference project).
Outputs:
  - bms_ml/output/phase3/audit/scorelog_audit.json
  - bms_ml/output/phase3/audit/plots/*.png
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "PlayerData" / "beatoraja"   # archive tree renamed 2026-09-11
OUT = ROOT / "bms_ml" / "output" / "phase3" / "audit"
PLOTS = OUT / "plots"

# ascii labels for plots (avoid CJK font issues)
PLAYERS = {
    "chuang": "chuang/player1",
    "muiclac": "muiclac/player1",
    "buzhang": "局长/player2",
}

CLEAR_NAMES = {
    0: "NO_PLAY", 1: "FAILED", 2: "ASSIST_EASY", 3: "LIGHT_ASSIST_EASY",
    4: "EASY", 5: "NORMAL", 6: "HARD", 7: "EX_HARD", 8: "FULL_COMBO",
    9: "PERFECT", 10: "MAX",
}


def load_table(player_dir: Path, db: str, table: str) -> pd.DataFrame:
    con = sqlite3.connect(player_dir / db)
    try:
        return pd.read_sql_query(f"SELECT * FROM [{table}]", con)
    finally:
        con.close()


def ex_score(epg, lpg, egr, lgr) -> np.ndarray:
    return (lpg + epg) * 2 + egr + lgr


def audit_player(name: str, rel: str, report: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pdir = DATA / rel
    score = load_table(pdir, "score.db", "score")
    log = load_table(pdir, "scorelog.db", "scorelog")
    datalog = load_table(pdir, "scoredatalog.db", "scoredatalog")
    player_daily = load_table(pdir, "score.db", "player")
    r: dict = {}

    # ---------- score table (best records) ----------
    r["score_rows"] = int(len(score))
    r["score_unique_sha256"] = int(score["sha256"].nunique())
    r["score_modes"] = score["mode"].value_counts().to_dict()
    r["score_state_dist"] = score["state"].value_counts().to_dict()
    r["score_notes_zero"] = int((score["notes"] == 0).sum())
    r["score_playcount_sum"] = int(score["playcount"].sum())
    # score rows keyed by (sha256, mode)?
    r["score_dup_keys"] = int(score.duplicated(subset=["sha256", "mode"]).sum())
    dates = pd.to_datetime(score["date"], unit="s", errors="coerce")
    r["score_date_min"] = str(dates.min())
    r["score_date_max"] = str(dates.max())
    r["score_date_null"] = int(dates.isna().sum())
    r["score_clear_dist"] = {CLEAR_NAMES.get(k, str(k)): int(v)
                             for k, v in score["clear"].value_counts().items()}

    # BP consistency: minbp vs bad+poor (+ empty miss) on best records
    bp_simple = score["ebd"] + score["lbd"] + score["epr"] + score["lpr"]
    bp_with_empty = bp_simple + score["ems"] + score["lms"]
    r["bp_check_minbp_eq_badpoor"] = int((score["minbp"] == bp_simple).sum())
    r["bp_check_minbp_eq_badpoor_empty"] = int((score["minbp"] == bp_with_empty).sum())
    r["score_rows"] = int(len(score))

    # accuracy on best records
    ex = ex_score(score["epg"], score["lpg"], score["egr"], score["lgr"])
    acc = np.where(score["notes"] > 0, ex * 50.0 / score["notes"].replace(0, np.nan), np.nan)
    r["score_acc_describe"] = pd.Series(acc).describe().round(4).to_dict()
    r["score_ex_eq_check_vs_judge"] = None  # score table has no per-play score col

    # ---------- scorelog (per-play log) ----------
    r["log_rows"] = int(len(log))
    r["log_unique_sha256"] = int(log["sha256"].nunique())
    r["log_modes"] = log["mode"].value_counts().to_dict()
    log_dates = pd.to_datetime(log["date"], unit="s", errors="coerce")
    r["log_date_min"] = str(log_dates.min())
    r["log_date_max"] = str(log_dates.max())
    r["log_date_null"] = int(log_dates.isna().sum())
    dup_keys = log.duplicated(subset=["sha256", "mode", "date"]).sum()
    r["log_dup_sha256_mode_date"] = int(dup_keys)
    r["log_clear_dist"] = {CLEAR_NAMES.get(k, str(k)): int(v)
                           for k, v in log["clear"].value_counts().items()}
    plays_per_chart = log.groupby(["sha256", "mode"]).size()
    r["log_plays_per_chart_describe"] = plays_per_chart.describe().round(3).to_dict()
    r["log_single_play_charts"] = int((plays_per_chart == 1).sum())

    # granularity test: does scorelog contain EVERY play, or only record updates?
    # compare per-chart playlog rows vs score.playcount (the authoritative counter)
    pc = score.set_index(["sha256", "mode"])["playcount"]
    log_counts = plays_per_chart
    joined = pd.concat([pc.rename("playcount"), log_counts.rename("log_rows")],
                       axis=1, join="inner")
    r["granularity"] = {
        "charts_in_both": int(len(joined)),
        "log_rows_eq_playcount": int((joined["log_rows"] == joined["playcount"]).sum()),
        "log_rows_lt_playcount": int((joined["log_rows"] < joined["playcount"]).sum()),
        "log_rows_gt_playcount": int((joined["log_rows"] > joined["playcount"]).sum()),
        "sum_playcount": int(joined["playcount"].sum()),
        "sum_log_rows_same_charts": int(joined["log_rows"].sum()),
    }
    r["log_charts_not_in_score"] = int(len(set(map(tuple, log[["sha256", "mode"]].drop_duplicates().values))
                                           - set(map(tuple, score[["sha256", "mode"]].values))))

    # oldscore semantics: first play should have oldscore==0
    first_by_chart = log.sort_values("date").groupby(["sha256", "mode"]).first()
    r["log_first_play_oldscore_zero"] = int((first_by_chart["oldscore"] == 0).sum())
    r["log_first_play_count"] = int(len(first_by_chart))

    # ---------- scoredatalog (per-play with judgements) ----------
    r["datalog_rows"] = int(len(datalog))
    r["datalog_unique_sha256"] = int(datalog["sha256"].nunique())
    r["datalog_modes"] = datalog["mode"].value_counts().to_dict()
    dl_dates = pd.to_datetime(datalog["date"], unit="s", errors="coerce")
    r["datalog_date_min"] = str(dl_dates.min())
    r["datalog_date_max"] = str(dl_dates.max())
    r["datalog_dup_sha256_mode_date"] = int(datalog.duplicated(subset=["sha256", "mode", "date"]).sum())

    # hypothesis: scoredatalog rows exist only when the best record was updated
    dl_counts = datalog.groupby(["sha256", "mode"]).size()
    j2 = pd.concat([pc.rename("playcount"), dl_counts.rename("dl_rows")], axis=1, join="inner")
    r["datalog_granularity"] = {
        "charts_in_both": int(len(j2)),
        "dl_rows_eq_playcount": int((j2["dl_rows"] == j2["playcount"]).sum()),
        "dl_rows_lt_playcount": int((j2["dl_rows"] < j2["playcount"]).sum()),
        "dl_rows_gt_playcount": int((j2["dl_rows"] > j2["playcount"]).sum()),
    }
    # does the latest scoredatalog row per chart match score row (best)?
    dl_last = datalog.sort_values("date").groupby(["sha256", "mode"]).last()
    cmp_cols = ["clear", "epg", "lpg", "egr", "lgr", "egd", "lgd", "ebd", "lbd", "epr", "lpr", "ems", "lms", "notes"]
    m = score.set_index(["sha256", "mode"])[cmp_cols].join(dl_last[cmp_cols], rsuffix="_dl", how="inner")
    eq = (m[cmp_cols].values == m[[c + "_dl" for c in cmp_cols]].values).all(axis=1)
    r["datalog_last_matches_score_best"] = {"n": int(len(m)), "match": int(eq.sum())}

    # EX-score consistency between scorelog and scoredatalog on matched (sha256, mode, date)
    dl_key = datalog.set_index(["sha256", "mode", "date"])
    lg = log.set_index(["sha256", "mode", "date"])
    common = lg.index.intersection(dl_key.index)
    if len(common):
        dl_ex = ex_score(dl_key.loc[common, "epg"], dl_key.loc[common, "lpg"],
                         dl_key.loc[common, "egr"], dl_key.loc[common, "lgr"])
        r["log_vs_datalog_score_match"] = {
            "n": int(len(common)),
            "score_eq_ex": int((lg.loc[common, "score"].values == dl_ex).sum()),
        }
        # lamp consistency on matched rows
        r["log_vs_datalog_clear_match"] = int((lg.loc[common, "clear"].values == dl_key.loc[common, "clear"].values).sum())
        # BP consistency on matched rows: minbp vs bad+poor(+empty)
        bp2 = dl_key.loc[common, "ebd"] + dl_key.loc[common, "lbd"] + dl_key.loc[common, "epr"] + dl_key.loc[common, "lpr"]
        bpe = bp2 + dl_key.loc[common, "ems"] + dl_key.loc[common, "lms"]
        r["log_vs_datalog_minbp_eq_badpoor"] = int((lg.loc[common, "minbp"].values == bp2.values).sum())
        r["log_vs_datalog_minbp_eq_badpoor_empty"] = int((lg.loc[common, "minbp"].values == bpe.values).sum())

    # option / random distribution (selection-bias + effective-chart relevance)
    r["log_option_dist"] = log["option"].value_counts().head(8).to_dict() if "option" in log else None
    r["datalog_option_dist"] = datalog["option"].value_counts().head(8).to_dict()
    r["datalog_random_dist"] = datalog["random"].value_counts().head(8).to_dict()

    # player daily activity table
    pd_dates = pd.to_datetime(player_daily["date"], unit="s", errors="coerce")
    r["player_daily_days"] = int(len(player_daily))
    r["player_daily_date_min"] = str(pd_dates.min())
    r["player_daily_date_max"] = str(pd_dates.max())
    r["player_daily_playtime_hours"] = round(float(player_daily["playtime"].sum()) / 3600.0, 1)
    r["player_daily_playcount_sum"] = int(player_daily["playcount"].sum())

    report[name] = r
    return score, log, datalog


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    report: dict = {"generated": str(pd.Timestamp.now()), "players": {}}
    frames = {}
    for name, rel in PLAYERS.items():
        score, log, datalog = audit_player(name, rel, report["players"])
        frames[name] = (score, log, datalog)

    # ---------- cross-player overlap (7K-ish modes only for overlap: use all then mode-aware) ----------
    ov: dict = {}
    mode_sets = {}
    for name, (score, log, dl) in frames.items():
        mode_sets[name] = set(zip(log["sha256"], log["mode"]))
    ov["log_key_sets"] = {k: len(v) for k, v in mode_sets.items()}
    c, m, b = (mode_sets[k] for k in ["chuang", "muiclac", "buzhang"])
    ov["pairs"] = {
        "chuang&muiclac": len(c & m), "chuang&buzhang": len(c & b), "muiclac&buzhang": len(m & b),
    }
    ov["all_three"] = len(c & m & b)
    ov["union"] = len(c | m | b)
    # first-play only views (each chart counted once per player regardless of mode)
    chart_sets = {k: set(frames[k][1]["sha256"]) for k in frames}
    cc, mm, bb = (chart_sets[k] for k in ["chuang", "muiclac", "buzhang"])
    ov["chart_level"] = {
        "chuang": len(cc), "muiclac": len(mm), "buzhang": len(bb),
        "chuang&muiclac": len(cc & mm), "chuang&buzhang": len(cc & bb), "muiclac&buzhang": len(mm & bb),
        "all_three": len(cc & mm & bb), "union": len(cc | mm | bb),
    }
    report["overlap"] = ov

    # ---------- ID alignment with our corpus ----------
    manifest_sha = set()
    manifest_meta = {}
    with open(ROOT / "bms_ml" / "output" / "corpus" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            manifest_sha.add(row["sha256"])
            manifest_meta[row["sha256"]] = row
    union_charts = set().union(*chart_sets.values())
    covered = union_charts & manifest_sha
    report["id_alignment"] = {
        "manifest_unique_sha256": len(manifest_sha),
        "player_union_charts": len(union_charts),
        "player_charts_in_manifest": len(covered),
        "player_charts_missing_from_manifest": len(union_charts - manifest_sha),
        "per_player_in_manifest": {
            k: len(v & manifest_sha) for k, v in chart_sets.items()
        },
    }

    # difficulty tables coverage
    tables = {}
    for tf in (ROOT / "bms_ml" / "output" / "tables").glob("*.json"):
        d = json.load(open(tf, encoding="utf-8"))
        body = d if isinstance(d, list) else d.get("body")
        if body is None:
            continue
        entries = {}
        for e in body:
            sha = e.get("sha256")
            if sha:
                entries[sha] = e.get("level")
        tables[tf.stem] = entries
    report["difficulty_tables"] = {}
    for tname, entries in tables.items():
        per = {k: len(set(entries) & v) for k, v in chart_sets.items()}
        report["difficulty_tables"][tname] = {
            "table_entries_with_sha256": len(entries),
            "per_player_matched": per,
            "union_matched": len(set(entries) & union_charts),
        }

    json.dump(report, open(OUT / "scorelog_audit.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)

    # ---------- plots ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # weekly play activity
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, (score, log, dl) in frames.items():
        d = pd.to_datetime(log["date"], unit="s", errors="coerce")
        weekly = pd.Series(1, index=d).resample("W").sum()
        ax.plot(weekly.index, weekly.values, label=name, lw=1)
    ax.set_title("plays per week (scorelog)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "weekly_plays.png", dpi=140)
    plt.close(fig)

    # cumulative unique charts over time
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, (score, log, dl) in frames.items():
        d = pd.to_datetime(log["date"], unit="s", errors="coerce")
        u = log.assign(d=d).dropna(subset=["d"]).sort_values("d").drop_duplicates(subset=["sha256", "mode"])
        cum = u.groupby(u["d"].dt.to_period("M").dt.to_timestamp()).size().cumsum()
        ax.plot(cum.index, cum.values, label=name)
    ax.set_title("cumulative unique (chart,mode) first-played per month")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "cumulative_charts.png", dpi=140)
    plt.close(fig)

    # first-play accuracy distribution per player (from scorelog: ex approximated? scorelog has EX score)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
    for ax, (name, (score, log, dl)) in zip(axes, frames.items()):
        first = log.sort_values("date").drop_duplicates(subset=["sha256", "mode"])
        # notes from score table to convert EX score to accuracy
        notes = score.set_index(["sha256", "mode"])["notes"]
        acc = []
        for _, row in first.iterrows():
            key = (row["sha256"], row["mode"])
            n = notes.get(key, np.nan)
            if n and n > 0:
                acc.append(row["score"] * 50.0 / n)
        ax.hist(acc, bins=40, color="tab:blue", alpha=0.8)
        ax.set_title(f"{name}: first-play acc n={len(acc)}")
        ax.set_xlabel("accuracy (%)")
    fig.tight_layout()
    fig.savefig(PLOTS / "first_play_accuracy.png", dpi=140)
    plt.close(fig)

    # Satellite difficulty coverage per player
    if tables.get("satellite_table"):
        sat = tables["satellite_table"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), sharey=True)
        for ax, (name, cs) in zip(axes, chart_sets.items()):
            levels = [sat[s] for s in cs if s in sat]
            lvl = pd.Series(levels).value_counts().sort_index()
            ax.bar(lvl.index.astype(str), lvl.values, color="tab:green", alpha=0.85)
            ax.set_title(f"{name}: Satellite coverage n={len(levels)}")
            ax.set_xlabel("sl level")
        fig.tight_layout()
        fig.savefig(PLOTS / "satellite_coverage.png", dpi=140)
        plt.close(fig)

    print(json.dumps(report["overlap"], indent=2))
    print(json.dumps(report["id_alignment"], indent=2))
    print("report ->", OUT / "scorelog_audit.json")
    print("plots  ->", PLOTS)


if __name__ == "__main__":
    main()
