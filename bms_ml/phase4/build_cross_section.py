"""Phase 4 step 3 — build the cross-sectional (player x chart) best-score table.

WHAT THIS IS
------------
One row per (player, client, sha256) describing that player's CURRENT BEST
recorded result on that chart. No timestamps, no first plays, no causal windows:
Phase 4 explicitly discards play time (PROTOCOL: "舍弃游玩时间，统一横向 schema").
The prediction unit is therefore

    (player, client, sha256, best_score, best_lamp, best_bp, notes)

and the model task is later defined as: given a player's observed cells, predict
the cell for a chart they have never played.

SOURCES AND WHAT EACH ONE ACTUALLY CONTAINS (verified by probes, 2026-09-12)
--------------------------------------------------------------------------
beatoraja `score.db`:
  * table `score` **is** one row per (sha256, mode) = current best, with
    `playcount`/`clearcount` for that chart. It is NOT a play log.
    (The historical per-play log is `scorelog.db`, which only records
    IMPROVEMENTS: an entry exists only when a play beat the previous best, and
    its `score` column is on a different scale from the judgement counts.
    So the "fold to best" work is already done for us — `score` wins over
    replaying `scorelog`.)
  * `player` table is a DAILY AGGREGATE log (one row per active day), not a chart
    table. It is not used here.
  * `notes` in `score` is the chart's note count as beatoraja knew it. It agrees
    with the corpus manifest for 83% of joined rows and is usually a few notes
    larger otherwise (chart revisions), so both are kept (`notes`,
    `manifest_notes`) and `notes_source` records the choice.
  * beatoraja does NOT store an accuracy column. Accuracy has to be rebuilt from
    the judgement counts, and the formula is NOT the LR2 one:

        acc = (2*(epg+lpg) + (egr+lgr)) / (2 * sum(all judgement columns)) * 100

    Key detail: beatoraja splits perfect/great into a "large" (normal) and a
    plain band (`lpg`/`lgr` are the majority of judgements). Using only
    `epg`/`egr` — i.e. the LR2 formula — gives a median of 45.6% and values up
    to 304%; including the large bands gives a median of 75.5% and a hard max of
    100.00 across all 18 archives. The denominator is the judgement sum, not
    `notes`, because the score table's `notes` is unreliable for ~15% of rows
    (median notes/judgement-sum ratio 0.99, so the sum is the self-consistent
    denominator).

LR2 (`lr2_reader.py`):
  * one row per chart = current best, with judgement counts, `minbp`, `clear`,
    play counters. `acc = (perfect*2 + great) / (2 * totalnotes) * 100`, already
    validated against the archive's own `rate` column (max |diff| 0.999).

THE TWO CLIENTS ARE NOT POOLED
------------------------------
PROTOCOL + PROTOCOL §1: separate matrices, separate reports. Their score scales
are not interchangeable (measured gap sd 10.16pp) and their lamp enums differ
(LR2 archive only carries 0..5; beatoraja carries its own 0..9 ladder). This
script therefore only ever emits a `client` column and never blends the two.

CHART UNIVERSE
--------------
Canonical charts = the corpus manifest rows that are 7-key playable
(`has_7k`). 48,290 unique sha256 in the manifest; a handful of duplicate-content
files collapse to one canonical row (`chart_row_rank == 0`). DP/PMS/bmson rows
are excluded at the `mode_kind` level: a `player_mode` other than 1P/2P gets
`mode_kind = "other"` so downstream code cannot silently mix 7-key and DP
matrices.

Outputs
-------
  bms_ml/output/phase4/dataset/cross_section.parquet
  bms_ml/output/phase4/build_cross_section.json   (provenance + drop accounting)
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ROOT, header, write_result  # noqa: E402

MANIFEST = ROOT / "bms_ml" / "output" / "corpus" / "manifest.jsonl"
PLAYERS_JSON = ROOT / "bms_ml" / "phase3" / "players.json"
FEATURES_SCHEMA = ROOT / "bms_ml" / "output" / "corpus" / "analysis" / "features_schema.json"
DEFAULT_OUT = ROOT / "bms_ml" / "output" / "phase4" / "dataset" / "cross_section.parquet"

# The 27 objective chart statistics from the corpus manifest. They are CHART-level
# (not player-level) and carry no time, no play history and no difficulty-table
# level, so they are admissible as the cold-chart content baseline required by
# PHASE4_PROTOCOL §6.3. `cs_` prefix keeps them visibly distinct from outcome columns.
#
# The ORDER below is the authoritative one: it is `bms_ml/features.py::FEATURE_NAMES`,
# which is what the corpus builder actually used to write `manifest.jsonl`. Do NOT
# reorder. `features_schema.json` (a side artifact) lists 27 names for a 26-value
# vector, so binding by index against that file would silently shift every feature
# by one; this list is checked against the real vector length at load time.
CONTENT_PREFIX = "cs_"
CONTENT_FEATURES = [
    "total_notes", "ln_ratio", "duration_sec", "measures", "initial_bpm", "min_bpm",
    "max_bpm", "bpm_change_count", "stop_count", "stop_total_sec", "lane0_scratch",
    "lane1", "lane2", "lane3", "lane4", "lane5", "lane6", "lane7", "scratch_ratio",
    "avg_nps", "peak_nps_1s", "peak_measure_nps", "chord_count", "chord2_count",
    "chord3plus_count", "jack_count",
]


def content_feature_names() -> list[str]:
    """Authoritative chart-feature order, cross-checked against the manifest itself."""
    names = CONTENT_FEATURES
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            n = len(json.loads(line).get("features") or [])
            if n != len(names):
                raise ValueError(
                    f"manifest feature vector has {n} values but {len(names)} names are "
                    f"declared; binding by index would mislabel every feature. Fix the "
                    f"name list against bms_ml/features.py::FEATURE_NAMES before building.")
            break
    return names

# beatoraja judgement columns. Sum(all) is the play's total judgements; the
# weighted numerator counts large + plain perfect as 2 and large + plain great
# as 1, which is the same EX relation LR2 uses.
BJ_JUDGE = ["epg", "lpg", "egr", "lgr", "egd", "lgd", "ebd", "lbd", "epr", "lpr", "ems", "lms"]
BJ_ACCURATE = ("epg", "lpg", "egr", "lgr")

# Chart universe / identity columns carried from the manifest.
CHART_COLS = ["sha256", "md5", "chart_key", "chart_row_rank", "manifest_sha_rows",
              "player_mode", "mode_kind", "has_7k", "quarantine", "flags",
              "total_notes", "ln_ratio", "duration_sec", "avg_nps", "peak_nps_1s"]
# Deterministic public schema. Fixed so that a stale parquet cannot be silently
# consumed by a script written against a different shape.
SCHEMA = [
    "player", "client", "sha256", "md5", "mode_kind",          # identity
    "acc", "acc_own_notes", "notes", "notes_source", "manifest_notes",  # target + denominator
    "lamp", "lamp_name", "bp", "playcount", "clearcount",      # outcome detail
    "chart_key", "chart_row_rank", "manifest_sha_rows", "has_7k",  # chart identity
    "player_mode", "quarantine", "flags",                      # chart provenance
    "total_notes", "ln_ratio", "duration_sec", "avg_nps", "peak_nps_1s",
    "notes_match_manifest", "n_judgements", "n_modes_collapsed", "acc_alt",
    "included_in_phase3",
] + [CONTENT_PREFIX + n for n in content_feature_names()]

# LR2 lamp ladder (from lr2_reader; the archive only carries 0..5).
LR2_LAMP = {0: "NO_PLAY", 1: "FAILED", 2: "EASY", 3: "NORMAL", 4: "HARD", 5: "EXHARD",
            6: "FC", 7: "PERFECT", 8: "MAX", 9: "ASSIST", 10: "L_ASSIST"}
# beatoraja lamp ladder (from beatoraja's own clear enum).
BJ_LAMP = {0: "NO_PLAY", 1: "FAILED", 2: "ASSIST_EASY", 3: "LIGHT_ASSIST_EASY", 4: "EASY",
           5: "NORMAL", 6: "HARD", 7: "EXHARD", 8: "FC", 9: "PERFECT", 10: "MAX"}


# --------------------------------------------------------------------------- #
# corpus manifest
# --------------------------------------------------------------------------- #
def load_manifest() -> pd.DataFrame:
    """Canonical chart table: one row per unique sha256, 7-key playable only."""
    names = content_feature_names()
    recs = []
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            feats = r.get("features") or []
            row = {
                "sha256": r["sha256"],
                "md5": r["md5"],
                "player_mode": r["player_mode"],
                "has_7k": bool(r["has_7k"]),
                "quarantine": list(r["quarantine"]),
                "flags": list(r["flags"]),
                "total_notes": (r.get("meta") or {}).get("total_notes"),
                "ln_ratio": (r.get("meta") or {}).get("ln_ratio"),
                "duration_sec": (r.get("meta") or {}).get("duration_sec"),
                "avg_nps": (r.get("meta") or {}).get("avg_nps"),
                "peak_nps_1s": (r.get("meta") or {}).get("peak_nps_1s"),
            }
            for i, n in enumerate(names):
                row[CONTENT_PREFIX + n] = float(feats[i]) if i < len(feats) else np.nan
            recs.append(row)
    m = pd.DataFrame(recs)
    m["manifest_sha_rows"] = m.groupby("sha256")["sha256"].transform("size")
    # rank within a duplicated-content group; rank 0 is the canonical row and is
    # the only one that survives below (the others are byte-identical charts).
    m["chart_row_rank"] = m.groupby("sha256").cumcount()
    m = m.sort_values(["sha256", "chart_row_rank"], kind="stable").reset_index(drop=True)

    out = []
    for sha, grp in m.groupby("sha256", sort=True):
        keep = grp.loc[grp["chart_row_rank"] == 0].iloc[0]
        row = keep.to_dict()
        # has_7k is OR'd across the duplicate group: the group is one chart file
        # in several copies, so a 7-key variant anywhere makes it 7-key playable.
        row["has_7k"] = bool(grp["has_7k"].any())
        row["chart_key"] = sha[:16]
        out.append(row)
    c = pd.DataFrame(out)
    c = c[c["has_7k"]].copy()
    c["mode_kind"] = np.where(c["player_mode"].isin(["1P", "2P"]), "sp", "other")
    # quarantine/flags kept as JSON-ish strings so the parquet stays flat & portable
    c["quarantine"] = c["quarantine"].apply(lambda v: "|".join(v))
    c["flags"] = c["flags"].apply(lambda v: "|".join(v))
    return c.reset_index(drop=True)


def manifest_stats() -> dict:
    """Audit numbers for the clean manifest, before any client join."""
    n_rows = 0
    n_uniq = set()
    n_7k = 0
    with open(MANIFEST, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n_rows += 1
            n_uniq.add(r["sha256"])
            n_7k += bool(r["has_7k"])
    return {"manifest_rows": n_rows, "manifest_unique_sha256": len(n_uniq),
            "manifest_rows_has_7k": n_7k,
            "manifest_duplicate_content_rows": n_rows - len(n_uniq)}


# --------------------------------------------------------------------------- #
# beatoraja
# --------------------------------------------------------------------------- #
def _beatoraja_acc(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(acc over judgement sum, judgement sum) for beatoraja score rows."""
    jsum = df[BJ_JUDGE].sum(axis=1).astype(float)
    num = (2.0 * (df["epg"] + df["lpg"]) + (df["egr"] + df["lgr"])).astype(float)
    acc = np.where(jsum > 0, num * 100.0 / (2.0 * jsum), np.nan)
    return pd.Series(acc, index=df.index), jsum


def read_beatoraja(name: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Current-best rows for one beatoraja player, collapsed to one per sha256."""
    db = ROOT / cfg["dir"] / "score.db"
    if not db.exists():
        return pd.DataFrame(), {"player": name, "client": "beatoraja", "status": "missing_score_db",
                                "why": "score.db absent; scorelog.db is an improvement log and "
                                       "carries no judgement counts, so no current-best table "
                                       "can be reconstructed"}
    con = sqlite3.connect(db)
    sc = pd.read_sql_query("SELECT * FROM score", con)
    con.close()

    info = {"player": name, "client": "beatoraja", "status": "ok",
            "db_raw_rows": int(len(sc)),
            "db_bytes": int(db.stat().st_size)}
    info["sha_len_counts"] = {str(k): int(v) for k, v in sc["sha256"].str.len().value_counts().items()}
    # Non-64-char keys are not corpus BMS sha256 (they are beatoraja's alternate
    # hash representations); they cannot join the manifest and are dropped.
    sc = sc[sc["sha256"].str.len() == 64].copy()
    info["rows_after_sha_len_filter"] = int(len(sc))

    acc, jsum = _beatoraja_acc(sc)
    sc["acc"] = acc
    sc["n_judgements"] = jsum
    info["acc_gt_100"] = int((sc["acc"] > 100).sum())
    info["n_judgements_zero"] = int((sc["n_judgements"] <= 0).sum())

    sc["n_modes_collapsed"] = sc.groupby("sha256")["sha256"].transform("size")
    # A sha can appear under several `mode` accents (e.g. the same chart scored
    # with and without an assist option). Keep the BEST recorded result:
    # highest accuracy, then better lamp, then fewer bad.
    order = sc.sort_values(
        ["sha256", "acc", "clear", "minbp", "playcount"],
        ascending=[True, False, False, True, False], kind="stable")
    best = order.drop_duplicates("sha256", keep="first").copy()
    info["rows_after_collapse"] = int(len(best))
    info["dup_sha_rows_collapsed"] = int(len(sc) - len(best))

    best = best.rename(columns={"notes": "notes", "minbp": "bp", "clear": "lamp"})
    best["client"] = "beatoraja"
    best["player"] = name
    best["lamp_name"] = best["lamp"].map(BJ_LAMP).fillna("UNKNOWN")
    best["notes_source"] = "beatoraja_score"
    best["acc_alt"] = np.nan
    best["acc_own_notes"] = np.where(
        best["notes"] > 0,
        (2.0 * (best["epg"] + best["lpg"]) + (best["egr"] + best["lgr"])) * 100.0
        / (2.0 * best["notes"]), np.nan)
    # columns downstream
    best = best[["player", "client", "sha256", "acc", "notes", "notes_source", "lamp",
                 "lamp_name", "bp", "playcount", "clearcount", "n_judgements",
                 "n_modes_collapsed", "acc_alt", "acc_own_notes"]]
    return best, info


# --------------------------------------------------------------------------- #
# LR2
# --------------------------------------------------------------------------- #
def read_lr2(name: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Current-best rows for one LR2 archive. Delegates parsing to lr2_reader."""
    sys.path.insert(0, str(ROOT / "bms_ml" / "phase3"))
    from lr2_reader import load_rows  # noqa: E402

    d = ROOT / cfg["dir"]
    # Each LR2 roster entry pins exactly ONE archive file. `PlayerData/LunaticRave2/`
    # holds archives of DIFFERENT people (V_soflan.db = player "V_soflan",
    # mengye.db = player "Caiwla"), so labelling must never come from a directory
    # glob. The glob fallback below exists only for legacy single-db directories
    # and is reported loudly, because silently merging two people is exactly the
    # class of error that would corrupt the matrix.
    pin = cfg.get("db")
    if pin:
        dbs = [d / pin]
        dbs = [p for p in dbs if p.exists()]
        if not dbs:
            return pd.DataFrame(), {"player": name, "client": "lr2", "status": "pinned_db_missing",
                                    "dir": str(d), "db": pin}
    else:
        dbs = sorted(d.glob("*.db"))
        info_extra = {"note": "no `db` pin: fell back to directory glob"}
        if len(dbs) > 1:
            raise ValueError(
                f"{name}: {d} holds {len(dbs)} archives and the roster entry has no "
                f"`db` pin; refusing to guess which one is {name}")
    frames, infos = [], []
    for db in dbs:
        rows = load_rows(db)
        rows["db"] = db.name
        frames.append(rows)
        infos.append({"db": db.name, "rows": int(len(rows)),
                      "player_id_in_db": sorted(map(str, rows["player"].unique()))})
    raw = pd.concat(frames, ignore_index=True)
    # One roster entry = one archive = one person. A multi-file entry would pool
    # two players into one matrix row set, so it is refused rather than merged.
    if raw["player"].nunique() > 1:
        raise ValueError(f"{name}: archive holds {sorted(raw['player'].unique())} "
                         f"-> that is more than one person; register one roster entry each")
    info = {"player": name, "client": "lr2", "status": "ok", "dbs": infos,
            "db_raw_rows": int(len(raw))}
    if not pin:
        info["note"] = "no `db` pin: fell back to directory glob"

    raw = raw[raw["notes"] > 0].copy()          # reader guarantees >0, kept as a guard
    raw["n_modes_collapsed"] = 1
    raw["n_judgements"] = raw["notes"]
    raw["lamp_name"] = raw["lamp"].map(LR2_LAMP).fillna("UNKNOWN")
    raw["notes_source"] = "lr2_archive"
    # `bj_lamp` is the explicit cross-client mapping; it is carried in the audit
    # only. Pooling the lamp ladders is still forbidden.
    info["lamp_counts"] = {str(k): int(v) for k, v in raw["lamp"].value_counts().sort_index().items()}
    info["clear0_rows"] = int((raw["lamp"] == 0).sum())

    raw["client"] = "lr2"
    raw = raw.rename(columns={"player": "player_id_in_db"})
    raw["player"] = name
    # acc_alt: the same accuracy re-expressed over the manifest note count. Kept
    # so the evaluation can show how much of a result depends on the denominator.
    raw["acc_alt"] = np.nan
    raw["acc_own_notes"] = raw["acc"]
    raw = raw[["player", "client", "sha256", "acc", "notes", "notes_source", "lamp",
               "lamp_name", "bp", "playcount", "clearcount", "n_judgements",
               "n_modes_collapsed", "acc_alt", "acc_own_notes", "player_id_in_db"]]
    info["player_id_in_db"] = sorted(map(str, raw["player_id_in_db"].unique()))
    return raw, info


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def build(clients: tuple[str, ...] = ("beatoraja", "lr2")) -> tuple[pd.DataFrame, dict]:
    roster = json.load(open(PLAYERS_JSON, encoding="utf-8"))["players"]
    recs, player_info = [], []

    for name, cfg in roster.items():
        if cfg["client"] not in clients:
            continue
        if cfg["client"] == "beatoraja":
            df, info = read_beatoraja(name, cfg)
        else:
            df, info = read_lr2(name, cfg)
        info["included_in_phase3"] = bool(cfg["include"])
        player_info.append(info)
        if len(df):
            df["included_in_phase3"] = bool(cfg["include"])
            recs.append(df)

    if not recs:
        raise SystemExit("no client rows read - nothing to build")
    obs = pd.concat(recs, ignore_index=True)

    charts = load_manifest()
    stats = manifest_stats()

    # join chart metadata; sha not in the canonical 7-key universe -> dropped
    obs = obs.merge(charts, on="sha256", how="left", indicator=True)
    joined = obs["_merge"] == "both"
    dropped_unmapped = int((~joined).sum())
    obs = obs[joined].drop(columns=["_merge"]).copy()

    obs["manifest_notes"] = obs["total_notes"]
    # acc_alt: client accuracy re-expressed over the manifest note count. Only
    # defined for clients whose accuracy is an EX relation (both are), and it is
    # the robustness variant, never the primary target.
    obs["acc_alt"] = np.where(
        (obs["manifest_notes"] > 0) & obs["acc"].notna(),
        obs["acc"] * obs["notes"] / obs["manifest_notes"], np.nan)
    obs["notes_match_manifest"] = obs["notes"] == obs["manifest_notes"]

    # md5: beatoraja has none; LR2's md5 comes from the reader's archive key.
    repo_md5 = charts.set_index("sha256")["md5"]
    if "md5" not in obs.columns:
        obs["md5"] = np.nan
    obs["md5"] = obs["md5"].where(obs["md5"].notna(), obs["sha256"].map(repo_md5))

    obs = obs[SCHEMA].sort_values(["client", "player", "sha256"], kind="stable").reset_index(drop=True)

    report = {**stats, "players": player_info,
              "observations_built": int(len(obs)),
              "dropped_unmapped_sha": dropped_unmapped,
              "clients": sorted(obs["client"].unique().tolist()),
              "per_client_rows": {k: int(v) for k, v in obs["client"].value_counts().items()},
              "per_client_players": {k: int(v) for k, v in
                                     obs.groupby("client")["player"].nunique().items()}}
    return obs, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 step 3: build cross_section.parquet")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--clients", default="beatoraja,lr2")
    args = ap.parse_args()

    clients = tuple(c.strip() for c in args.clients.split(",") if c.strip())
    obs, report = build(clients)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    obs.to_parquet(args.out, index=False)

    cfg = {"manifest": str(MANIFEST.relative_to(ROOT)),
           "players_json": str(PLAYERS_JSON.relative_to(ROOT)),
           "clients": list(clients),
           "chart_universe": "manifest has_7k (7-key playable), unique sha256",
           "acc_beatoraja": "(2*(epg+lpg)+(egr+lgr))/(2*sum(judgements))*100",
           "acc_lr2": "(perfect*2+great)/(2*totalnotes)*100",
           "lamp_scales_kept_separate": True}
    out = write_result("build_cross_section.json",
                       header(script="build_cross_section.py", data_config=cfg), report)

    print(f"wrote {args.out.relative_to(ROOT)}  rows={len(obs)}")
    for cl, n in report["per_client_rows"].items():
        print(f"  {cl:<10} rows={n:>6}  players={report['per_client_players'][cl]}")
    print(f"  dropped unmapped sha: {report['dropped_unmapped_sha']}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
