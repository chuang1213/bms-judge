"""Phase 3 v0 dataset: player-chart first-play prediction samples.

Data semantics (see PHASE3_AUDIT.md):
- scorelog is an improvement log, NOT a complete play log. The earliest row per
  (player, sha256) is the exact first play (oldscore=0 by construction).
- mode column encodes play options: mode>=100 or len(sha256)!=64 => course rows, dropped.
- first plays with clear==NO_PLAY(0) are aborted/practice plays, dropped (v0).
- acc = first-play EX score * 50 / manifest total_notes (manifest notes, not score.notes).

v0 scope (confirmed 2026-09-03): targets restricted to sl/st/insane tables AND manifest;
players = chuang/muiclac/tzh/nanji (buzhang dropped).

Time split per player: T_train = q50, T_test = q75 of that player's first-play times.
train samples: targets first-played in (T_train, T_test], features from history <= T_train.
test samples:  targets first-played in (T_test, end],   features from history <= T_test.
Each first-play event is used as a target at most once; history features never include
events after the cutoff.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "玩家资料"
CORPUS = ROOT / "bms_ml" / "output" / "corpus"
TABLES = ROOT / "bms_ml" / "output" / "tables"
OUT = ROOT / "bms_ml" / "output" / "phase3" / "dataset"


def load_roster() -> tuple[dict, dict]:
    """Active players and their time modes come from players.json (managed by
    ingest_player.py). include=false players are skipped entirely."""
    roster = json.load(open(Path(__file__).resolve().parent / "players.json",
                            encoding="utf-8"))
    active = {n: e for n, e in roster["players"].items() if e.get("include")}
    return ({n: e["dir"] for n, e in active.items()},
            {n: e.get("time", "real") for n, e in active.items()})


PLAYERS, TIME_MODE = load_roster()
TRAIN_Q, TEST_Q = 0.50, 0.75


def load_manifest() -> pd.DataFrame:
    rows = []
    with open(CORPUS / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "sha256": r["sha256"],
                "md5": r["md5"],
                "title": r.get("title"),
                "notes": r["meta"]["total_notes"],
                "features": r["features"],
                "quarantine": r.get("quarantine"),
            })
    df = pd.DataFrame(rows)
    feat = pd.DataFrame(df.pop("features").tolist(),
                        columns=json.load(open(CORPUS / "analysis" / "features_schema.json"))["names"])
    feat.columns = [f"c_{c}" for c in feat.columns]
    return pd.concat([df.reset_index(drop=True), feat], axis=1)


def load_tables() -> pd.DataFrame:
    """sha256 -> (table_name, level). Priority satellite > stella > insane."""
    md5_to_sha: dict[str, str] = {}
    with open(CORPUS / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            md5_to_sha[r["md5"]] = r["sha256"]
    out = {}
    for name, fname in [("satellite", "satellite_data.json"), ("stella", "stella_data.json"),
                        ("insane", "insane_data.json")]:
        for e in json.load(open(TABLES / fname)):
            sha = e.get("sha256") or md5_to_sha.get((e.get("md5") or "").lower())
            if not sha:
                continue
            lvl = e.get("level")
            try:
                lvl = float(lvl)
            except (TypeError, ValueError):
                continue  # '??' etc.
            if sha not in out:  # first table wins (priority order above)
                out[sha] = (name, lvl)
    return pd.DataFrame([(k, v[0], v[1]) for k, v in out.items()],
                        columns=["sha256", "table", "level"])


def load_firstplays() -> pd.DataFrame:
    """First-play events per player, one row per (player, sha256)."""
    parts = []
    for player, rel in PLAYERS.items():
        con = sqlite3.connect(ROOT / rel / "scorelog.db")
        df = pd.read_sql_query(
            "SELECT rowid, sha256, mode, clear, score, minbp, date FROM scorelog", con)
        con.close()
        df = df[(df["sha256"].str.len() == 64) & (df["mode"].astype(int) < 100)]
        df = df.sort_values(["date", "rowid"]).drop_duplicates("sha256", keep="first")
        df = df[df["clear"] != 0]  # NO_PLAY first rows: aborted/practice, not a real attempt
        # ex==0 first rows are non-attempts too (immediate quit); they also carry the
        # minbp=INT32_MAX sentinel (observed once in chuang). See PHASE3_AUDIT.md §9.
        df = df[df["score"] > 0]
        if TIME_MODE.get(player) == "synthetic":
            # clients without reliable timestamps (LR2): play-order ordinal days.
            # ordering preserved, absolute-time semantics lost (see PROTOCOL.md)
            df = df.sort_values(["date", "rowid"])
            df["date"] = (np.arange(len(df), dtype=np.int64) + 1) * 86400
        parts.append(pd.DataFrame({
            "player": player,
            "sha256": df["sha256"].values,
            "time": pd.to_datetime(df["date"].values, unit="s"),
            "lamp": df["clear"].values,       # first-play lamp (exact, see module docstring)
            "ex": df["score"].values,         # first-play EX score (= best-after on first row)
            "bp": df["minbp"].values,
        }))
    return pd.concat(parts, ignore_index=True)


def build_history_features(fp: pd.DataFrame, log_counts: pd.DataFrame,
                           cutoff: pd.Timestamp) -> pd.DataFrame:
    """Per-player history statistics strictly from events at or before `cutoff`.

    fp: all first-play rows (any player). log_counts: (player, time) of all scorelog rows.
    """
    past = fp[fp["time"] <= cutoff].copy()
    past["bp_ratio"] = past["bp"] / past["notes"]
    rows = []
    for player, g in past.groupby("player"):
        known = g.dropna(subset=["acc"])
        last_active = log_counts.loc[log_counts["player"] == player, "time"].max()
        recent = log_counts[(log_counts["player"] == player)
                            & (log_counts["time"] > cutoff - pd.Timedelta(days=30))]
        rows.append({
            "player": player,
            "h_n_firstplays": len(g),
            "h_n_known_acc": len(known),
            "h_acc_mean": known["acc"].mean() if len(known) else np.nan,
            "h_acc_std": known["acc"].std() if len(known) else np.nan,
            "h_acc_last10": known.sort_values("time")["acc"].tail(10).mean() if len(known) else np.nan,
            "h_bp_mean": known["bp"].mean() if len(known) else np.nan,
            "h_bp_ratio_mean": known["bp_ratio"].mean() if len(known) else np.nan,
            "h_fail_rate": (g["lamp"] == 1).mean(),
            "h_fc_rate": (g["lamp"] >= 8).mean(),
            "h_days_since_active": (cutoff - last_active).total_seconds() / 86400.0,
            "h_plays_last30d": len(recent),
            "h_days_span": (cutoff - g["time"].min()).total_seconds() / 86400.0,
        })
    return pd.DataFrame(rows)


def knn_acc_feature(past_Z: np.ndarray, past_acc: np.ndarray, target_z: np.ndarray,
                    k: int = 20) -> float:
    """Mean acc of the player's k nearest past charts in standardized 26-dim stat
    space — the objective, difficulty-table-free replacement for h_level_acc."""
    if len(past_acc) == 0:
        return np.nan
    d = np.sqrt(((past_Z - target_z) ** 2).sum(1))
    idx = np.argsort(d)[:k]
    return float(np.mean(past_acc[idx]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    tables = load_tables()
    fp = load_firstplays()

    # chart-side annotation
    fp = fp.merge(manifest, on="sha256", how="left")
    fp = fp.merge(tables, on="sha256", how="left")
    fp["acc"] = np.where(fp["notes"] > 0, fp["ex"] * 50.0 / fp["notes"], np.nan)
    # BP plausibility guard: BP counts misses, cannot exceed the note count by much
    fp = fp[fp["bp"] <= fp["notes"] + 5].copy()
    # v0 restricted targets to sl/st/insane tables; since h_knn_acc (objective kNN in
    # stat space) replaced the table-level feature (PHASE3_2_REPORT.md §7), the table
    # filter is dropped — target = any chart with manifest stats. `table`/`level`
    # columns are kept only for scope-comparison subsetting.
    fp = fp[fp["notes"].notna() & (fp["notes"] > 0)].copy()
    fp = fp.reset_index(drop=True)  # positional alignment for the kNN stat matrix
    fp["bp_ratio"] = fp["bp"] / fp["notes"]  # normalized BP: misses per note

    # all scorelog rows (activity features need the full row stream, not just first plays)
    log_rows = []
    for player, rel in PLAYERS.items():
        con = sqlite3.connect(ROOT / rel / "scorelog.db")
        df = pd.read_sql_query("SELECT date FROM scorelog", con)
        con.close()
        log_rows.append(pd.DataFrame({"player": player,
                                      "time": pd.to_datetime(df["date"], unit="s")}))
    log_counts = pd.concat(log_rows, ignore_index=True)

    # time cutoffs per player
    cut = fp.groupby("player")["time"].quantile([TRAIN_Q, TEST_Q]).unstack()
    cut.columns = ["T_train", "T_test"]
    print("cutoffs:\n", cut)

    # standardized stat space for the objective kNN history feature
    from sklearn.preprocessing import StandardScaler
    stat_cols = [c for c in fp.columns if c.startswith("c_")]
    scaler = StandardScaler().fit(fp[stat_cols].values)
    fp_Z = np.nan_to_num(scaler.transform(fp[stat_cols].values))

    samples = []
    for player, row in cut.iterrows():
        pf = fp[fp["player"] == player]
        pf_pos = pf.index.values
        pf_Z = fp_Z[pf.index.values]
        pf_times = pf["time"].values
        pf_acc = pf["acc"].values
        # train: targets in (T_train, T_test], features at T_train (prediction time)
        # test:  targets in (T_test, end],    features at T_test
        phases = [("train", row.T_train, row.T_test),
                  ("test", row.T_test, pf["time"].max())]
        for phase, lo, hi in phases:
            targets = pf[(pf["time"] > lo) & (pf["time"] <= hi)]
            if not len(targets):
                continue
            hist = build_history_features(fp, log_counts, hi)
            hs = hist[hist["player"] == player].iloc[0]
            n_past = int((pf_times <= np.datetime64(hi)).sum())
            past_Z = pf_Z[:n_past]
            past_acc = pf_acc[:n_past]
            tz = scaler.transform(targets[stat_cols].values)
            t = targets.copy()
            t["phase"] = phase
            t["cutoff"] = hi
            t["h_knn_acc"] = [knn_acc_feature(past_Z, past_acc, z)
                              for z in tz]
            for k, v in hs.items():
                if k != "player":
                    t[k] = v
            samples.append(t)
    samples = pd.concat(samples, ignore_index=True)

    fp.to_parquet(OUT / "firstplays.parquet")
    samples.to_parquet(OUT / "samples.parquet")

    # summary
    summ = samples.groupby(["phase", "player"]).agg(
        n=("sha256", "size"), acc_mean=("acc", "mean"), fail_rate=("lamp", lambda s: (s == 1).mean()))
    print(summ.round(3))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
