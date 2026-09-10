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
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# Archive tree was reorganised 2026-09-11: 玩家资料/<name>/player1 -> PlayerData/beatoraja/<name>/player1
# (per-player paths live in players.json; this constant is only the tree root).
DATA = ROOT / "PlayerData" / "beatoraja"
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
# Survival-scope experiment (user decision 2026-09-04): drop FAILED and acc<50%
# first plays from BOTH targets and history — models only see completed,
# non-collapsed attempts (lamp 4/5/6 = EASY/NORMAL/HARD). Revert by setting False.
SURVIVAL_SCOPE_ONLY = False

# The sl/st/発狂2018 fence is a TARGET-space quality control (PROTOCOL.md §1,
# PHASE3_AUDIT §12: "训练/评估目标限定...表并集内的谱面"). But the fence is applied
# before history construction, so it also censors what the model may know about a
# player: 13,016 / 38,120 parseable first plays (34%) never enter any history
# feature — 66% of yangtao's, 54% of hl's, ~48% of reiaki/muiclac/vsoflan's.
#
# OFF by default (preserves current behaviour exactly). Flip to True to let
# off-table-but-parseable plays count as player history. This is a protocol-level
# question, not a bug fix — measure before adopting. See PHASE3_5_REVIEW.md.
# A/B without editing the file:
#   P3_HISTORY_OFFTABLE=1 python bms_ml/phase3/data.py
HISTORY_USES_OFFTABLE = os.environ.get("P3_HISTORY_OFFTABLE", "0") == "1"


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
                "c_jrank": r.get("rank"),  # #RANK judge window tier (parser default 2)
            })
    df = pd.DataFrame(rows)
    # The corpus library contains the SAME chart file under 2-3 paths (e.g.
    # BMS/gremlin_ogg/x.bms and BMS/GREMLIN/x.bms), so manifest.jsonl has 329
    # duplicated sha256 (655 rows). Merging on sha256 without dedup duplicated
    # first-play events: 242 rows of samples.parquet were exact key-duplicates
    # (128 test / 114 train = 2.0% of test double-counted), and because the two
    # copies carry the SAME timestamp, one of them had its own chart inside its
    # strict-prior history window - a subtle self-leak in h_knn_acc. Fixed
    # 2026-09-11 (see EXPERIMENT_LOG). Keep the manifest 1:1 on sha256.
    df = df.drop_duplicates("sha256", keep="first").reset_index(drop=True)
    feat = pd.DataFrame(df.pop("features").tolist(),
                        columns=json.load(open(CORPUS / "analysis" / "features_schema.json",
                                               encoding="utf-8"))["names"])
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
        # encoding is explicit: the table JSONs are UTF-8 and a non-UTF-8 locale
        # (cp936/gbk on zh-CN Windows) otherwise raises UnicodeDecodeError here,
        # which breaks `python data.py` — step 4 of the new-player SOP.
        for e in json.load(open(TABLES / fname, encoding="utf-8")):
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
        if SURVIVAL_SCOPE_ONLY:
            # lamp part of the survival filter; the acc>=50 part needs `notes`
            # from the manifest merge, applied in main()
            df = df[(df["clear"] >= 4) & (df["clear"] <= 6)]
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


def build_history_features(fp_sorted: pd.DataFrame, log_counts: pd.DataFrame,
                           Z: np.ndarray, k: int = 20) -> pd.DataFrame:
    """Per-sample history = the player's first plays STRICTLY BEFORE the target's own
    first-play time — the player state at the moment they walk up to the target chart.

    Corrects the Phase 3.1-3.3 bug where history windows were bounded by the phase
    cutoff (or the player's last play), which included the target's own outcome in
    h_knn_acc/h_acc_mean and, for test rows, the whole test window (leakage).
    Time features are bounded by the target time, never by a cutoff. Returns a
    DataFrame aligned 1:1 with fp_sorted rows."""
    n = len(fp_sorted)
    acc = fp_sorted["acc"].values
    lamp = fp_sorted["lamp"].values
    bp = fp_sorted["bp"].values
    bp_ratio = (fp_sorted["bp"] / fp_sorted["notes"]).values
    known = ~np.isnan(acc)

    cols = {c: np.full(n, np.nan) for c in [
        "h_n_firstplays", "h_n_known_acc", "h_acc_mean", "h_acc_std", "h_acc_last10",
        "h_bp_mean", "h_bp_ratio_mean", "h_fail_rate", "h_fc_rate",
        "h_days_since_active", "h_plays_last30d", "h_days_span", "h_knn_acc"]}

    log_by = {p: np.sort(g["time"].values.astype("datetime64[s]").astype(np.int64))
              for p, g in log_counts.groupby("player")}
    times = fp_sorted["time"].values.astype("datetime64[s]").astype(np.int64)

    for player, idx in fp_sorted.groupby("player", sort=False).indices.items():
        pos = np.asarray(idx)
        r = len(pos)
        t = times[pos]
        lt = log_by[player]
        known_p = known[pos]
        # prefix sums over the player's chronological first plays
        cs_known = np.concatenate([[0], np.cumsum(known_p)])
        cs_acc = np.concatenate([[0.0], np.cumsum(np.where(known_p, acc[pos], 0.0))])
        cs_acc2 = np.concatenate([[0.0], np.cumsum(np.where(known_p, acc[pos] ** 2, 0.0))])
        cs_fail = np.concatenate([[0], np.cumsum(lamp[pos] == 1)])
        cs_fc = np.concatenate([[0], np.cumsum(lamp[pos] >= 8)])
        cs_bp = np.concatenate([[0.0], np.cumsum(bp[pos])])
        cs_bpr = np.concatenate([[0.0], np.cumsum(bp_ratio[pos])])
        kp = np.flatnonzero(known_p)          # ranks with known acc
        # kNN among strictly-prior rows, chunked
        Zi = Z[pos]
        knn = np.full(r, np.nan)
        lo0 = 0
        while lo0 < r:
            hi0 = min(lo0 + 256, r)
            if hi0 == 0:
                break
            D = np.sqrt(((Zi[lo0:hi0, None, :] - Zi[None, :hi0, :]) ** 2).sum(-1))
            for j in range(lo0, hi0):
                drow = D[j - lo0, :j]
                if len(drow) == 0:
                    continue
                kk = min(k, len(drow))
                near = np.argpartition(drow, kk - 1)[:kk]
                knn[j] = np.mean(acc[pos][near])
            lo0 = hi0
        for j in range(r):
            c_known = cs_known[j]
            cols["h_n_firstplays"][pos[j]] = j
            cols["h_n_known_acc"][pos[j]] = c_known
            if c_known:
                cols["h_acc_mean"][pos[j]] = cs_acc[j] / c_known
                if c_known > 1:
                    m = cs_acc[j] / c_known
                    var = max(cs_acc2[j] / c_known - m * m, 0.0)
                    cols["h_acc_std"][pos[j]] = np.sqrt(var)
            cnt10 = np.searchsorted(kp, j)
            if cnt10:
                sel = kp[max(0, cnt10 - 10):cnt10]
                cols["h_acc_last10"][pos[j]] = np.mean(acc[pos[sel]])
            cols["h_bp_mean"][pos[j]] = cs_bp[j] / j if j else np.nan
            cols["h_bp_ratio_mean"][pos[j]] = cs_bpr[j] / j if j else np.nan
            cols["h_fail_rate"][pos[j]] = cs_fail[j] / j if j else np.nan
            cols["h_fc_rate"][pos[j]] = cs_fc[j] / j if j else np.nan
            cols["h_knn_acc"][pos[j]] = knn[j]
            ts = t[j]
            hi_b = np.searchsorted(lt, ts, side="left")
            lo_b = np.searchsorted(lt, ts - 30 * 86400, side="left")
            cols["h_plays_last30d"][pos[j]] = float(hi_b - lo_b)
            if hi_b:
                cols["h_days_since_active"][pos[j]] = (ts - lt[hi_b - 1]) / 86400.0
            if j:
                cols["h_days_span"][pos[j]] = (ts - t[0]) / 86400.0
    return pd.DataFrame(cols)


def knn_acc_feature(past_Z: np.ndarray, past_acc: np.ndarray, target_z: np.ndarray,
                    k: int = 20) -> float:
    """(kept for reference) Mean acc of the k nearest past charts in standardized
    26-dim stat space."""
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
    # Invariant: exactly one row per (player, chart). A violated key both
    # double-weights the row in train/test and, since the copies share a timestamp,
    # lets a duplicate of the target chart sit inside its own strict-prior history
    # window (self-leak in h_knn_acc). Guard against regressions of the 2026-09-11 fix.
    assert not fp.duplicated(["player", "sha256"]).any(), \
        "duplicate (player, sha256) first-play rows: a reference table is not 1:1 on sha256"
    fp["acc"] = np.where(fp["notes"] > 0, fp["ex"] * 50.0 / fp["notes"], np.nan)
    # BP plausibility guard: BP counts misses, cannot exceed the note count by much
    fp = fp[fp["bp"] <= fp["notes"] + 5].copy()
    # Sample space (user decision 2026-09-03): targets restricted to the sl/st/発狂2018
    # union — off-table charts are quality-uncontrolled. FEATURES remain table-free
    # (h_knn_acc + 26D stats; see chart_repr.py contract and PROTOCOL.md §1).
    # Parseable = we have manifest notes + the objective stats, needed for acc and
    # for the kNN stat space.
    fp = fp[fp["notes"].notna() & (fp["notes"] > 0)].copy()
    if SURVIVAL_SCOPE_ONLY:
        # acc>=50 part: acc = ex*50/notes >= 50  <=>  ex >= notes
        fp = fp[fp["acc"] >= 50].copy()
    # `is_target` = eligible to be PREDICTED (in-table). The history frame may be
    # wider when HISTORY_USES_OFFTABLE is on.
    fp["is_target"] = fp["table"].notna()
    if not HISTORY_USES_OFFTABLE:
        fp = fp[fp["is_target"]].copy()
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

    # time cutoffs per player (define the train/test split of TARGETS only;
    # features never use them — see build_history_features)
    # Cutoffs come from the TARGET timeline only, so they do not shift when the
    # history frame widens (keeps train/test bands comparable across the flag).
    cut = fp[fp["is_target"]].groupby("player")["time"].quantile([TRAIN_Q, TEST_Q]).unstack()
    cut.columns = ["T_train", "T_test"]
    print("cutoffs:")
    print(cut)

    from sklearn.preprocessing import StandardScaler
    stat_cols = [c for c in fp.columns if c.startswith("c_")]
    scaler = StandardScaler().fit(fp[stat_cols].values)
    fp_sorted = fp.sort_values(["player", "time"]).reset_index(drop=True)
    Z = np.nan_to_num(scaler.transform(fp_sorted[stat_cols].values)).astype(np.float64)

    H = build_history_features(fp_sorted, log_counts, Z)
    for c in H.columns:
        fp_sorted[c] = H[c].values

    t_sec = fp_sorted["time"]
    Ttr = fp_sorted["player"].map(cut["T_train"])
    Tte = fp_sorted["player"].map(cut["T_test"])
    phase = np.where(t_sec <= Ttr, "history", np.where(t_sec <= Tte, "train", "test"))
    fp_sorted["phase"] = phase
    fp_sorted["phase_cutoff"] = np.where(phase == "train", Ttr, Tte)
    samples = fp_sorted[(fp_sorted["phase"] != "history")
                        & fp_sorted["is_target"]].reset_index(drop=True)
    fp = fp_sorted

    fp.to_parquet(OUT / "firstplays.parquet")
    samples.to_parquet(OUT / "samples.parquet")

    # summary
    summ = samples.groupby(["phase", "player"]).agg(
        n=("sha256", "size"), acc_mean=("acc", "mean"), fail_rate=("lamp", lambda s: (s == 1).mean()))
    print(summ.round(3))
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
