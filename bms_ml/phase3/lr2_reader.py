"""LunaticRave2 archive reader (first real LR2 data landed 2026-09-11).

What an LR2 archive IS (verified against PlayerData/LunaticRave2/V_soflan.db)
---------------------------------------------------------------------------
Two SQLite tables:
  score  - ONE row per chart (keyed by the chart's MD5): judgment counts
           (perfect/great/good/bad/poor), totalnotes, minbp, `clear`, `rank`,
           `rate`, and the play counters playcount / clearcount / failcount,
           plus op_history and rseed for the best play.
  player - one row of account totals.

What it CANNOT give: timestamps (no date column at all) and per-play rows. Both are
load-bearing for this project's protocol:

  * targets are FIRST PLAYS in a causal window (PROTOCOL.md §1) - LR2 rows are
    current BESTS, one per chart, so the target definition does not apply;
  * history must be events strictly EARLIER than the target's own first play
    (PROTOCOL.md §2) - LR2 rows cannot be placed on that timeline.

So an LR2 archive is usable as PLAYER STATE ONLY, and only when the player has some
timestamped archive to anchor the split (the LR2 portion then has to be declared as
a pre-period assumption). This matches the time-ablation verdict (PHASE3_4_TIME_ABLATION
§6): LR2 data is grade B - fine for prediction, degraded for interaction research.

What it ADDS over beatoraja: the counters. beatoraja's scorelog is an improvement log,
so `playcount` is precisely the quantity it cannot reconstruct - 6,439 charts here, with
playcount up to 155. That is real "how much have they ground this chart" information.

Label construction (validated on the real archive)
--------------------------------------------------
  acc  = (perfect*2 + great) / (2 * totalnotes) * 100      max |acc - stored rate| = 0.999
                                                            (rate is the truncated int)
  lamp = `clear`, the gauge-based clear type. NOTE: this archive only contains
         0..5 (NO_PLAY/FAILED/EASY/NORMAL/HARD/EXHARD) - no FC/PERFECT rows - so the
         LR2 lamp scale is NOT directly comparable to a beatoraja archive's.
Chart identity: LR2 keys on MD5; the manifest's md5 -> sha256 bridge joins 84% of the
rows to the parsed corpus (the rest are charts outside our BMS library).

MEASURED: the two clients are NOT interchangeable (2026-09-11)
--------------------------------------------------------------
vsoflan has 4,576 charts in BOTH archives, so the gap is directly observable. Comparing
each client's BEST score on the same chart with the same note denominator (manifest
total_notes, so this is not a notes-counting artefact):

    acc gap (LR2 - beatoraja):  mean +2.14 | median +0.46 | sd 10.16
    |gap| > 1pp on 58.5% of charts, > 5pp on 29.4%;  corr = 0.84
    non-LN charts only (n=2241): mean +4.08, sd 8.35  -> the sd is not an LN effect
    corr(|gap|, ln_ratio) = +0.016 -> it is judgement/recording, not scoring bookkeeping

(The gap mixes the client effect with the fact that the two bests may have been set
years apart, so read +2.14 as a mixture; the sd of 10pp is the number that matters, and
it is larger than the model's own acc MAE of 6.5.)

Conclusion, matching the user's intent to treat the clients separately: DO NOT pool
LR2 and beatoraja rows into one player-state feature set. `data.py` refuses to build
features for any included player whose client is not beatoraja.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "bms_ml" / "output" / "corpus"

# LR2 lamp IDs, from beatoraja's own compatibility comment
# (ref_repo/beatoraja-master/src/bms/player/beatoraja/skin/lr2/LR2SelectSkinLoader.java:
#  "0:NO PLAY, 1:FAILED, 2:EASY, 3:NORMAL, 4:HARD, 5:EXH, 6:FC, 7:PERFECT, 8:MAX, 9:ASSIST, 10:L-ASSIST")
LR2_CLEAR = {0: "NO_PLAY", 1: "FAILED", 2: "EASY", 3: "NORMAL", 4: "HARD",
             5: "EXHARD", 6: "FC", 7: "PERFECT", 8: "MAX", 9: "ASSIST",
             10: "L_ASSIST"}

# beatoraja's own enum, from
# ref_repo/beatoraja-master/src/bms/player/beatoraja/external/ScreenShotExporter.java
BEATORAJA_CLEAR = ["NO_PLAY", "FAILED", "ASSIST_EASY", "LIGHT_ASSIST_EASY", "EASY",
                   "NORMAL", "HARD", "EXHARD", "FC", "PERFECT", "MAX"]


def lr2_clear_to_beatoraja(v: int) -> int:
    """Map an LR2 lamp id onto the beatoraja enum the rest of the project uses.

    The two enums are the same gauge ladder but beatoraja inserts ASSIST_EASY /
    LIGHT_ASSIST_EASY at 2 and 3, so the ladder is a clean +2 shift:
        0,1 unchanged; 2..8 -> +2; 9 -> 2 (ASSIST), 10 -> 3 (L_ASSIST).
    This exists so a cross-client comparison is EXPLICIT. It does NOT make the two
    clients' lamps equivalent: the real archive only contains LR2 0..5 and the
    measured acc discrepancy below means they should be kept separate anyway.
    """
    v = int(v)
    if v in (0, 1):
        return v
    if 2 <= v <= 8:
        return v + 2
    if v == 9:
        return 2
    if v == 10:
        return 3
    raise ValueError(f"unknown LR2 clear id {v}")


def _manifest_bridge() -> dict[str, str]:
    """md5 -> sha256, from the parsed corpus manifest (PHASE3_AUDIT §4)."""
    out: dict[str, str] = {}
    with open(CORPUS / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["md5"]] = r["sha256"]
    return out


def compute_acc(perfect, great, notes):
    """LR2 EX score = perfect*2 + great, over a maximum of notes*2.

    Validated against the archive's own integer `rate` column: max |acc - rate| = 0.999
    (rate is the truncated integer) and corr = 0.9999. Extracted as a function so the
    label definition is unit-testable without the sqlite fixture.
    """
    notes = np.asarray(notes, dtype=float)
    ex = np.asarray(perfect, dtype=float) * 2.0 + np.asarray(great, dtype=float)
    return np.where(notes > 0, ex * 100.0 / (2.0 * notes), np.nan)


def find_db(path: str | Path) -> Path:
    """Accept either the .db itself or a folder holding exactly one .db."""
    p = Path(path)
    if p.is_file():
        return p
    dbs = sorted(p.glob("*.db"))
    if len(dbs) != 1:
        raise FileNotFoundError(f"expected exactly one .db under {p}, found {len(dbs)}")
    return dbs[0]


def load_rows(path: str | Path, bridge: dict[str, str] | None = None) -> pd.DataFrame:
    """Normalised per-chart frame: one row per LR2 score entry, corpus-joined."""
    db = find_db(path)
    bridge = bridge if bridge is not None else _manifest_bridge()
    con = sqlite3.connect(db)
    sc = pd.read_sql_query("SELECT * FROM score", con)
    try:
        pl = pd.read_sql_query("SELECT * FROM player", con)
        player = str(pl["id"].iloc[0]) if len(pl) else db.stem
    except Exception:  # noqa: BLE001 - player table is optional
        player = db.stem
    con.close()

    sc["md5"] = sc["hash"].str.lower()
    sc["sha256"] = sc["md5"].map(bridge)
    notes = sc["totalnotes"].astype(float)
    sc["notes"] = notes
    sc["acc"] = compute_acc(sc["perfect"], sc["great"], notes)
    sc["lamp"] = sc["clear"].astype(int)
    # explicit cross-client lamp, for comparison only - do NOT pool the two into one
    # feature column (see the measured gap in the module docstring)
    sc["bj_lamp"] = sc["lamp"].map(lr2_clear_to_beatoraja)
    sc["bp"] = sc["minbp"].astype(int)
    sc["player"] = player
    return sc[["player", "md5", "sha256", "notes", "acc", "lamp", "bj_lamp", "bp",
               "playcount", "clearcount", "failcount"]]


def audit(path: str | Path) -> dict:
    """Ingest-style audit: what is here, what joins, what the labels look like."""
    rows = load_rows(path)
    joined = rows["sha256"].notna()
    db = find_db(path)
    out = {
        "client": "lr2",
        "db": str(db.relative_to(ROOT)) if str(db).startswith(str(ROOT)) else str(db),
        "db_bytes": db.stat().st_size,
        "charts": int(len(rows)),
        "joined_to_corpus": int(joined.sum()),
        "join_rate": round(float(joined.mean()), 4),
        "duplicate_md5": int(rows["md5"].duplicated().sum()),
        "has_timestamps": False,
        "has_first_play_rows": False,
        "clear_levels": {LR2_CLEAR.get(int(k), str(k)): int(v)
                         for k, v in rows["lamp"].value_counts().sort_index().items()},
        "acc_median": round(float(rows["acc"].median()), 2),
        "playcount_median": float(rows["playcount"].median()),
        "playcount_max": int(rows["playcount"].max()),
        "issues": [],
        "fatal": [],
    }
    if out["join_rate"] < 0.5:
        out["issues"].append(f"only {out['join_rate']:.0%} of charts join the corpus - "
                             f"check that the BMS library matches this archive")
    if not set(rows["lamp"].unique()) - set(range(6)):
        out["issues"].append("no FC/PERFECT (clear>=6) rows: LR2 lamp levels are NOT "
                             "directly comparable to beatoraja's on a mixed roster")
    out["issues"].append("LR2 has no timestamps and no per-play rows -> usable as "
                         "player state only, never as first-play targets")
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="LR2 archive reader / audit")
    ap.add_argument("path", help="the .db file or the folder holding it")
    ap.add_argument("--dump", action="store_true", help="also print the normalised head")
    args = ap.parse_args()
    info = audit(args.path)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if args.dump:
        print(load_rows(args.path).head(10).to_string())


if __name__ == "__main__":
    main()
