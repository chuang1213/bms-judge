"""Phase 3.3: unified player data ingestion entry.

New player data should enter the pipeline ONLY through this script:

    python ingest_player.py add <name> <dir> [--client beatoraja] [--time real|synthetic]
    python ingest_player.py list

It validates the raw save folder (beatoraja score.db/scorelog.db/scoredatalog.db) or an
LR2 archive (a single .db — see lr2_reader.py, implemented 2026-09-11 once real LR2 data
arrived), writes a provenance entry into players.json (include=false until reviewed),
and prints an audit summary.

`--time synthetic` is for clients without reliable timestamps (e.g. LR2): the loader
will replace dates with play-order ordinal days. Ordering is preserved, absolute time
semantics are lost, and cross-player time features degrade — this is recorded in
provenance, never faked as real time.

⚠️ LR2 is NOT merely "a client without timestamps". Its score table holds ONE row per
chart (best score + counters), so it has neither first plays nor any ordering at all:
the synthetic ordinal days are then the DB's arbitrary row order, and the archive can
serve as PLAYER STATE ONLY, never as first-play targets. Left at include=false until
the target protocol for LR2-only players is decided.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROSTER = Path(__file__).resolve().parent / "players.json"


def load_roster() -> dict:
    return json.load(open(ROSTER, encoding="utf-8"))


def save_roster(roster: dict) -> None:
    with open(ROSTER, "w", encoding="utf-8", newline="\n") as f:
        json.dump(roster, f, ensure_ascii=False, indent=2)


def audit_dir(rel_dir: str, client: str) -> dict:
    d = ROOT / rel_dir
    info: dict = {"dir": rel_dir, "client": client, "files": {},
                  "issues": [], "fatal": []}
    if not d.exists():
        info["fatal"].append("directory does not exist")
        return info
    if client == "beatoraja":
        for f in ["score.db", "scorelog.db", "scoredatalog.db"]:
            p = d / f
            info["files"][f] = p.exists()
        if not info["files"]["scorelog.db"]:
            info["fatal"].append("scorelog.db is the ONLY required file (first-play "
                                 "events + timestamps); it is missing")
            return info
        if not info["files"]["score.db"]:
            info["issues"].append("score.db missing (ok for the current pipeline; "
                                  "loses playcount/ghost aggregates)")
        if not info["files"]["scoredatalog.db"]:
            info["issues"].append("scoredatalog.db missing (ok for the current "
                                  "pipeline; loses per-play judgment detail)")
        con = sqlite3.connect(d / "scorelog.db")
        try:
            n = con.execute("SELECT COUNT(*) FROM scorelog").fetchone()[0]
            dates = con.execute("SELECT MIN(date), MAX(date) FROM scorelog").fetchone()
            bad_len = con.execute(
                "SELECT COUNT(*) FROM scorelog WHERE length(sha256) != 64").fetchone()[0]
            modes = con.execute(
                "SELECT mode, COUNT(*) FROM scorelog GROUP BY mode ORDER BY 2 DESC LIMIT 5").fetchall()
        finally:
            con.close()
        info["scorelog_rows"] = n
        info["date_range"] = [datetime.fromtimestamp(dates[0]).isoformat()
                              if dates[0] else None,
                              datetime.fromtimestamp(dates[1]).isoformat()
                              if dates[1] else None]
        info["non_sha256_rows"] = bad_len          # course rows etc., filtered downstream
        info["top_modes"] = modes
        if dates[0] == 0 and dates[1] == 0:
            info["issues"].append("no usable timestamps -> use --time synthetic")
    elif client == "lr2":
        # 2026-09-11: real parser (was a stub). `d` may be the .db itself or a folder.
        from lr2_reader import audit as lr2_audit
        sub = lr2_audit(d)
        info.update({k: v for k, v in sub.items()
                     if k not in ("issues", "fatal", "client")})
        info["issues"] += sub.get("issues", [])
        info["fatal"] += sub.get("fatal", [])
    else:
        info["fatal"].append(f"unknown client '{client}' (supported: beatoraja, lr2)")
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["add", "list"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--client", default="beatoraja", choices=["beatoraja", "lr2"])
    ap.add_argument("--time", choices=["real", "synthetic"], default="real")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    if args.client == "lr2" and args.time != "synthetic":
        print("[note] LR2 has no timestamps (and no play order); forcing --time "
              "synthetic. The ordinal days will follow the DB's row order, which is "
              "arbitrary - not a play sequence.")
        args.time = "synthetic"

    roster = load_roster()
    if args.cmd == "list":
        for name, e in roster["players"].items():
            print(f"{name:10} include={e['include']!s:5} time={e['time']:9} {e['dir']}  | {e.get('note','')}")
        return

    if not args.name or not args.dir:
        sys.exit("usage: ingest_player.py add <name> <dir> [--client ...] [--time ...]")
    audit = audit_dir(args.dir, args.client)
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
    if audit["fatal"]:
        sys.exit("validation failed; player NOT added: " + "; ".join(audit["fatal"]))
    roster["players"][args.name] = {
        "dir": args.dir, "client": args.client, "time": args.time,
        "include": False, "note": args.note,
        "added": datetime.now().isoformat(timespec="seconds"),
        "audit": {k: v for k, v in audit.items() if k != "files"},
    }
    save_roster(roster)
    print(f"added '{args.name}' with include=false; review the audit above, then "
          f"set include=true in players.json (or via edit) before rebuilding the dataset.")


if __name__ == "__main__":
    main()
