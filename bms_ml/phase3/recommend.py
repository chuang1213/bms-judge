"""Phase 3.6 deliverable: BMS first-play recommender with a LOCAL-SERVER architecture.

   python bms_ml/phase3/recommend.py --serve            # then drag scorelog.db onto the page
   python bms_ml/phase3/recommend.py --player vsoflan   # headless: CSV only

Architecture (2026-09-11, user direction)
-----------------------------------------
The frontend is an ENTRY ONLY: `recommend_ui.html` embeds zero data. The user drags a
beatoraja `scorelog.db` onto the page, the browser POSTs the file to this local server
(127.0.0.1), the server rebuilds the player's first-play frame FROM THE UPLOADED DB
(exact replica of data.load_firstplays' per-player logic), scores every unplayed fence
chart with the protocol model, and returns JSON for the page to render. Nothing is
written back into the page, so the UI can never go stale the way an embedded-payload
report does.

The uploaded db is identified by md5 against the roster; a match scores as that player,
no match scores as a NEW player - which is precisely the zero-shot deployment path the
transfer experiments validated (features need no roster membership, only this archive).

Model training is expensive (3 point heads + 3 quantile heads + a pass classifier) and
player-independent, so it happens ONCE per server lifetime and is cached; each upload
then costs only feature construction (seconds).

What the score means
--------------------
NOT "practising this will make you stronger": the project has no longitudinal /
intervention data (PHASE3_3_READINESS §6). It is "what the model expects the FIRST play
to look like" (acc 0-100 / lamp 1-9 / BP misses), plus conformalised 80% acc bands and a
calibrated pass probability. Grouping thresholds follow the lamp head's ACTUAL spread
(measured: predictions compress toward the middle, so a "predicted FC" group never fires):

  暂缓区  pred_lamp_raw <  3.5   expected fail
  挑战区  3.5 .. 6.0             plausible pass, needs a real attempt
  冲刺区  >= 6.0                 expected clean pass, near the model's ceiling

How candidate features are built (the part worth reading)
---------------------------------------------------------
Candidates are appended to the player's chronological frame as FUTURE rows with NaN
targets, then the *unmodified* protocol machinery runs on the extended frame:

  build_history_features -> the history features, because every cumulative window simply
                            sees the real plays and stops;
  build_table            -> the acc/lamp/bp response blocks AND h_dev_*, because the OLS
                            window is `y_valid & x_valid`, so NaN-target candidates never
                            enter each other's windows.

Zero re-implementation of the causal logic, so deployment features cannot drift from
training features. The one exception is h_knn_acc, which build_history_features would
pollute (a candidate's k-NN window may contain another candidate with NaN acc), so it is
recomputed here: the KNN_K nearest of the player's REAL plays in the same standardised
stat space, same scaler, same k. NOTE (bug fixed 2026-09-11): the player's own rows MUST
carry log1p_bp - the earlier build never added it, so the whole BP response block was
silently all-NaN (imputed downstream) in the recommender.

Scoring time is "the player's last recorded session" (newest scorelog row), not
wall-clock now: a wall-clock gap of months would push h_days_since_active far beyond the
training distribution - the exact few-shot trap PHASE3_4_TRANSFER documented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from chart_repr import BEST_FEATURES, HISTORY_FEATURES, MSD_AXES, OBJECTIVE_STAT_COLS
from common import HGBModel, HGB_KW, load_samples, mae
from data import (HISTORY_USES_OFFTABLE, PLAYERS, ROOT, build_history_features,
                  load_firstplays, load_manifest, load_tables)
from response_decay import MIN_HISTORY
from response_features import axis_sd, build_table
from uncertainty_eval import QHead

PHASE3 = ROOT / "bms_ml" / "output" / "phase3"
DS = PHASE3 / "dataset"
OUT = PHASE3 / "recommend"
UI_PATH = Path(__file__).resolve().parent / "recommend_ui.html"
FEATS = BEST_FEATURES                     # the fused best configuration (5.887)
HALF_LIFE = 180.0                         # must match history_response.py's default
TABLE_SYM = {"satellite": "sl", "stella": "st", "insane": "発狂", "normal": "☆",
             "overjoy": "★★"}
# levels are only comparable WITHIN a table (satellite 12 vs insane 12 are different
# animals), so any cross-table level sort must go through this difficulty order first.
# Priority as loaded by data.load_tables, ascending difficulty.
TABLE_ORDER = {"normal": 0, "satellite": 1, "stella": 2, "insane": 3, "overjoy": 4}
KNN_K = 20                                # data.build_history_features default
LAMP_NAME = {1: "FAILED", 2: "ASSIST EZ", 3: "LIGHT ASSIST EZ", 4: "EASY", 5: "CLEAR",
             6: "HARD", 7: "EX HARD", 8: "FULL COMBO", 9: "PERFECT"}
UPLOAD_MAX = 200 * 1024 * 1024            # a scorelog.db is a few MB; guard anyway


# ---------------------------------------------------------------- protocol context
_CTX: dict | None = None
_CTX_LOCK = threading.Lock()


def protocol_context() -> dict:
    """Everything player-independent the scorer needs: manifest (+v2 axes +MSD skillsets),
    the difficulty tables, the chart-stat scaler fitted on the fence frame exactly as
    data.py main() fits it, and the response blocks' global axis sd. Cached."""
    global _CTX
    if _CTX is not None:
        return _CTX
    with _CTX_LOCK:
        if _CTX is not None:
            return _CTX
        man = load_manifest()
        # the four v2 response axes live in a side table keyed on sha256, not in the
        # manifest - without them build_table cannot even index the axis columns; the MSD
        # skillsets come from dataset/msd.parquet the same way (charts the WASM never saw
        # keep NaN and their MSD-block features impute downstream)
        from chart_repr import RESPONSE_AXES
        v2 = pd.read_parquet(DS / "chart_stats_v2.parquet")
        v2cols = [c for c in set(RESPONSE_AXES.values()) if c in v2.columns]
        man = man.merge(v2[["sha256"] + v2cols], on="sha256", how="left")
        man = man.merge(pd.read_parquet(DS / "msd.parquet"), on="sha256", how="left")
        tab = load_tables()

        fp = (load_firstplays()
              .merge(man, on="sha256", how="left")
              .merge(tab, on="sha256", how="left"))
        assert not fp.duplicated(["player", "sha256"]).any(), \
            "duplicate (player, sha256): the fence is not 1:1 on sha256"
        fp["acc"] = np.where(fp["notes"] > 0, fp["ex"] * 50.0 / fp["notes"], np.nan)
        fp = fp[fp["bp"] <= fp["notes"] + 5].copy()
        fp = fp[fp["notes"].notna() & (fp["notes"] > 0)].copy()
        fp["is_target"] = fp["table"].notna()
        if not HISTORY_USES_OFFTABLE:
            fp = fp[fp["is_target"]].copy()
        fp = fp.reset_index(drop=True)
        fp["bp_ratio"] = fp["bp"] / fp["notes"]
        stat_cols = [c for c in fp.columns if c.startswith("c_")]
        assert set(stat_cols) == set(OBJECTIVE_STAT_COLS), "chart stat drift vs the registry"
        scaler = StandardScaler().fit(fp[stat_cols].values)

        # the response blocks' slope scaling must match history_response.py exactly: it
        # computes its sd on the FULL first-play frame (before any filtering), so do the same
        fp_full = (pd.read_parquet(DS / "firstplays.parquet")
                   .merge(v2[["sha256"] + v2cols], on="sha256", how="left")
                   .merge(pd.read_parquet(DS / "msd.parquet"), on="sha256", how="left")
                   .sort_values(["player", "time"], kind="stable").reset_index(drop=True))
        _CTX = {"man": man, "tab": tab, "stat_cols": stat_cols, "scaler": scaler,
                "sd_struct": axis_sd(fp_full), "sd_msd": axis_sd(fp_full, MSD_AXES)}
        return _CTX


# ---------------------------------------------------------------------- uploaded db
def firstplays_from_db(db_path: Path, player: str) -> pd.DataFrame:
    """First-play events from ONE scorelog.db - an exact replica of the per-player logic
    in data.load_firstplays (real-time mode): earliest row per sha256 is the first play,
    course rows (mode>=100 / short sha) dropped, NO_PLAY(0) and score==0 rows are
    aborted/practice plays. The uploaded archive is always treated as beatoraja/real
    time; the P3_TIME_MODE ablation flags deliberately do NOT apply here."""
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT rowid, sha256, mode, clear, score, minbp, date FROM scorelog", con)
    con.close()
    df = df[(df["sha256"].str.len() == 64) & (df["mode"].astype(int) < 100)]
    df = df.sort_values(["date", "rowid"]).drop_duplicates("sha256", keep="first")
    df = df[df["clear"] != 0]
    df = df[df["score"] > 0]
    return pd.DataFrame({
        "player": player,
        "sha256": df["sha256"].values,
        "time": pd.to_datetime(df["date"].values, unit="s"),
        "lamp": df["clear"].values,       # first-play lamp (exact, see data.py docstring)
        "ex": df["score"].values,
        "bp": df["minbp"].values,
    })


def logrows_from_db(db_path: Path, player: str) -> pd.DataFrame:
    """The FULL scorelog row stream (activity features need every row, not just firsts)."""
    con = sqlite3.connect(db_path)
    d = pd.read_sql_query("SELECT date FROM scorelog", con)
    con.close()
    return pd.DataFrame({"player": player,
                         "time": pd.to_datetime(d["date"], unit="s")})


# --------------------------------------------------------------------- model cache
_MDL: dict | None = None
_MDL_LOCK = threading.Lock()


def train_models() -> dict:
    """Train every head once per server lifetime and cache. The fused features come from
    the dataset artifact: samples.parquet carries the chart stats and the hand-crafted
    history, history_response.parquet (built at the same 180d half-life) carries every
    response/dev block, MSD family included."""
    global _MDL
    if _MDL is not None:
        return _MDL
    with _MDL_LOCK:
        if _MDL is not None:
            return _MDL
        print("training the protocol model on the train band ...", flush=True)
        hr = pd.read_parquet(DS / "history_response.parquet")
        static = OBJECTIVE_STAT_COLS + HISTORY_FEATURES
        need = [c for c in FEATS if c not in static]
        df = load_samples().merge(hr[["player", "sha256"] + [c for c in need
                                                             if c in hr.columns]],
                                  on=["player", "sha256"], how="left")
        missing = [c for c in FEATS if c not in df.columns]
        if missing:
            raise SystemExit(f"history_response.parquet lacks {len(missing)} columns "
                             f"(e.g. {missing[:3]}); run history_response.py")
        tr = df[df["phase"] == "train"]
        mdl: dict = {"df": df,
                     "models": {"acc": HGBModel(tr, FEATS, tr["acc"].values),
                                "lamp": HGBModel(tr, FEATS, tr["lamp"].values.astype(float)),
                                "bp": HGBModel(tr, FEATS, np.log1p(tr["bp"].values))},
                     "bp_cap": float(np.log1p(df["bp"].max()))}

        # ---- uncertainty heads, fitted ONCE (feature-conditioned bands + pass prob) --
        # Quantile-HGB heads at q10/q50/q90 per target, conformalised (CQR): the
        # correction Q is the 80th percentile of the calibration nonconformity, fitted on
        # a per-player time split of the train band, never on the test band or the
        # candidates. The variance diagnostic showed per-player error is mostly that
        # player's own behavioural variance (corr +0.882), and the measured band widths
        # track it (+0.846: reiaki ~10pp wide, yangtao ~50pp).
        fit_idx, cal_idx = [], []
        for _p, g in tr.groupby("player"):
            g = g.sort_values("time")
            k = max(1, int(len(g) * 0.8))
            fit_idx.extend(g.index[:k])
            cal_idx.extend(g.index[k:])
        fit, cal = tr.loc[fit_idx], tr.loc[cal_idx]

        def conformal(heads: dict, y_cal: np.ndarray, frame: pd.DataFrame) -> float:
            s = np.maximum(heads[0.1].predict(frame) - y_cal,
                           y_cal - heads[0.9].predict(frame))
            qq = min(1.0, np.ceil((len(s) + 1) * 0.8) / len(s))
            return float(np.quantile(s, qq))

        acc_q = {q: QHead(fit, FEATS, fit["acc"].values, q) for q in (0.1, 0.9)}
        mdl["acc_q"], mdl["qa"] = acc_q, conformal(acc_q, cal["acc"].values, cal)
        bp_q = {q: QHead(fit, FEATS, np.log1p(fit["bp"].values), q) for q in (0.1, 0.9)}
        mdl["bp_q"], mdl["qb"] = bp_q, conformal(bp_q, np.log1p(cal["bp"].values), cal)

        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(strategy="median").fit(fit[FEATS])
        sc = StandardScaler().fit(imp.transform(fit[FEATS]))
        clf = HistGradientBoostingClassifier(random_state=0, **HGB_KW)
        clf.fit(sc.transform(imp.transform(fit[FEATS])),
                (fit["lamp"].values >= 4).astype(int))
        mdl["imp"], mdl["sc"], mdl["clf"] = imp, sc, clf
        _MDL = mdl
        return _MDL


def expected_mae(mdl: dict, player: str) -> float | None:
    """Out-of-sample accuracy for THIS player: their test rows were never trained on."""
    te = mdl["df"][(mdl["df"]["player"] == player) & (mdl["df"]["phase"] == "test")]
    return float(mae(te["acc"], mdl["models"]["acc"].predict(te))) if len(te) else None


# ------------------------------------------------------------------------ scoring
def score_archive(db_path: Path, label: str, mdl: dict, ctx: dict,
                  ) -> tuple[list[dict], dict, float | None]:
    """Score every unplayed fence chart for the archive at db_path. `label` is the
    matched roster name or a synthetic new-player id (features need no roster
    membership; only pmae needs one)."""
    man, tab = ctx["man"], ctx["tab"]
    stat_cols, scaler = ctx["stat_cols"], ctx["scaler"]

    own = (firstplays_from_db(db_path, label)
           .merge(man, on="sha256", how="left")
           .merge(tab, on="sha256", how="left"))
    if own["notes"].isna().all():
        raise ValueError("uploaded db has no charts in the corpus - wrong file?")
    own["acc"] = np.where(own["notes"] > 0, own["ex"] * 50.0 / own["notes"], np.nan)
    own = own[own["bp"] <= own["notes"] + 5]
    own = own[own["notes"].notna() & (own["notes"] > 0)]
    own = own[own["table"].notna()]                      # the fence is the target space
    own = own.sort_values(["player", "time"], kind="stable").reset_index(drop=True)
    if not len(own):
        raise ValueError("uploaded db has no usable fence first plays")
    # BUG FIX 2026-09-11: the BP response block reads the `log1p_bp` column as its
    # target; the earlier build never added it, so every b_* feature was NaN.
    own["log1p_bp"] = np.log1p(own["bp"].values.astype(np.float64))

    log_counts = logrows_from_db(db_path, label)
    top_log = log_counts["time"].max()

    cand = man.merge(tab, on="sha256", how="inner")
    cand = cand[cand["notes"].notna() & (cand["notes"] > 0)]
    cand = cand[~cand["sha256"].isin(set(own["sha256"]))].reset_index(drop=True)

    # append as future rows; NaN targets keep them out of every causal window
    cext = cand.reindex(columns=own.columns)
    cext["player"] = label
    cext["time"] = top_log
    for c in ("acc", "lamp", "ex", "bp", "bp_ratio", "log1p_bp"):
        cext[c] = np.nan
    ext = pd.concat([own, cext], ignore_index=True)

    Z_own = np.nan_to_num(scaler.transform(own[stat_cols].values)).astype(np.float64)
    Z_cand = np.nan_to_num(scaler.transform(cand[stat_cols].values)).astype(np.float64)
    H = build_history_features(ext, log_counts, np.vstack([Z_own, Z_cand]))
    hist = H.iloc[len(own):].reset_index(drop=True)

    # h_knn_acc must be recomputed: for every candidate after the first, the built-in
    # window would contain another candidate whose acc is NaN, killing the mean
    acc_own = own["acc"].values.astype(np.float64)
    knn = np.empty(len(cand))
    for i in range(len(cand)):
        d = np.sqrt(((Z_own - Z_cand[i]) ** 2).sum(1))
        kk = min(KNN_K, len(d))
        knn[i] = float(np.mean(acc_own[np.argpartition(d, kk - 1)[:kk]]))
    hist["h_knn_acc"] = knn

    blocks = [
        build_table(ext, ctx["sd_struct"], min_n=MIN_HISTORY, target="acc", prefix="h_",
                    clip=(0.0, 100.0), half_life=HALF_LIFE),
        build_table(ext, ctx["sd_struct"], min_n=MIN_HISTORY, target="lamp", prefix="l_",
                    clip=(1.0, 9.0), half_life=HALF_LIFE),
        build_table(ext, ctx["sd_struct"], min_n=MIN_HISTORY, target="log1p_bp",
                    prefix="b_", clip=(0.0, 12.0), half_life=HALF_LIFE),
        # the MinaCalc axis family (same axes/clips as history_response.py)
        build_table(ext, ctx["sd_msd"], min_n=MIN_HISTORY, target="acc", prefix="m_",
                    clip=(0.0, 100.0), half_life=HALF_LIFE, axes=MSD_AXES),
        build_table(ext, ctx["sd_msd"], min_n=MIN_HISTORY, target="lamp", prefix="ml_",
                    clip=(1.0, 9.0), half_life=HALF_LIFE, axes=MSD_AXES),
        build_table(ext, ctx["sd_msd"], min_n=MIN_HISTORY, target="log1p_bp", prefix="mb_",
                    clip=(0.0, 12.0), half_life=HALF_LIFE, axes=MSD_AXES),
    ]
    feats = pd.concat([hist] + [b.iloc[len(own):].reset_index(drop=True) for b in blocks],
                      axis=1)
    out = pd.concat([cand, feats], axis=1)
    out["has_msd"] = out["msd_overall"].notna()

    # ---- predictions (point + uncertainty + group) --------------------------------
    out["pred_acc"] = mdl["models"]["acc"].predict(out)
    out["pred_lamp_raw"] = mdl["models"]["lamp"].predict(out)
    out["pred_lamp"] = np.clip(np.round(out["pred_lamp_raw"]), 1, 9).astype(int)
    out["pred_lamp_name"] = out["pred_lamp"].map(LAMP_NAME)
    out["pred_bp"] = np.expm1(np.clip(mdl["models"]["bp"].predict(out), 0, mdl["bp_cap"]))
    out["pred_acc_lo"] = mdl["acc_q"][0.1].predict(out) - mdl["qa"]
    out["pred_acc_hi"] = mdl["acc_q"][0.9].predict(out) + mdl["qa"]
    blo = mdl["bp_q"][0.1].predict(out) - mdl["qb"]
    out["pred_bp_lo"] = np.expm1(np.clip(blo, 0, 12))
    out["p_pass"] = mdl["clf"].predict_proba(
        mdl["sc"].transform(mdl["imp"].transform(out[FEATS])))[:, 1]
    out["group"] = np.where(out["pred_lamp_raw"] >= 6.0, "冲刺区",
                            np.where(out["pred_lamp_raw"] >= 3.5, "挑战区", "暂缓区"))

    rows = [{"t": r.table, "l": int(r.level), "ti": str(r.title), "ar": str(r.artist),
             "g": r.group, "pl": int(r.pred_lamp), "pn": str(r.pred_lamp_name),
             "pp": round(float(r.p_pass), 4), "pa": round(float(r.pred_acc), 1),
             "lo": int(round(float(r.pred_acc_lo))), "hi": int(round(float(r.pred_acc_hi))),
             "pb": int(round(float(r.pred_bp))), "m": bool(r.has_msd)}
            for r in out.itertuples(index=False)]
    meta = {"n_archive": int(len(own)), "n_candidates": int(len(out)),
            "n_no_msd": int((~out["has_msd"]).sum()), "as_of": str(top_log)[:19],
            "acc_mean": float(np.nanmean(own["acc"].values)),
            "acc_last20": float(np.nanmean(own["acc"].values[-20:]))}
    return rows, meta, expected_mae(mdl, label)


# ------------------------------------------------------------------- player lookup
_ROSTER_MD5: dict[str, str] | None = None


def roster_md5() -> dict[str, str]:
    """md5 -> player for every roster scorelog.db, computed once (drag-in identification)."""
    global _ROSTER_MD5
    if _ROSTER_MD5 is None:
        out = {}
        for player, rel in PLAYERS.items():
            p = ROOT / rel / "scorelog.db"
            if p.exists():
                h = hashlib.md5()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                out[h.hexdigest()] = player
        _ROSTER_MD5 = out
    return _ROSTER_MD5


# -------------------------------------------------------------------------- server
def handle_upload(body: bytes) -> dict:
    if len(body) > UPLOAD_MAX:
        return {"ok": False, "error": f"file too large ({len(body)} bytes)"}
    UP = OUT / "_upload.db"
    OUT.mkdir(parents=True, exist_ok=True)
    UP.write_bytes(body)
    digest = hashlib.md5(body).hexdigest()
    matched = roster_md5().get(digest)
    label = matched or "new_player"
    try:
        mdl, ctx = train_models(), protocol_context()
        rows, meta, pmae = score_archive(UP, label, mdl, ctx)
    except Exception as exc:                      # surface as a 400, not a stack trace
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "label": label, "matched": matched is not None,
            "pmae": pmae, "meta": meta, "rows": rows}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8",
                       UI_PATH.read_bytes())
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self) -> None:
        if self.path != "/score":
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        result = handle_upload(body)
        resp = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self._send(200, "application/json; charset=utf-8", resp)

    def log_message(self, fmt: str, *args) -> None:
        pass                    # quiet: the handler prints its own one-line notes


def serve(port: int, open_browser: bool = True) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"serving on {url}  (Ctrl+C to stop)\n"
          f"  drag a beatoraja scorelog.db onto the page that just opened", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


# ----------------------------------------------------------------------- headless
def run_cli(player: str, top: int) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    mdl, ctx = train_models(), protocol_context()
    db_path = ROOT / PLAYERS[player] / "scorelog.db"
    print(f"scoring for {player} ({db_path}) ...")
    rows, meta, pmae = score_archive(db_path, player, mdl, ctx)
    rows.sort(key=lambda r: ({"挑战区": 0, "冲刺区": 1, "暂缓区": 2}[r["g"]],
                             -r["pp"]))
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / f"{player}.csv"
    head = ["group", "table", "level", "title", "artist", "pred_lamp", "p_pass",
            "pred_acc", "acc_lo", "acc_hi", "pred_bp", "has_msd"]
    lines = [",".join(head)]
    for r in rows:
        lines.append(",".join([r["g"], r["t"], str(r["l"]),
                               '"' + str(r["ti"]).replace('"', '""') + '"',
                               '"' + str(r["ar"]).replace('"', '""') + '"',
                               str(r["pl"]), str(r["pp"]), str(r["pa"]),
                               str(r["lo"]), str(r["hi"]), str(r["pb"]), str(r["m"])]))
    csv_path.write_text("\ufeff" + "\n".join(lines), encoding="utf-8")
    print(f"archive {meta['n_archive']} | candidates {meta['n_candidates']} "
          f"| as of {meta['as_of']} | your out-of-sample acc MAE "
          f"{pmae if pmae is not None else float('nan'):.2f}")
    for g in ("挑战区", "冲刺区", "暂缓区"):
        sub = [r for r in rows if r["g"] == g]
        print(f"\n== {g}  ({len(sub)} charts) ==")
        for r in sub[:top]:
            print(f"  {TABLE_SYM.get(r['t'], r['t']):>3}{r['l']:<4}"
                  f" lamp~{r['pl']}({r['pn']:<10}) pass~{r['pp']:5.0%}"
                  f" acc~{r['pa']:5.1f} bp~{r['pb']:5}  {r['ti'][:44]}")
    print(f"\nsaved -> {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="BMS first-play recommender")
    ap.add_argument("--serve", action="store_true",
                    help="start the local server (drag scorelog.db onto the page)")
    ap.add_argument("--port", type=int, default=8932)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--player", choices=sorted(PLAYERS),
                    help="headless mode: score this roster player, write CSV")
    ap.add_argument("--top", type=int, default=25,
                    help="rows printed per group in headless mode")
    args = ap.parse_args()
    if args.serve:
        serve(args.port, open_browser=not args.no_browser)
    elif args.player:
        run_cli(args.player, args.top)
    else:
        ap.error("choose --serve or --player")


if __name__ == "__main__":
    main()
