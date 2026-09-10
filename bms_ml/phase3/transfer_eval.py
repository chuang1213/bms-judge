"""Phase 3.4 (transfer): cross-player transfer / few-shot adaptation.

Core question: does data from other players help a player who never participated
in training?

Models (protocol per PROTOCOL.md; scope = samples.parquet's current scope):
  M0  chart-only       — HGB trained on ALL other players, objective chart features
                         only. Population's notion of chart difficulty; no player info.
  M1  D-local          — prediction formed ONLY from D's own prefix: kNN-20 mean acc
                         of D's most similar played charts in objective stat space
                         (fallback: prefix mean). No other player's data anywhere.
  M2  cross-player     — HGB trained on ALL other players (chart + history features,
                         causal), conditioned on D via history features computed from
                         D's prefix ONLY. D appears nowhere in training.

Few-shot: for k in {0,1,5,10,20,50,100,200}, D's first k plays form the prefix,
targets are D's plays strictly after the prefix; all D history features derive from
the prefix (not D's full archive). M0 is constant in k; the M2-vs-M1 gap vs k is
the sample-efficiency test.

Exp-1 (strict protocol band): M0/M1/M2 with FULL causal history features on each
D's protocol test band (time > q75), matching PROTOCOL.md.

Leakage guards (asserted):
  - training frames contain zero rows of D;
  - every prefix event precedes its target in time;
  - target sha256 never in its own prefix;
  - the kNN stat-space scaler is fitted on training players' chart stats only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

from chart_repr import (HISTORY_FEATURES, HISTORY_FEW_FEATURES, HISTORY_RESPONSE_COLS,
                        OBJECTIVE_STAT_COLS, RESPONSE_AXES)
from common import HGBModel, add_region, hgb_fit_predict, load_firstplays, mae
from response_features import axis_sd, build_table, prefix_response

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"
KS = [0, 1, 5, 10, 20, 50, 100, 200]
KNN_K = 20
STAT, HIST = OBJECTIVE_STAT_COLS, HISTORY_FEATURES
HIST_FEW = HISTORY_FEW_FEATURES

# v2 ablation (Phase 3.5 task 6): add the threshold-free distribution stats to the
# chart side of M0/M2. A/B without editing: P3_USE_V2=1 python transfer_eval.py
# NB: the kNN stat space (scaler in prefix_hist_features) stays on the v1 27-dim —
# the training h_knn_acc comes from data.py's v1 space, so mixing v2 into the
# few-shot kNN would make the SAME feature incomparable between train and eval.
USE_V2 = os.environ.get("P3_USE_V2", "0") == "1"
# perm_space ablation (Phase 3.6, 2026-09-11): add the permutation-space hand-travel
# geometry to the chart side of M0/M2. Same switch pattern as USE_V2.
#   P3_USE_PERM=1 python transfer_eval.py
USE_PERM = os.environ.get("P3_USE_PERM", "0") == "1"
# personal response profile in the few-shot curve (Phase 3.6, 2026-09-11).
#   P3_USE_RESP=1 python transfer_eval.py --tag _resp
# The profile is windowed here (RESP_WIN) and the TRAINING rows are truncation-
# augmented, because a few-shot row only ever has its prefix while a protocol row has
# the whole archive - fitting "all available history" would use a different estimator
# on the two sides (the recency-feature trap from PHASE3_4_TRANSFER). See
# response_features.py. Only the few-shot curve changes: the exp-1 protocol band keeps
# its full-causal reference untouched so the two remain comparable.
USE_RESP = os.environ.get("P3_USE_RESP", "0") == "1"
RESP_WIN = int(os.environ.get("P3_RESP_WINDOW", "50"))
RESP_MIN = int(os.environ.get("P3_RESP_MIN", "5"))
# zero-slope prior worth this many events (see response_features.response_columns).
# Without it the k=5 slope is pure noise yet the truncation augmentation teaches the
# model to trust it: measured k=5 = 20.51 vs the 17.50 baseline (worse than k=1).
RESP_LAMBDA = float(os.environ.get("P3_RESP_LAMBDA", "20"))


def prefix_hist_features(prefix: pd.DataFrame, targets: pd.DataFrame,
                         scaler: StandardScaler) -> pd.DataFrame:
    """The 12 history features for each target, computed from `prefix` events only
    (all prefix events are strictly before every target by construction)."""
    n = len(targets)
    Zp = scaler.transform(prefix[STAT].values)
    Tp = prefix["time"].values.astype("datetime64[s]").astype(np.int64) // 10**9
    acc = prefix["acc"].values
    known = ~np.isnan(acc)
    Zk, ak, tk = Zp[known], acc[known], Tp[known]
    lamp = prefix["lamp"].values
    bp = prefix["bp"].values
    bpr = (prefix["bp"] / prefix["notes"].replace(0, np.nan)).values
    tt = targets["time"].values.astype("datetime64[s]").astype(np.int64) // 10**9
    Zt = scaler.transform(targets[STAT].values)

    d = np.sqrt(((Zt[:, None, :] - Zk[None, :, :]) ** 2).sum(-1)) if len(Zk) else \
        np.full((n, 0), np.nan)
    knn = np.full(n, np.nan)
    for j in range(n):
        if d.shape[1] == 0:
            knn[j] = ak.mean() if len(ak) else np.nan
            continue
        kk = min(KNN_K, d.shape[1])
        near = np.argpartition(d[j], kk - 1)[:kk]
        knn[j] = ak[near].mean()

    out = pd.DataFrame(index=targets.index)
    out["h_knn_acc"] = knn
    out["h_n_firstplays"] = float(len(prefix))
    known_t = ~np.isnan(targets["acc"].values)
    for col, vals in [("h_acc_mean", acc[known]), ("h_acc_last10", ak[-10:]),
                      ("h_bp_mean", bp), ("h_bp_ratio_mean", bpr)]:
        out[col] = float(np.nanmean(vals)) if len(vals) else np.nan
    out["h_acc_std"] = float(np.nanstd(acc)) if len(acc) > 1 else np.nan
    out["h_fail_rate"] = float((lamp == 1).mean())
    out["h_fc_rate"] = float((lamp >= 8).mean())
    out["h_days_since_active"] = (tt - Tp.max()) / 86400.0 if len(Tp) else np.nan
    out["h_plays_last30d"] = (((tt[:, None] - Tp[None, :]) / 86400.0) <= 30).sum(axis=1) \
        if len(Tp) else np.nan
    out["h_days_span"] = (tt - Tp.min()) / 86400.0 if len(Tp) else np.nan
    return out[[c for c in HIST]]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="",
                    help="output suffix; the default writes transfer_results.json "
                         "(the canonical protocol run). Use e.g. _v2_perm for ablations "
                         "so an ablation can never silently replace the baseline.")
    args = ap.parse_args()
    fp = add_region(load_firstplays().reset_index(drop=True))
    # v2 chart-side ablation: merge the distribution stats in as extra columns
    CHART = list(STAT)
    if USE_V2:
        from chart_repr import OBJECTIVE_V2_COLS
        fp = fp.merge(pd.read_parquet(DS / "chart_stats_v2.parquet"),
                      on="sha256", how="left")
        CHART = CHART + OBJECTIVE_V2_COLS
    if USE_PERM:
        from chart_repr import OBJECTIVE_PERM_COLS
        fp = fp.merge(pd.read_parquet(DS / "chart_perm_space.parquet"),
                      on="sha256", how="left")
        CHART = CHART + OBJECTIVE_PERM_COLS

    # ---- personal response profile (few-shot only; see USE_RESP above) ----
    RESP = []
    if USE_RESP:
        # four axes are v2 columns kept in a side table
        need = [c for c in set(RESPONSE_AXES.values()) if c not in fp.columns]
        if need:
            v2tab = pd.read_parquet(DS / "chart_stats_v2.parquet")
            fp = fp.merge(v2tab[["sha256"] + [c for c in need if c in v2tab.columns]],
                          on="sha256", how="left")
        fp = fp.sort_values(["player", "time"], kind="stable").reset_index(drop=True)
        # TRAINING columns: windowed estimator + deployment-style truncation
        # augmentation, so the model sees every possible fill level. This depends only
        # on each row's own past, so it is computed once for the whole roster.
        sd_axes = axis_sd(fp)
        tr_resp = build_table(fp, sd_axes, min_n=RESP_MIN, window=RESP_WIN,
                              rng=np.random.RandomState(0), shrink=RESP_LAMBDA)
        for c in HISTORY_RESPONSE_COLS:
            fp[c] = tr_resp[c].values
        RESP = list(HISTORY_RESPONSE_COLS)

    players = sorted(fp["player"].unique())
    results: dict = {"scope_rows": len(fp), "players": players, "ks": KS,
                     "use_v2": USE_V2, "use_perm": USE_PERM, "use_resp": USE_RESP,
                     "resp_window": RESP_WIN if USE_RESP else None, "lopo": {}}

    for D in players:
        others = fp[fp["player"] != D]
        mine = fp[fp["player"] == D].sort_values("time").reset_index(drop=True)
        assert (others["player"] != D).all()
        # strict scaler: training players' chart stats only (leakage guard);
        # always the v1 space — see the USE_V2 note above
        scaler = StandardScaler().fit(others[STAT].values)

        res = {"n_events": int(len(mine))}

        # ---------- Exp 1: protocol band, FULL causal features ----------
        te_band = mine[mine["phase"] == "test"]
        tr_others = others  # all rows are causal w.r.t. their own targets
        m0 = {k: hgb_fit_predict(tr_others, te_band, CHART, tr_others[k].values)
              for k in ("acc", "lamp")}
        m0["bp"] = np.expm1(hgb_fit_predict(tr_others, te_band, CHART,
                                         np.log1p(tr_others["bp"].values)))
        m2 = {k: hgb_fit_predict(tr_others, te_band, CHART + HIST, tr_others[k].values)
              for k in ("acc", "lamp")}
        m2["bp"] = np.expm1(hgb_fit_predict(tr_others, te_band, CHART + HIST,
                                         np.log1p(tr_others["bp"].values)))
        m1_acc = te_band["h_knn_acc"].values  # D-local: causal kNN within own archive
        m1_lamp = te_band["h_acc_mean"].values / 100 * 6  # crude D-local lamp proxy
        m1_bp = te_band["h_bp_mean"].values

        def metrics(acc_p, lamp_p, bp_p):
            lerr = np.abs(te_band["lamp"].values - lamp_p)
            return {"acc_mae": round(mae(te_band["acc"], acc_p), 3),
                    "lamp_mae": round(float(lerr.mean()), 3),
                    "lamp_qwk": round(float(cohen_kappa_score(
                        te_band["lamp"], np.clip(np.round(lamp_p), 1, 9).astype(int),
                        weights="quadratic", labels=list(range(1, 10)))), 3),
                    "bp_mae": round(mae(te_band["bp"], bp_p), 2)}

        res["exp1"] = {"n_eval": int(len(te_band)),
                       "M0": metrics(m0["acc"], m0["lamp"], m0["bp"]),
                       "M1": metrics(m1_acc, m1_lamp, m1_bp),
                       "M2": metrics(m2["acc"], m2["lamp"], m2["bp"])}

        # per-region M2-M1 on protocol band
        reg = {}
        for r, sub in te_band.groupby("region"):
            idx = sub.index
            reg[r] = {"n": int(len(sub)),
                      "M1": round(mae(sub["acc"], m1_acc[te_band.index.get_indexer(idx)]), 2),
                      "M2": round(mae(sub["acc"], m2["acc"][te_band.index.get_indexer(idx)]), 2)}
        res["exp1_regions"] = reg

        # ---------- Exp 2: few-shot prefix curve ----------
        # targets = the plays IMMEDIATELY following the prefix (window W), so the
        # prefix is temporally adjacent to the targets — matching real deployment
        # ("player has k plays so far, predict their next charts") and keeping the
        # history-recency feature distribution consistent with training.
        W = 150
        # The k-loop varies ONLY the evaluation frame: tr_others, the labels and the
        # feature lists are identical for every k. Refitting imputer+scaler+HGB once
        # per k did the same fit 8x; hoisting it is numerically identical (see
        # common.HGBModel) and is the bulk of this script's speedup.
        # NB: M2_bp deliberately uses CHART + HIST (all 12), not CHART + HIST_FEW as
        # acc/lamp do — preserved as-is for continuity of the reported curve.
        bp_cap = float(np.log1p(tr_others["bp"].max()))
        M2F = CHART + HIST_FEW + RESP
        M2B = CHART + HIST + RESP
        m0_acc = HGBModel(tr_others, CHART, tr_others["acc"].values)
        m0_lamp = HGBModel(tr_others, CHART, tr_others["lamp"].values.astype(float))
        m2_acc = HGBModel(tr_others, M2F, tr_others["acc"].values)
        m2_lamp = HGBModel(tr_others, M2F, tr_others["lamp"].values.astype(float))
        m0_bp = HGBModel(tr_others, CHART, np.log1p(tr_others["bp"].values))
        m2_bp = HGBModel(tr_others, M2B, np.log1p(tr_others["bp"].values))

        curve = []
        for k in KS:
            if k >= len(mine) - 1:
                continue
            prefix = mine.iloc[:k]
            targets = mine.iloc[k:k + W]
            if not len(targets):
                continue
            assert (targets["time"].values > prefix["time"].values[-1]).all() if k else True
            assert not prefix["sha256"].isin(targets["sha256"]).any()
            hf = prefix_hist_features(prefix, targets, scaler) if k else \
                pd.DataFrame(np.nan, index=targets.index, columns=HIST)
            te_k = targets.copy()
            te_k[HIST_FEW] = hf[HIST_FEW].values
            if RESP:
                # prefix-only profile (see response_features.prefix_response): the
                # curve is fitted on the prefix alone and evaluated at each target's
                # own axis value, so no target can inform another.
                te_k[RESP] = prefix_response(prefix, targets, sd_axes,
                                             min_n=RESP_MIN, window=RESP_WIN,
                                             shrink=RESP_LAMBDA).values

            p0_acc = m0_acc.predict(te_k)
            p0_lamp = m0_lamp.predict(te_k)
            p2_acc = m2_acc.predict(te_k)
            p2_lamp = m2_lamp.predict(te_k)
            p0_bp = np.expm1(np.clip(m0_bp.predict(te_k), 0, bp_cap))
            p2_bp = np.expm1(np.clip(m2_bp.predict(te_k), 0, bp_cap))
            pk_acc = te_k["h_knn_acc"].values
            pk_lamp = te_k["h_acc_mean"].values / 100 * 6
            pk_bp = te_k["h_bp_mean"].values

            curve.append({
                "k": k, "n_eval": int(len(te_k)),
                "M0_acc": round(mae(te_k["acc"], p0_acc), 3),
                "M1_acc": round(mae(te_k["acc"], pk_acc), 3),
                "M2_acc": round(mae(te_k["acc"], p2_acc), 3),
                "M0_lamp": round(mae(te_k["lamp"], p0_lamp), 3),
                "M1_lamp": round(mae(te_k["lamp"], pk_lamp), 3),
                "M2_lamp": round(mae(te_k["lamp"], p2_lamp), 3),
                "M0_bp": round(mae(te_k["bp"], p0_bp), 2),
                "M1_bp": round(mae(te_k["bp"], pk_bp), 2),
                "M2_bp": round(mae(te_k["bp"], p2_bp), 2),
            })
        res["fewshot"] = curve
        results["lopo"][D] = res
        e1 = res["exp1"]
        print(f"{D:9} exp1: M0 {e1['M0']['acc_mae']:.2f} M1 {e1['M1']['acc_mae']:.2f} "
              f"M2 {e1['M2']['acc_mae']:.2f} | k=50: "
              f"M1 {next(c for c in curve if c['k']==50)['M1_acc']:.2f} "
              f"M2 {next(c for c in curve if c['k']==50)['M2_acc']:.2f}")

    # ---------- aggregate few-shot curves (weighted by n_eval) ----------
    agg = []
    for k in KS:
        rows = [c for D in players for c in results["lopo"][D]["fewshot"] if c["k"] == k]
        if not rows:
            continue
        w = np.array([c["n_eval"] for c in rows], float)
        entry = {"k": k, "n_eval": int(w.sum())}
        for key in ["M0_acc", "M1_acc", "M2_acc", "M0_lamp", "M1_lamp", "M2_lamp",
                    "M0_bp", "M1_bp", "M2_bp"]:
            entry[key] = round(float(np.average([c[key] for c in rows], weights=w)), 3)
        agg.append(entry)
    results["fewshot_aggregate"] = agg

    json.dump(results, open(OUT / f"transfer_results{args.tag}.json", "w"),
              indent=2, default=str)
    print("\nfew-shot aggregate (weighted):")
    for e in agg:
        print(f"  k={e['k']:3}: M0 {e['M0_acc']:.2f}  M1 {e['M1_acc']:.2f}  "
              f"M2 {e['M2_acc']:.2f} | lamp M1 {e['M1_lamp']:.2f} M2 {e['M2_lamp']:.2f} "
              f"| BP M1 {e['M1_bp']:.1f} M2 {e['M2_bp']:.1f}")


if __name__ == "__main__":
    main()
