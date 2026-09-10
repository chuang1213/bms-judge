"""Phase 3.6: evaluate the per-axis personal response profile (history side).

Same rows / split / protocol / HGB as compare_nolevel.py, so numbers are directly
comparable. The question (PHASE3_5_REVIEW §4.3): does replacing the single 27-dim
kNN conditioning with an explicit per-axis response curve improve the player state?

  H           12 history scalars (current)
  H_resp_only 14 response-profile terms alone          <- standalone information?
  H_full      12 + 14 = 26
  B           chart + H
  B_full      chart + H + profile                      <- candidate
  B_full_v2   + v2 chart stats

Group ablations isolate the chart-conditioned half (`h_resp_*`, includes mean/std)
from the pure-trait half (`h_slope_*`).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from chart_repr import (HISTORY_FEATURES, HISTORY_RESPONSE_COLS, OBJECTIVE_STAT_COLS,
                        OBJECTIVE_V2_COLS, feature_manifest)
from common import centered_r2, hgb_fit_predict, load_samples, mae, r2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"

RESP = [c for c in HISTORY_RESPONSE_COLS if c.startswith("h_resp_")]
SLOPE = [c for c in HISTORY_RESPONSE_COLS if c.startswith("h_slope_")]


def evaluate(tr, te, feats, seed=0) -> dict:
    acc_p = hgb_fit_predict(tr, te, feats, tr["acc"].values, seed=seed)
    lamp_p = hgb_fit_predict(tr, te, feats, tr["lamp"].values.astype(float), seed=seed)
    bp_p = np.expm1(hgb_fit_predict(tr, te, feats, np.log1p(tr["bp"].values), seed=seed))
    return {
        "acc": {"mae": round(mae(te["acc"], acc_p), 3),
                "r2": round(r2(te["acc"], acc_p), 3),
                "centered_r2": round(centered_r2(te, acc_p, "acc"), 3),
                "per_player": {p: round(mae(g["acc"], acc_p[te["player"].values == p]), 3)
                               for p, g in te.groupby("player")}},
        "lamp": {"ord_mae": round(mae(te["lamp"], lamp_p), 3),
                 "qwk": round(float(cohen_kappa_score(
                     te["lamp"], np.clip(np.round(lamp_p), 1, 9).astype(int),
                     weights="quadratic", labels=list(range(1, 10)))), 3)},
        "bp": {"raw_mae": round(mae(te["bp"], bp_p), 2)},
    }


def main() -> None:
    df = load_samples()
    hr = pd.read_parquet(DS / "history_response.parquet")
    assert not hr.duplicated(["player", "sha256"]).any(), "response key not unique"
    df = df.merge(hr[["player", "sha256"] + HISTORY_RESPONSE_COLS],
                  on=["player", "sha256"], how="left")
    v2 = pd.read_parquet(DS / "chart_stats_v2.parquet")
    df = df.merge(v2, on="sha256", how="left")
    cov = float(df["h_resp_mean"].isna().mean())
    tr, te = df[df["phase"] == "train"], df[df["phase"] == "test"]

    V1, V2, Hv = OBJECTIVE_STAT_COLS, OBJECTIVE_V2_COLS, HISTORY_FEATURES
    sets = {
        "H": Hv,
        "H_resp_only": HISTORY_RESPONSE_COLS,
        "H_full": Hv + HISTORY_RESPONSE_COLS,
        "B": V1 + Hv,
        "B+resp": V1 + Hv + RESP + ["h_resp_mean", "h_resp_std"],
        "B+slope": V1 + Hv + SLOPE,
        "B_full": V1 + Hv + HISTORY_RESPONSE_COLS,
        "B_full_v2": V1 + V2 + Hv + HISTORY_RESPONSE_COLS,
    }

    results: dict = {"n_train": len(tr), "n_test": len(te),
                     "h_resp_mean_missing_frac": round(cov, 4)}
    print(f"{'set':14}{'acc':>8}{'cR2':>9}{'lamp':>8}{'QWK':>8}{'BP':>9}")
    for tag, feats in sets.items():
        results[tag] = evaluate(tr, te, feats)
        r = results[tag]
        print(f"{tag:14}{r['acc']['mae']:8.3f}{r['acc']['centered_r2']:+9.3f}"
              f"{r['lamp']['ord_mae']:8.3f}{r['lamp']['qwk']:8.3f}{r['bp']['raw_mae']:9.1f}")

    for tag in ("H", "B", "B_full_v2"):
        ms = [mae(te["acc"], hgb_fit_predict(tr, te, sets[tag], tr["acc"].values, seed=s))
              for s in (0, 1, 2)]
        results[f"{tag}_seed_std"] = round(float(np.std(ms)), 4)

    # who gains? per-player delta on the headline comparison
    pa_h = hgb_fit_predict(tr, te, sets["H"], tr["acc"].values)
    pa_f = hgb_fit_predict(tr, te, sets["H_full"], tr["acc"].values)
    results["per_player_H_vs_Hfull"] = {
        p: round(float(mae(g["acc"], pa_f[te["player"].values == p])
                       - mae(g["acc"], pa_h[te["player"].values == p])), 3)
        for p, g in te.groupby("player")}

    # correlation of the new terms with what h_knn_acc already carries
    base = V1 + Hv
    results["resp_vs_base_maxabs_corr"] = {
        c: round(float(np.nanmax(np.abs(df[base + [c]].corr()[c].drop(c).values))), 3)
        for c in HISTORY_RESPONSE_COLS}

    results["features_used"] = feature_manifest(*sets.values())
    json.dump(results, open(OUT / "response_eval.json", "w"), indent=2, default=str)
    print("saved ->", OUT / "response_eval.json")


if __name__ == "__main__":
    main()
