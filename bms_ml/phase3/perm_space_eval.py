"""Phase 3.6: evaluate the permutation-space chart geometry encoder (perm_space).

Same rows / split / protocol / HGB as compare_nolevel.py and chart_v2_eval.py, so
every number is directly comparable to the v1 (27-dim) and v2 (14-dim) baselines.

  A            chart-only v1 (27)
  A_v2         chart-only v1 + v2 (27+14)
  A_perm       chart-only v1 + perm (27+22)              <- chart-side marginal value
  A_v2_perm    chart-only v1 + v2 + perm (27+36)
  B            chart+history v1                          <- current best
  B_v2         + v2
  B_perm       + perm
  B_v2_perm    + v2 + perm                               <- candidate

Group ablations on top of B_v2 isolate smooth / tight / base / spread / scratch,
and `perm_redundancy` reports the max |corr| of each perm feature against v1+v2 to
show the family is not a re-parameterisation of what we already had.

Deterministic HGB (fixed random_state) — no seed spread for the headline table;
the seed spread of HGB is reported for B/B_v2_perm as a sanity check.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from chart_repr import (HISTORY_FEATURES, OBJECTIVE_PERM_COLS, OBJECTIVE_STAT_COLS,
                        OBJECTIVE_V2_COLS, feature_manifest)
from common import centered_r2, hgb_fit_predict, load_samples, mae, r2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"

GROUPS = {
    "smooth": [c for c in OBJECTIVE_PERM_COLS if c.startswith("ps_smooth")],
    "tight": [c for c in OBJECTIVE_PERM_COLS if c.startswith("ps_tight")],
    "base": [c for c in OBJECTIVE_PERM_COLS if c.startswith("ps_base_")],
    "spread": [c for c in OBJECTIVE_PERM_COLS if c.startswith("ps_spread")],
    "scratch": [c for c in OBJECTIVE_PERM_COLS if c.startswith("ps_scratch")],
}
# permutation-space DISPERSION terms: how much the chart's hand-travel demand depends
# on the random arrangement. Not derivable from any single-arrangement statistic.
DISPERSION = [c for c in OBJECTIVE_PERM_COLS
              if c.endswith("_std") or c.endswith("_base_pct")]


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
    v2 = pd.read_parquet(DS / "chart_stats_v2.parquet")
    ps = pd.read_parquet(DS / "chart_perm_space.parquet")
    df = df.merge(v2, on="sha256", how="left").merge(ps, on="sha256", how="left")
    assert df["v2_ioi_lane_p05"].notna().mean() > 0.99, "v2 join failed"
    frac = float(df[OBJECTIVE_PERM_COLS].isna().all(axis=1).mean())
    tr, te = df[df["phase"] == "train"], df[df["phase"] == "test"]

    V1, V2, PS, H = (OBJECTIVE_STAT_COLS, OBJECTIVE_V2_COLS, OBJECTIVE_PERM_COLS,
                     HISTORY_FEATURES)
    sets = {
        "A": V1,
        "A_v2": V1 + V2,
        "A_perm": V1 + PS,
        "A_v2_perm": V1 + V2 + PS,
        "B": V1 + H,
        "B_v2": V1 + V2 + H,
        "B_perm": V1 + PS + H,
        "B_v2_perm": V1 + V2 + PS + H,
    }
    for g, cols in GROUPS.items():
        sets[f"B_v2+{g}"] = V1 + V2 + cols + H
    # the correlation audit shows *_std / *_base_pct are the ONLY genuinely new terms
    # (max |corr| vs v1+v2 = 0.14-0.33, vs 0.55-0.72 for *_mean/min/max and 0.93-0.96
    # for ps_scratch_*). This set tests whether "new information" is also USEFUL.
    sets["B+disp"] = V1 + DISPERSION + H
    sets["B_v2+disp"] = V1 + V2 + DISPERSION + H

    results: dict = {"n_train": len(tr), "n_test": len(te),
                     "perm_all_nan_frac": round(frac, 4)}
    print(f"{'set':16}{'acc':>8}{'cR2':>9}{'lamp':>8}{'QWK':>8}{'BP':>9}")
    for tag, feats in sets.items():
        results[tag] = evaluate(tr, te, feats)
        r = results[tag]
        print(f"{tag:16}{r['acc']['mae']:8.3f}{r['acc']['centered_r2']:+9.3f}"
              f"{r['lamp']['ord_mae']:8.3f}{r['lamp']['qwk']:8.3f}{r['bp']['raw_mae']:9.1f}")

    # seed spread sanity check (HGB is deterministic; this shows the numerical floor)
    for tag in ("B", "B_v2_perm"):
        ms = [mae(te["acc"], hgb_fit_predict(tr, te, sets[tag], tr["acc"].values, seed=s))
              for s in (0, 1, 2)]
        results[f"{tag}_seed_std"] = round(float(np.std(ms)), 4)

    # is the family a re-parameterisation of v1+v2?  max |corr| per perm feature
    base = V1 + V2
    corr = {}
    for c in PS:
        cc = df[base + [c]].corr()[c].drop(c)
        corr[c] = round(float(np.nanmax(np.abs(cc.values))), 3)
    results["perm_vs_v1v2_maxabs_corr"] = corr

    results["features_used"] = feature_manifest(*sets.values())
    json.dump(results, open(OUT / "perm_space_eval.json", "w"), indent=2, default=str)
    print("saved ->", OUT / "perm_space_eval.json")


if __name__ == "__main__":
    main()
