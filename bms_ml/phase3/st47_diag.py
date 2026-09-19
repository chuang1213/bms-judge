"""Phase 3.5 (task 3): why does cross-player transfer FAIL in ST4-7?

The only region where M2 (population) loses to M1 (D-local): M2 - M1 = -1.11
acc MAE (PHASE3_4_TRANSFER.md §6). Two competing explanations:
  (a) too little training mass (~8% of rows are ST4-7)  -> upsampling should fix it
  (b) genuine player heterogeneity in the high-mid band -> upsampling cannot fix it

Diagnostic: replicate the LOPO M2 with ST4-7 training rows upweighted
(sample_weight, no data duplication), evaluate on each held-out player's
ST4-7 test rows only. Weights 1 (baseline) / 3 / 6. If the gap to M1 closes
with weight, it was data; if not, it is heterogeneity.

This is an ANALYSIS, not a protocol change: nothing here alters default
training. Region labels are reporting coordinates only (PROTOCOL.md §1).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from chart_repr import HISTORY_FEATURES, OBJECTIVE_STAT_COLS
from common import Imputer, add_region, difficulty_region, load_firstplays, mae

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
WEIGHTS = [1, 3, 6]
TARGET_REGION = "ST4-7"


def hgb_pred_weighted(tr, te, feats, y, w):
    imp = Imputer()
    Xtr = imp.fit_transform(tr[feats])
    sc = StandardScaler().fit(Xtr)
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                      max_depth=3, random_state=0)
    m.fit(sc.transform(Xtr), y, sample_weight=w)
    return m.predict(sc.transform(imp.transform(te[feats])))


def main() -> None:
    fp = add_region(load_firstplays().reset_index(drop=True))
    players = sorted(fp["player"].unique())
    per_player, agg = {}, {w: [] for w in WEIGHTS}

    for D in players:
        others = fp[fp["player"] != D]
        mine = fp[fp["player"] == D]
        te = mine[mine["phase"] == "test"]
        te_r = te[te["region"] == TARGET_REGION]
        if len(te_r) < 10:            # too few ST4-7 test rows to judge
            continue
        feats = OBJECTIVE_STAT_COLS + HISTORY_FEATURES
        base_w = np.ones(len(others))
        reg_w = np.where(others["region"].values == TARGET_REGION, 1.0, 1.0)
        m1 = mae(te_r["acc"], te_r["h_knn_acc"])
        row = {"n": int(len(te_r)), "M1": round(m1, 2)}
        for w in WEIGHTS:
            ww = base_w.copy()
            if w > 1:
                ww[others["region"].values == TARGET_REGION] = float(w)
            p = hgb_pred_weighted(others, te_r, feats, others["acc"].values, ww)
            row[f"M2_w{w}"] = round(mae(te_r["acc"], p), 2)
            agg[w].append((len(te_r), mae(te_r["acc"], p), m1))
        per_player[D] = row
        print(D, row)

    summary = {}
    for w in WEIGHTS:
        n = sum(a[0] for a in agg[w])
        m2 = sum(a[0] * a[1] for a in agg[w]) / n
        m1 = sum(a[0] * a[2] for a in agg[w]) / n
        summary[f"w={w}"] = {"n_eval": n, "M2_acc": round(m2, 2), "M1_acc": round(m1, 2),
                             "M2_minus_M1": round(m2 - m1, 2)}
    print("\nST4-7 weighted summary:", json.dumps(summary, indent=1))
    json.dump({"target_region": TARGET_REGION, "per_player": per_player,
               "summary": summary},
              open(OUT / "st47_diag.json", "w"), indent=2)
    print("saved ->", OUT / "st47_diag.json")


if __name__ == "__main__":
    main()
