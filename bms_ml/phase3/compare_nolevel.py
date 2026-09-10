"""Phase 3 baselines: objective chart features vs hand-crafted history statistics.

Baselines (PROTOCOL.md §5 — A is the lower bound, H/B are the fixed references):
  A  chart-only      27-dim objective chart statistics
  H  history-only    12 hand-crafted player-state statistics
  B  chart + history both

Scope note (2026-09-07, Phase 3.5): this script used to report a `table` and an
`all` scope. `data.py` now fences targets to the sl/st/発狂2018 union, so both
scopes became byte-identical; the duplicate run was removed rather than kept as a
silently-equal second number.

Feature lists come from `chart_repr.py` (PROTOCOL.md §3 single registry); metrics
and the HGB wrapper come from `common.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from chart_repr import (HISTORY_FEATURES, HISTORY_RESPONSE_COLS,
                        HISTORY_RESPONSE_LAMP_COLS, OBJECTIVE_STAT_COLS,
                        feature_manifest)
from common import centered_r2, hgb_fit_predict, load_samples, mae, r2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"


def main() -> None:
    df = load_samples()
    res = {"n_train": 0, "n_test": 0}

    sets = {"A": OBJECTIVE_STAT_COLS,
            "H": HISTORY_FEATURES,
            "B": OBJECTIVE_STAT_COLS + HISTORY_FEATURES}

    # B_resp = the current best configuration (PHASE3_6_REPORT.md §1): all of B plus the
    # per-axis personal response blocks. Acc block full (24), lamp block chart-
    # conditioned only (13). Soft dependency: response_eval.py owns the full study and
    # history_response.py produces the parquet, so if it is missing we say so rather
    # than making the canonical baseline script fail.
    resp_path = DS / "history_response.parquet"
    if resp_path.exists():
        hr = pd.read_parquet(resp_path)
        df = df.merge(hr[["player", "sha256"] + HISTORY_RESPONSE_COLS
                         + HISTORY_RESPONSE_LAMP_COLS],
                      on=["player", "sha256"], how="left")
        lamp_resp = [c for c in HISTORY_RESPONSE_LAMP_COLS
                     if c.startswith("l_resp_")
                     and c not in ("l_resp_mean", "l_resp_std")]
        sets["B_resp"] = (OBJECTIVE_STAT_COLS + HISTORY_FEATURES
                          + HISTORY_RESPONSE_COLS + lamp_resp
                          + ["l_resp_mean", "l_resp_std"])
    else:
        print(f"[note] {resp_path.name} missing -> B_resp not reported "
              f"(run bms_ml/phase3/history_response.py)")

    tr, te = df[df["phase"] == "train"], df[df["phase"] == "test"]
    res["n_train"], res["n_test"] = len(tr), len(te)
    for tag, feats in sets.items():
        acc_p = hgb_fit_predict(tr, te, feats, tr["acc"].values)
        lamp_p = hgb_fit_predict(tr, te, feats, tr["lamp"].values.astype(float))
        # BP target is log1p(raw) — fixed 2026-09-03 (skew 2.56 -> 0.15); the
        # ratio track is a reporting view only, never a training target.
        bp_p = np.expm1(hgb_fit_predict(tr, te, feats, np.log1p(tr["bp"].values)))
        res[tag] = {
            "acc": {"mae": round(mae(te["acc"], acc_p), 3),
                    "r2": round(r2(te["acc"], acc_p), 3),
                    "centered_r2": round(centered_r2(te, acc_p, "acc"), 3),
                    "per_player": {p: round(mae(g["acc"], acc_p[te["player"].values == p]), 3)
                                   for p, g in te.groupby("player")}},
            "lamp": {"ord_mae": round(mae(te["lamp"], lamp_p), 3),
                     "qwk": round(float(cohen_kappa_score(
                         te["lamp"], np.clip(np.round(lamp_p), 1, 9).astype(int),
                         weights="quadratic", labels=list(range(1, 10)))), 3)},
            "bp": {"raw_mae": round(mae(te["bp"], bp_p), 2),
                   "ratio_mae_x1000": round(mae(te["bp"] / te["notes"],
                                                bp_p / te["notes"]) * 1000, 2)},
        }

    res["features_used"] = feature_manifest(OBJECTIVE_STAT_COLS, HISTORY_FEATURES)
    json.dump(res, open(OUT / "compare_nolevel.json", "w"), indent=2)
    print({t: round(res[t]["acc"]["mae"], 3) for t in sets},
          "| n =", len(tr), "/", len(te))
    print("saved ->", OUT / "compare_nolevel.json")


if __name__ == "__main__":
    main()
