"""Phase 3.6: which response-profile axes actually carry the information?

B_full = v1(27) + history(12) + response profile(24 = 11 axes x {resp, slope}, +mean/std)
and beats B by 0.426 acc (EXPERIMENT_LOG 2026-09-11). Before extending the idea (more
axes? other estimators?) it is worth knowing WHERE the gain comes from.

Two complementary views, both against the identical base model:
  leave-one-axis-out  drop h_resp_<a> and h_slope_<a>  -> how much is that axis worth?
  axis-only           base + only that axis's two columns -> how much does it carry alone?

Leave-one-out is the primary view but it understates correlated axes (dropping one of two
near-duplicates costs nothing); axis-only understates axes whose value is their PATTERN
across axes. Read them together, and do not treat a ~0.02 delta as real - HGB is
deterministic so the numbers are reproducible, but that is not the same as meaningful.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from chart_repr import (HISTORY_FEATURES, HISTORY_RESPONSE_COLS, OBJECTIVE_STAT_COLS,
                        RESPONSE_AXES, feature_manifest)
from common import centered_r2, hgb_fit_predict, load_samples, mae

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"


def evaluate(tr, te, feats) -> dict:
    acc_p = hgb_fit_predict(tr, te, feats, tr["acc"].values)
    lamp_p = hgb_fit_predict(tr, te, feats, tr["lamp"].values.astype(float))
    bp_p = np.expm1(hgb_fit_predict(tr, te, feats, np.log1p(tr["bp"].values)))
    return {"acc_mae": round(mae(te["acc"], acc_p), 3),
            "cR2": round(centered_r2(te, acc_p, "acc"), 3),
            "lamp": round(mae(te["lamp"], lamp_p), 3),
            "qwk": round(float(cohen_kappa_score(
                te["lamp"], np.clip(np.round(lamp_p), 1, 9).astype(int),
                weights="quadratic", labels=list(range(1, 10)))), 3),
            "bp": round(mae(te["bp"], bp_p), 1)}


def main() -> None:
    df = load_samples()
    hr = pd.read_parquet(DS / "history_response.parquet")
    df = df.merge(hr[["player", "sha256"] + HISTORY_RESPONSE_COLS],
                  on=["player", "sha256"], how="left")
    tr, te = df[df["phase"] == "train"], df[df["phase"] == "test"]

    CORE = OBJECTIVE_STAT_COLS + HISTORY_FEATURES
    FULL = CORE + HISTORY_RESPONSE_COLS
    results: dict = {}

    def run(tag, feats, show=True):
        r = evaluate(tr, te, feats)
        results[tag] = r
        if show:
            print(f"{tag:26}{r['acc_mae']:8.3f}{r['cR2']:+9.3f}"
                  f"{r['lamp']:8.3f}{r['qwk']:7.3f}{r['bp']:8.1f}")
        return r

    print(f"{'variant':26}{'acc':>8}{'cR2':>9}{'lamp':>8}{'QWK':>7}{'BP':>8}")
    base = run("B (no profile)", CORE)
    full = run("B_full (all 11 axes)", FULL)
    print(f"{'-- leave-one-axis-out':26}  d_acc   d_cR2")
    loo = {}
    for a in RESPONSE_AXES:
        drop = [f"h_resp_{a}", f"h_slope_{a}"]
        r = run(f"  -{a}", [c for c in FULL if c not in drop], show=False)
        loo[a] = {"d_acc": round(r["acc_mae"] - full["acc_mae"], 3),
                  "d_cR2": round(r["cR2"] - full["cR2"], 3)}
        print(f"  -{a:20}{loo[a]['d_acc']:+8.3f}{loo[a]['d_cR2']:+9.3f}")
    results["leave_one_axis_out"] = loo

    print(f"{'-- axis-only (base + 2 cols)':26}")
    only = {}
    for a in RESPONSE_AXES:
        r = run(f"  {a}", CORE + [f"h_resp_{a}", f"h_slope_{a}"], show=False)
        only[a] = {"d_acc": round(r["acc_mae"] - base["acc_mae"], 3),
                   "d_cR2": round(r["cR2"] - base["cR2"], 3)}
        print(f"  {a:20}{only[a]['d_acc']:+8.3f}{only[a]['d_cR2']:+9.3f}")
    results["axis_only"] = only

    # means / stds on their own, and the two halves separately
    R = [c for c in HISTORY_RESPONSE_COLS if c.startswith("h_resp_") and
         c not in ("h_resp_mean", "h_resp_std")]
    S = [c for c in HISTORY_RESPONSE_COLS if c.startswith("h_slope_")]
    print(f"{'-- halves':26}")
    results["resp_only"] = run("h_resp_* (11 cols)", CORE + R)
    results["slope_only"] = run("h_slope_* (11 cols)", CORE + S)
    results["mean_std_only"] = run("h_resp_mean/std", CORE + ["h_resp_mean", "h_resp_std"])

    results["features_used"] = feature_manifest(CORE, FULL)
    json.dump(results, open(OUT / "response_ablation.json", "w"), indent=2, default=str)
    print("saved ->", OUT / "response_ablation.json")


if __name__ == "__main__":
    main()
