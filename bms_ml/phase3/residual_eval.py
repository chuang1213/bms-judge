"""Phase 3.6: does a RESIDUAL target (predict within-player deviation) beat the
absolute target?

Rationale (PHASE3_5_REVIEW §4.6): the project's headline research metric is centered
R², i.e. within-player interaction, while the model is trained on absolute acc. The
between-player spread (player means range 48.8-88.1, sd ~10.1) is larger than the
model's own MAE (~7), so most of the training loss is spent on a component the model
already gets almost for free from h_acc_mean - and errors on the few players with
extreme levels dominate the gradient. Fitting the residual makes the loss equal to
the quantity the project actually cares about.

Offsets are causal constants: each player's mean acc over the TRAIN band only (test
rows never contribute to their own offset). Two variants are compared against the
identical plain-target model:

  plain      HGB(B_full -> acc)
  residual   HGB(B_full -> acc - offset), prediction = fit + offset
  residual+  same, but the offset is the response-profile mean (h_resp_mean), which is
             a strictly-prior, time-varying estimate of the player's level - the
             deployable form when no train band exists (a brand-new player).

No leakage: offsets come from the train band (or from strictly-prior history), never
from the row's own outcome.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from chart_repr import (HISTORY_FEATURES, HISTORY_RESPONSE_COLS, OBJECTIVE_STAT_COLS,
                        feature_manifest)
from common import centered_r2, hgb_fit_predict, load_samples, mae, r2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"

FEATS = OBJECTIVE_STAT_COLS + HISTORY_FEATURES + HISTORY_RESPONSE_COLS


def _metrics(te, pred) -> dict:
    return {"acc": {"mae": round(mae(te["acc"], pred), 3),
                    "r2": round(r2(te["acc"], pred), 3),
                    "centered_r2": round(centered_r2(te, pred, "acc"), 3),
                    "per_player": {p: round(mae(g["acc"], pred[te["player"].values == p]), 3)
                                   for p, g in te.groupby("player")}}}


def main() -> None:
    df = load_samples()
    hr = pd.read_parquet(DS / "history_response.parquet")
    df = df.merge(hr[["player", "sha256"] + HISTORY_RESPONSE_COLS],
                  on=["player", "sha256"], how="left")
    tr, te = df[df["phase"] == "train"], df[df["phase"] == "test"]

    # causal offsets
    off_tr = tr.groupby("player")["acc"].mean()
    o_te = te["player"].map(off_tr).values.astype(float)
    o_te_fallback = te["h_resp_mean"].values.astype(float)   # deployable, prior-only

    preds = {}
    preds["plain"] = hgb_fit_predict(tr, te, FEATS, tr["acc"].values)
    y_tr = tr["acc"].values
    off_fit = tr["player"].map(off_tr).values.astype(float)
    r = hgb_fit_predict(tr, te, FEATS, y_tr - off_fit)
    preds["residual"] = r + o_te
    # deployable variant: both sides use the prior-only response-profile mean
    off_tr2 = tr["h_resp_mean"].values.astype(float)
    off_tr2 = np.where(np.isnan(off_tr2), np.nanmean(off_tr2), off_tr2)
    r2v = hgb_fit_predict(tr, te, FEATS, y_tr - off_tr2)
    preds["residual_prior"] = r2v + np.where(np.isnan(o_te_fallback),
                                             np.nanmean(off_tr2), o_te_fallback)

    results = {"n_train": len(tr), "n_test": len(te)}
    print(f"{'variant':16}{'acc':>8}{'cR2':>9}{'R2':>8}")
    for k, p in preds.items():
        results[k] = _metrics(te, p)
        m = results[k]["acc"]
        print(f"{k:16}{m['mae']:8.3f}{m['centered_r2']:+9.3f}{m['r2']:8.3f}")

    results["features_used"] = feature_manifest(FEATS)
    json.dump(results, open(OUT / "residual_eval.json", "w"), indent=2, default=str)
    print("saved ->", OUT / "residual_eval.json")


if __name__ == "__main__":
    main()
