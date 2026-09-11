"""Phase 3.6: MinaCalc MSD as a response/dev axis family - "better axes" or "more axes"?

The MMA reference project ships MinaCalc (Etterna's crowd-calibrated difficulty model) as
a WASM, and dataset/msd.parquet carries 7 skillset ratings per chart. This is the
hypothesis test flagged in the review: MSD encodes crowd-calibrated difficulty SEMANTICS
("this chart's chordjack demand is 14.1"), which our raw structural stats do not. But
dev20 already showed that ADDING axes is not automatically a gain, so this is decided by
the standard evaluation plus a paired test, not by intuition.

Configs, all on top of the current best (B_resp+bpresp+dev at 180d half-life):
  BASE+msdraw          the 7 MSD ratings as plain chart-side features
  BASE+msdresp         a full response/dev block on the 7 MSD axes (prefix m_/ml_/mb_)
  BASE+msdraw+msdresp  both
  B_msdonly            structural response blocks REPLACED by the MSD block - can crowd
                       semantics substitute our structural profile?
  shuffled controls    the msdresp columns permuted across rows (capacity check)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chart_repr import feature_manifest
from common import load_samples
from response_decay import V1, Hv, build_response
from response_dev import DEV, SETS_DEV
from response_eval import evaluate
from response_features import _block_cols, axis_sd, build_table

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "bms_ml" / "output" / "phase3" / "dataset"
HALF_LIFE = 180.0
MIN_HISTORY = 20

MSD_AXES = {k: f"msd_{k}" for k in
            ("overall", "stream", "jumpstream", "handstream", "stamina",
             "jackspeed", "chordjack")}
MSD_BLOCK = _block_cols("m_", MSD_AXES)      # acc block only; lamp/bp variants below
MSD_RAW = list(MSD_AXES.values())
BASE = SETS_DEV["B_resp+bpresp+dev"]         # the acc best (6.117)
STRUCT = SETS_DEV["B_resp+bpresp"]           # structural response config, for reference


def build_msd_block() -> tuple[pd.DataFrame, float]:
    """Response/dev features on the MSD axes for every first-play row.

    The MSD axes are player-relative too: m_resp_* is the player's own OLS of past acc on
    past MSD, evaluated at the target chart's MSD; m_dev_* is the target's MSD z-scored
    within the player's own played range. Raw MSD columns enter separately (msdraw).
    """
    fp = pd.read_parquet(DS / "firstplays.parquet")
    fp = fp.merge(pd.read_parquet(DS / "msd.parquet"), on="sha256", how="left")
    fp = fp.sort_values(["player", "time"], kind="stable").reset_index(drop=True)
    sd = axis_sd(fp, MSD_AXES)
    fp["log1p_bp"] = np.log1p(fp["bp"].values.astype(np.float64))
    blocks = [
        build_table(fp, sd, min_n=MIN_HISTORY, target="acc", prefix="m_",
                    clip=(0.0, 100.0), half_life=HALF_LIFE, axes=MSD_AXES),
        build_table(fp, sd, min_n=MIN_HISTORY, target="lamp", prefix="ml_",
                    clip=(1.0, 9.0), half_life=HALF_LIFE, axes=MSD_AXES),
        build_table(fp, sd, min_n=MIN_HISTORY, target="log1p_bp", prefix="mb_",
                    clip=(0.0, 12.0), half_life=HALF_LIFE, axes=MSD_AXES),
    ]
    feats = pd.concat(blocks, axis=1)
    cov = float(feats["m_resp_overall"].notna().mean())
    print(f"msd block: {len(feats.columns)} columns, coverage {cov:.3f}")
    return pd.concat([fp[["player", "sha256"]], feats], axis=1), cov


def main() -> None:
    resp, sd = build_response(HALF_LIFE, return_sd=True)
    msd_tbl, cov = build_msd_block()
    df = (load_samples()
          .merge(resp, on=["player", "sha256"], how="left")
          .merge(pd.read_parquet(DS / "msd.parquet"), on="sha256", how="left")
          .merge(msd_tbl, on=["player", "sha256"], how="left"))
    tr, te = df[df["phase"] == "train"], df[df["phase"] == "test"]

    sets = {
        "BASE": BASE,
        "BASE+msdraw": BASE + MSD_RAW,
        "BASE+msdresp": BASE + MSD_BLOCK,
        "BASE+msdraw+msdresp": BASE + MSD_RAW + MSD_BLOCK,
        "B_msdonly": V1 + Hv + MSD_BLOCK,
        "STRUCT(ref)": STRUCT,
    }
    results: dict = {"half_life": HALF_LIFE, "msd_coverage": cov}
    print(f"{'set':24}{'acc':>8}{'cR2':>9}{'lamp':>8}{'QWK':>8}{'BP':>9}")
    for tag, feats in sets.items():
        r = evaluate(tr, te, feats)
        results[tag] = r
        print(f"{tag:24}{r['acc']['mae']:8.3f}{r['acc']['centered_r2']:+9.3f}"
              f"{r['lamp']['ord_mae']:8.3f}{r['lamp']['qwk']:8.3f}"
              f"{r['bp']['raw_mae']:9.1f}")

    for s in (0, 1):
        rs = np.random.RandomState(100 + s)
        sh = df.copy()
        for c in MSD_BLOCK:
            sh[c] = rs.permutation(sh[c].values)
        r = evaluate(sh[sh["phase"] == "train"], sh[sh["phase"] == "test"],
                     sets["BASE+msdresp"])
        results[f"msdresp_shuffled_s{s}"] = r
        print(f"{'  msdresp shuffled s' + str(s):24}{r['acc']['mae']:8.3f}"
              f"{r['acc']['centered_r2']:+9.3f}{r['lamp']['ord_mae']:8.3f}"
              f"{r['lamp']['qwk']:8.3f}{r['bp']['raw_mae']:9.1f}")

    better = [t for t in ("BASE+msdraw", "BASE+msdresp", "BASE+msdraw+msdresp")
              if results[t]["acc"]["mae"] < results["BASE"]["acc"]["mae"]]
    if better:
        from common import hgb_fit_predict
        from scipy import stats as _st
        best = min(better, key=lambda t: results[t]["acc"]["mae"])
        errs = {}
        for tag in ("BASE", best):
            pr = hgb_fit_predict(tr, te, sets[tag], tr["acc"].values)
            errs[tag] = np.abs(te["acc"].values - pr)
        diff = errs["BASE"] - errs[best]
        tt = _st.ttest_1samp(diff, 0.0)
        rng = np.random.RandomState(0)
        boot = diff[rng.randint(0, len(diff), size=(2000, len(diff)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        results["paired"] = {"vs": "BASE", "best": best,
                             "mean": round(float(diff.mean()), 4),
                             "ci95": [round(float(lo), 4), round(float(hi), 4)],
                             "t": round(float(tt.statistic), 3),
                             "p": float(f"{tt.pvalue:.3g}"),
                             "helped": round(float((diff > 0).mean()), 4)}
        print(f"\npaired {best} vs BASE: +{diff.mean():.4f} "
              f"(95% CI {lo:.4f}..{hi:.4f}) t={tt.statistic:.2f} p={tt.pvalue:.3g} "
              f"helped {(diff > 0).mean():.1%}")

    results["features_used"] = feature_manifest(*sets.values())
    json.dump(results, open(ROOT / "bms_ml" / "output" / "phase3"
                            / "response_msd.json", "w"), indent=2, default=str)
    print("saved -> response_msd.json")


if __name__ == "__main__":
    main()
