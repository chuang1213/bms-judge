"""Phase 3.6: recency-weighted response profile — half-life sweep.

Motivation (2026-09-11). The response profile is the single biggest gain in the model
(B 7.009 -> B_resp+bpresp 6.495) but it was fitted with a UNIFORM-weighted OLS over the
player's entire causal history. Measured facts that make that suspicious:

  * archives span a median of 959 days (max 2000);
  * within-player acc drifts by +8.4pp on average across the archive (sd 11.8, >5pp on
    13/18 players, tzh +41.9, yangtao -15.5);
  * the project already established that calendar time is worth ~1.33 acc MAE and that
    removing the time features collapses centered R2 to ~0.

So the profile described each player's LIFETIME average response while every target row
sits in their most recent quartile. This script refits all three response blocks
(acc / lamp / bp) with exponential forgetting, weight = 0.5 ** (age_days / H), and sweeps
H. H = None reproduces the current (uniform) numbers.

Caveat worth stating: part of the measured "drift" is chart-mix drift rather than skill
drift (players attempt harder charts as they improve). The response profile conditions on
the target chart's axis values, but not on everything, so recency can help through either
channel. This sweep measures the total effect, not its decomposition.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chart_repr import (HISTORY_FEATURES, HISTORY_RESPONSE_BP_COLS,
                        HISTORY_RESPONSE_COLS, HISTORY_RESPONSE_LAMP_COLS,
                        OBJECTIVE_STAT_COLS, RESPONSE_AXES, feature_manifest)
from common import load_samples
from response_eval import evaluate
from response_features import axis_sd, build_table

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"

HALF_LIVES = [None, 720.0, 365.0, 180.0, 90.0, 30.0]
MIN_HISTORY = 20

LAMP_RESP = [c for c in HISTORY_RESPONSE_LAMP_COLS
             if c.startswith("l_resp_") and c not in ("l_resp_mean", "l_resp_std")]
BP_RESP = [c for c in HISTORY_RESPONSE_BP_COLS
           if c.startswith("b_resp_") and c not in ("b_resp_mean", "b_resp_std")]
CUR = HISTORY_RESPONSE_COLS + LAMP_RESP + ["l_resp_mean", "l_resp_std"]
V1, Hv = OBJECTIVE_STAT_COLS, HISTORY_FEATURES

SETS = {
    "B": V1 + Hv,
    "B_full": V1 + Hv + HISTORY_RESPONSE_COLS,
    # the current best configuration
    "B_resp+bpresp": (V1 + Hv + CUR + BP_RESP + ["b_resp_mean", "b_resp_std"]),
}


def build_response(half_life, return_sd: bool = False):
    """Response blocks for every first-play row.

    `return_sd=True` also hands back the global axis std dict, so a caller that needs to
    score NEW charts (the recommender) can reuse the exact scaling the training rows saw
    instead of recomputing it on a different frame and silently shifting the slopes.
    """
    fp = pd.read_parquet(DS / "firstplays.parquet")
    v2 = pd.read_parquet(DS / "chart_stats_v2.parquet")
    need = [c for c in set(RESPONSE_AXES.values())
            if c not in fp.columns and c in v2.columns]
    if need:
        fp = fp.merge(v2[["sha256"] + need], on="sha256", how="left")
    fp = fp.sort_values(["player", "time"], kind="stable").reset_index(drop=True)
    sd = axis_sd(fp)
    fp["log1p_bp"] = np.log1p(fp["bp"].values.astype(np.float64))
    blocks = [
        build_table(fp, sd, min_n=MIN_HISTORY, target="acc", prefix="h_",
                    clip=(0.0, 100.0), half_life=half_life),
        build_table(fp, sd, min_n=MIN_HISTORY, target="lamp", prefix="l_",
                    clip=(1.0, 9.0), half_life=half_life),
        build_table(fp, sd, min_n=MIN_HISTORY, target="log1p_bp", prefix="b_",
                    clip=(0.0, 12.0), half_life=half_life),
    ]
    feats = pd.concat(blocks, axis=1)
    out = pd.concat([fp[["player", "sha256"]], feats], axis=1)
    return (out, sd) if return_sd else out


def main() -> None:
    df = load_samples()
    results: dict = {"half_lives": [str(h) for h in HALF_LIVES], "min_history": MIN_HISTORY}

    for H in HALF_LIVES:
        tag = "uniform" if H is None else f"H={int(H)}d"
        hr = build_response(H)
        assert not hr.duplicated(["player", "sha256"]).any(), "response key not unique"
        d = df.merge(hr, on=["player", "sha256"], how="left")
        cov = float(d["h_resp_mean"].notna().mean())
        tr, te = d[d["phase"] == "train"], d[d["phase"] == "test"]
        results[tag] = {"coverage": round(cov, 4)}
        print(f"\n=== {tag}  (h_resp coverage {cov:.3f}) ===")
        print(f"{'set':18}{'acc':>8}{'cR2':>9}{'lamp':>8}{'QWK':>8}{'BP':>9}")
        for stag, feats in SETS.items():
            if stag != "B" or H is None:      # B does not depend on the half-life
                r = evaluate(tr, te, feats)
                results[tag][stag] = r
                print(f"{stag:18}{r['acc']['mae']:8.3f}{r['acc']['centered_r2']:+9.3f}"
                      f"{r['lamp']['ord_mae']:8.3f}{r['lamp']['qwk']:8.3f}"
                      f"{r['bp']['raw_mae']:9.1f}")
        # slope magnitude: a decayed fit should show stronger, less diluted slopes
        results[tag]["slope_abs_mean"] = {
            c: round(float(np.nanmean(np.abs(d[c]))), 3)
            for c in ("h_slope_nps", "h_slope_ln", "h_slope_jrank")}

    print("\nacc MAE vs half-life (B_resp+bpresp):")
    for tag in results:
        if isinstance(results[tag], dict) and "B_resp+bpresp" in results[tag]:
            print(f"  {tag:10}{results[tag]['B_resp+bpresp']['acc']['mae']:8.3f}"
                  f"{results[tag]['B_resp+bpresp']['acc']['centered_r2']:+9.3f}")

    # ---- is the best half-life's gain bigger than the seed noise? ----------------
    # PROTOCOL.md: a single seed is not a conclusion. HGB is only weakly stochastic
    # (bin subsampling), but the project records seed_std for exactly this reason, so
    # report mean +/- sd over 3 seeds for the winner against the uniform reference.
    best = min((t for t in results if isinstance(results[t], dict)
                and "B_resp+bpresp" in results[t] and t != "uniform"),
               key=lambda t: results[t]["B_resp+bpresp"]["acc"]["mae"])
    results["best_half_life"] = best
    results["_best_H"] = (None if best == "uniform"
                          else float(best.split("=")[1].rstrip("d")))
    print(f"\nseed spread on B_resp+bpresp (best = {best}):")
    for tag, H in (("uniform", None), (best, None if best == "uniform"
                                       else float(best.split("=")[1].rstrip("d")))):
        d = df.merge(build_response(H), on=["player", "sha256"], how="left")
        tr, te = d[d["phase"] == "train"], d[d["phase"] == "test"]
        ms, cs = [], []
        for s in (0, 1, 2):
            r = evaluate(tr, te, SETS["B_resp+bpresp"], seed=s)
            ms.append(r["acc"]["mae"])
            cs.append(r["acc"]["centered_r2"])
        results[f"{tag}_3seed"] = {"acc_mae_mean": round(float(np.mean(ms)), 3),
                                   "acc_mae_sd": round(float(np.std(ms)), 4),
                                   "per_seed": ms,
                                   "cr2_mean": round(float(np.mean(cs)), 4)}
        print(f"  {tag:10}acc {np.mean(ms):.3f} +/- {np.std(ms):.3f}   per-seed {ms}"
              f"   cR2 {np.mean(cs):+.3f}")

    # HGB turned out deterministic here (sd 0.000), so the remaining uncertainty is
    # SAMPLING: which test rows. Pair the per-row absolute errors and bootstrap them,
    # otherwise a 0.1 MAE difference has no error bar attached to it.
    from scipy import stats as _st
    from common import hgb_fit_predict
    errs = {}
    for tag, H in (("uniform", None), (best, results["_best_H"])):
        d = df.merge(build_response(H), on=["player", "sha256"], how="left")
        tr, te = d[d["phase"] == "train"], d[d["phase"] == "test"]
        pr = hgb_fit_predict(tr, te, SETS["B_resp+bpresp"], tr["acc"].values)
        errs[tag] = np.abs(te["acc"].values - pr)
    diff = errs["uniform"] - errs[best]                 # >0 means the decay helped
    tt = _st.ttest_1samp(diff, 0.0)
    rng = np.random.RandomState(0)
    boot = diff[rng.randint(0, len(diff), size=(2000, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    results["paired_vs_uniform"] = {
        "n_test": int(len(diff)),
        "mean_abs_err_gain": round(float(diff.mean()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "t": round(float(tt.statistic), 3), "p": float(f"{tt.pvalue:.3g}"),
        "rows_helped_frac": round(float((diff > 0).mean()), 4)}
    print(f"\npaired on {len(diff)} test rows: mean |err| gain {diff.mean():+.4f} "
          f"(95% CI {lo:+.4f}..{hi:+.4f}), t={tt.statistic:.2f} p={tt.pvalue:.3g}, "
          f"rows helped {(diff > 0).mean():.1%}")

    results["features_used"] = feature_manifest(*SETS.values())
    results.pop("_best_H", None)          # internal, not part of the record
    json.dump(results, open(OUT / "response_decay.json", "w"), indent=2, default=str)
    print("saved ->", OUT / "response_decay.json")


if __name__ == "__main__":
    main()
