"""Phase 3.6: MULTIVARIATE ridge response vs the univariate per-axis fit.

The shipped response block fits 11 INDEPENDENT OLS y ~ x_axis. Wherever the axes
correlate (density, NPS and chord rates move together), each univariate slope absorbs
the others' effects - omitted-variable bias. The multivariate estimator
(response_columns_mv) fits ONE joint ridge over the same causal window and emits the
same schema, so the block is a drop-in swap: identical feature names, identical windows,
identical decay - only the estimator differs.

Configs on the fused BEST_FEATURES (five-table fence, 180d half-life):
  B_full          shipped univariate acc block (reference)
  B_full_mv{lam}  univariate acc block REPLACED by the joint-ridge block (lambda sweep)
Lamp/BP blocks stay univariate in both arms, so the comparison isolates the acc
estimator. Paired test on the winner.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chart_repr import (BEST_FEATURES, HISTORY_FEATURES, MSD_AXES,
                        HISTORY_RESPONSE_BP_COLS, OBJECTIVE_STAT_COLS,
                        RESPONSE_AXES, feature_manifest)
from common import load_samples
from response_eval import evaluate
from response_features import axis_sd, build_table

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "bms_ml" / "output" / "phase3" / "dataset"
HALF_LIFE = 180.0
MIN_HISTORY = 20

LAMP_RESP = [f"l_resp_{k}" for k in RESPONSE_AXES]
BP_RESP = [f"b_resp_{k}" for k in RESPONSE_AXES]
DEV11 = [f"h_dev_{k}" for k in RESPONSE_AXES]


def mv_acc_block(ridge: float) -> tuple[pd.DataFrame, list[str]]:
    """Joint-ridge acc response block under the hv_ prefix (avoids colliding with the
    shipped univariate h_ block; the eval feature list renames at the end)."""
    fp = pd.read_parquet(DS / "firstplays.parquet")
    v2 = pd.read_parquet(DS / "chart_stats_v2.parquet")
    need = [c for c in set(RESPONSE_AXES.values()) if c not in fp.columns]
    if need:
        fp = fp.merge(v2[["sha256"] + need], on="sha256", how="left")
    if (DS / "msd.parquet").exists():
        fp = fp.merge(pd.read_parquet(DS / "msd.parquet"), on="sha256", how="left")
    fp = fp.sort_values(["player", "time"], kind="stable").reset_index(drop=True)
    sd = axis_sd(fp)
    blk = build_table(fp, sd, min_n=MIN_HISTORY, target="acc", prefix="hv_",
                      clip=(0.0, 100.0), half_life=HALF_LIFE,
                      multivariate=True, ridge=ridge)
    cols = ([f"hv_resp_{k}" for k in RESPONSE_AXES] + [f"hv_slope_{k}" for k in RESPONSE_AXES]
            + ["hv_resp_mean", "hv_resp_std"] + [f"hv_dev_{k}" for k in RESPONSE_AXES])
    return pd.concat([fp[["player", "sha256"]], blk[cols]], axis=1), cols


def rename(feats: list[str]) -> list[str]:
    return [c.replace("h_", "hv_") if c.startswith(("h_resp_", "h_slope_", "h_dev_"))
            else c for c in feats]


def main() -> None:
    hr = pd.read_parquet(DS / "history_response.parquet")
    static = OBJECTIVE_STAT_COLS + HISTORY_FEATURES
    need = [c for c in BEST_FEATURES if c not in static]
    df = (load_samples()
          .merge(hr[["player", "sha256"] + [c for c in need if c in hr.columns]],
                 on=["player", "sha256"], how="left"))
    tr, te = df[df["phase"] == "train"], df[df["phase"] == "test"]

    results: dict = {"half_life": HALF_LIFE}
    sets = {"B_full": BEST_FEATURES}
    print(f"{'set':16}{'acc':>8}{'cR2':>9}{'lamp':>8}{'QWK':>8}{'BP':>9}")
    r = evaluate(tr, te, BEST_FEATURES)
    results["B_full"] = r
    print(f"{'B_full':16}{r['acc']['mae']:8.3f}{r['acc']['centered_r2']:+9.3f}"
          f"{r['lamp']['ord_mae']:8.3f}{r['lamp']['qwk']:8.3f}{r['bp']['raw_mae']:9.1f}")

    for lam in (0.05, 0.2, 0.5):
        blk, _ = mv_acc_block(lam)
        d = df.merge(blk, on=["player", "sha256"], how="left")
        feats = rename(BEST_FEATURES)
        assert not set(feats) - set(d.columns), f"missing {set(feats) - set(d.columns)}"
        tr2, te2 = d[d["phase"] == "train"], d[d["phase"] == "test"]
        r = evaluate(tr2, te2, feats)
        tag = f"B_full_mv{lam}"
        sets[tag] = feats
        results[tag] = r
        print(f"{tag:16}{r['acc']['mae']:8.3f}{r['acc']['centered_r2']:+9.3f}"
              f"{r['lamp']['ord_mae']:8.3f}{r['lamp']['qwk']:8.3f}{r['bp']['raw_mae']:9.1f}")

    better = [t for t in sets if t != "B_full"
              and results[t]["acc"]["mae"] < results["B_full"]["acc"]["mae"]]
    if better:
        from common import hgb_fit_predict
        from scipy import stats as _st
        best = min(better, key=lambda t: results[t]["acc"]["mae"])
        lam = float(best.replace("B_full_mv", ""))
        blk, _ = mv_acc_block(lam)
        d = df.merge(blk, on=["player", "sha256"], how="left")
        errs = {}
        for tag, frame, a, b in (("B_full", df, tr, te),
                                 (best, d, d[d.phase == "train"], d[d.phase == "test"])):
            pr = hgb_fit_predict(a, b, sets[tag], a["acc"].values)
            errs[tag] = np.abs(b["acc"].values - pr)
        diff = errs["B_full"] - errs[best]
        tt = _st.ttest_1samp(diff, 0.0)
        rng = np.random.RandomState(0)
        boot = diff[rng.randint(0, len(diff), size=(2000, len(diff)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        results["paired"] = {"best": best, "mean": round(float(diff.mean()), 4),
                             "ci95": [round(float(lo), 4), round(float(hi), 4)],
                             "t": round(float(tt.statistic), 3),
                             "p": float(f"{tt.pvalue:.3g}"),
                             "helped": round(float((diff > 0).mean()), 4)}
        print(f"\npaired {best} vs B_full: +{diff.mean():.4f} "
              f"(95% CI {lo:.4f}..{hi:.4f}) t={tt.statistic:.2f} p={tt.pvalue:.3g} "
              f"helped {(diff > 0).mean():.1%}")

    results["features_used"] = feature_manifest(*sets.values())
    json.dump(results, open(ROOT / "bms_ml" / "output" / "phase3"
                            / "response_ridge.json", "w"), indent=2, default=str)
    print("saved -> response_ridge.json")


if __name__ == "__main__":
    main()
