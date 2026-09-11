"""Phase 3.6: player-relative axis deviation — does "how unusual is this chart FOR ME" help?

`resp_<axis>` is the player's LINEAR prediction at the target chart's axis value, so the
response profile can only express a straight line. `dev_<axis>` = (x_target -
mean_player_history) / sd_player_history says how far outside the player's own usual
range the chart sits. A tree splitting on it places thresholds in player-relative units,
which is the mechanism for a saturating / cliff response the linear profile cannot
represent. It reuses the same window statistics as the OLS, so it is nearly free.

Baseline is the current best: B_resp+bpresp with the 180-day half-life (response_decay).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chart_repr import HISTORY_RESPONSE_DEV_COLS, feature_manifest
from response_decay import CUR, BP_RESP, SETS, V1, Hv, build_response
from response_eval import evaluate
from common import load_samples

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DEV = HISTORY_RESPONSE_DEV_COLS

BASE = V1 + Hv + CUR + BP_RESP + ["b_resp_mean", "b_resp_std"]
SETS_DEV = {
    # no response block at all: is dev informative on its own?
    "B+dev": V1 + Hv + DEV,
    # the current best, unchanged
    "B_resp+bpresp": BASE,
    # ... plus the deviations
    "B_resp+bpresp+dev": BASE + DEV,
}


def shuffled_dev(df, seed: int):
    """Capacity control: same 11 dev columns, same marginals, pairing destroyed.

    The project has been burned before by gains that were just extra HGB width
    (response_eval's B+shuffled24 control), and this adds 11 columns to a 76-column
    model, so the pairing has to be shown to matter.
    """
    import pandas as pd
    rs = np.random.RandomState(seed)
    sh = df.copy()
    for c in DEV:
        sh[c] = rs.permutation(sh[c].values)
    return sh[sh["phase"] == "train"], sh[sh["phase"] == "test"]


def main() -> None:
    df = load_samples()
    results: dict = {"half_life": 180.0, "sets": list(SETS_DEV)}
    H = 180.0
    d = df.merge(build_response(H), on=["player", "sha256"], how="left")
    tr, te = d[d["phase"] == "train"], d[d["phase"] == "test"]
    cov = {c: round(float(d[c].notna().mean()), 3) for c in DEV[:3]}
    results["dev_coverage_sample"] = cov
    print(f"half-life {H}  dev coverage (h_dev_nps/ln/scratch): {cov}")
    print(f"{'set':28}{'acc':>8}{'cR2':>9}{'lamp':>8}{'QWK':>8}{'BP':>9}")
    for tag, feats in SETS_DEV.items():
        r = evaluate(tr, te, feats)
        results[tag] = r
        print(f"{tag:28}{r['acc']['mae']:8.3f}{r['acc']['centered_r2']:+9.3f}"
              f"{r['lamp']['ord_mae']:8.3f}{r['lamp']['qwk']:8.3f}"
              f"{r['bp']['raw_mae']:9.1f}")

    for s in (0, 1, 2):
        tr_s, te_s = shuffled_dev(d, 100 + s)
        r = evaluate(tr_s, te_s, SETS_DEV["B_resp+bpresp+dev"])
        results[f"dev_shuffled_s{s}"] = r
        print(f"{'  dev shuffled s' + str(s):28}{r['acc']['mae']:8.3f}"
              f"{r['acc']['centered_r2']:+9.3f}{r['lamp']['ord_mae']:8.3f}"
              f"{r['lamp']['qwk']:8.3f}{r['bp']['raw_mae']:9.1f}")

    # paired check for the headline comparison, so the delta carries an error bar
    if results["B_resp+bpresp+dev"]["acc"]["mae"] < results["B_resp+bpresp"]["acc"]["mae"]:
        from common import hgb_fit_predict
        from scipy import stats as _st
        errs = {}
        for tag in ("B_resp+bpresp", "B_resp+bpresp+dev"):
            pr = hgb_fit_predict(tr, te, SETS_DEV[tag], tr["acc"].values)
            errs[tag] = np.abs(te["acc"].values - pr)
        diff = errs["B_resp+bpresp"] - errs["B_resp+bpresp+dev"]
        tt = _st.ttest_1samp(diff, 0.0)
        rng = np.random.RandomState(0)
        boot = diff[rng.randint(0, len(diff), size=(2000, len(diff)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        results["paired"] = {"mean": round(float(diff.mean()), 4),
                             "ci95": [round(float(lo), 4), round(float(hi), 4)],
                             "t": round(float(tt.statistic), 3),
                             "p": float(f"{tt.pvalue:.3g}"),
                             "helped": round(float((diff > 0).mean()), 4)}
        print(f"\npaired: +{diff.mean():.4f} (95% CI {lo:.4f}..{hi:.4f}) "
              f"t={tt.statistic:.2f} p={tt.pvalue:.3g} helped {(diff > 0).mean():.1%}")

    results["features_used"] = feature_manifest(*SETS_DEV.values())
    json.dump(results, open(OUT / "response_dev.json", "w"), indent=2, default=str)
    print("saved ->", OUT / "response_dev.json")


if __name__ == "__main__":
    main()
