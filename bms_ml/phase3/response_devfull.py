"""Phase 3.6: extend the player-relative deviation from the 11 response axes to the
remaining 20 objective chart statistics.

dev on the 11 axes gave the largest single gain so far (+0.243). The other 20 chart
statistics (BPM and its variation, stops, lane distribution, trill, ...) have never been
seen through the player's own scale. Same construction: for row i and column c,
pdv_c = (x_ic - mean_{j<i, same player}(x_jc)) / sd_{j<i}(x_jc).

Implementation note: this does not need the response block's OLS machinery, only causal
per-player moments, so it uses plain prefix sums (optionally decayed with the same
half-life as the response block). Columns already covered by RESPONSE_AXES are excluded
so nothing is duplicated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chart_repr import OBJECTIVE_STAT_COLS, RESPONSE_AXES, feature_manifest
from common import load_samples
from response_decay import SETS as DECAY_SETS
from response_decay import V1, Hv
from response_eval import evaluate

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"
HALF_LIFE = 180.0
MIN_N = 20

AXIS_COLS = set(RESPONSE_AXES.values())
NEW_COLS = [c for c in OBJECTIVE_STAT_COLS if c not in AXIS_COLS]
DEV11 = [f"h_dev_{k}" for k in RESPONSE_AXES]
PDEV = [f"pdv_{c}" for c in NEW_COLS]
BASE = DECAY_SETS["B_resp+bpresp"] + DEV11


def _cum(a: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(a)])


def player_relative_dev(fp: pd.DataFrame, cols: list[str], min_n: int = MIN_N,
                        half_life: float | None = HALF_LIFE) -> pd.DataFrame:
    """Causal per-player z-score of each chart column, strictly-prior window."""
    out = {f"pdv_{c}": np.full(len(fp), np.nan) for c in cols}
    tv = None
    if half_life is not None:
        t = fp["time"].values
        if np.issubdtype(t.dtype, np.datetime64):
            t = t.astype("datetime64[s]").astype(np.float64)
        tv = np.asarray(t, dtype=np.float64) / 86400.0
    for _p, pos in fp.groupby("player", sort=False).indices.items():
        pos = np.asarray(pos)
        up = None
        if tv is not None:
            up = np.log(2.0) * (tv[pos] - tv[pos][0]) / half_life
        for c in cols:
            x = fp[c].values[pos].astype(np.float64)
            v = np.isfinite(x)
            # 0 * nan is nan, so sanitise before weighting (same trap as response_features)
            xs = np.where(v, x, 0.0)
            w = (np.exp(up) if up is not None else np.ones(len(pos))) * v
            # CW[j] = sum over prior rows 0..j-1 of THIS player (the prepended zero makes
            # the shift free); note the group-local index, not the global row position
            rescale = np.exp(-up) if up is not None else np.ones(len(pos))
            ne = _cum(w)[:-1] * rescale
            sx = _cum(w * xs)[:-1] * rescale
            sxx = _cum(w * xs * xs)[:-1] * rescale
            safe = np.maximum(ne, 1.0)
            mu = np.where(ne > 0, sx / safe, 0.0)
            var = np.where(ne > 0, np.maximum(sxx / safe - mu * mu, 0.0), 0.0)
            sdv = np.sqrt(var)
            d = np.where((ne >= min_n) & (sdv > 1e-9),
                         (xs - mu) / np.maximum(sdv, 1e-9), np.nan)
            out[f"pdv_{c}"][pos] = d
    return pd.DataFrame(out, index=fp.index)


def main() -> None:
    df = load_samples()
    fp = pd.read_parquet(DS / "firstplays.parquet")
    v2 = pd.read_parquet(DS / "chart_stats_v2.parquet")
    need = [c for c in AXIS_COLS if c not in fp.columns and c in v2.columns]
    if need:
        fp = fp.merge(v2[["sha256"] + need], on="sha256", how="left")
    fp = fp.sort_values(["player", "time"], kind="stable").reset_index(drop=True)
    pdev = player_relative_dev(fp, NEW_COLS)
    print(f"new dev columns: {len(PDEV)}  "
          f"coverage {float(pdev[PDEV[0]].notna().mean()):.3f}")

    # both side tables join on the first-play key; pdev is built on the sorted fp, so it
    # must carry the key explicitly (index alignment between fp and samples is NOT 1:1)
    from response_decay import build_response
    d = (df.merge(build_response(HALF_LIFE), on=["player", "sha256"], how="left")
           .merge(pdev.assign(_k=fp["player"].values, _s=fp["sha256"].values)
                  .rename(columns={"_k": "player", "_s": "sha256"}),
                  on=["player", "sha256"], how="left"))
    tr, te = d[d["phase"] == "train"], d[d["phase"] == "test"]

    sets = {
        "B_resp+bpresp+dev11": BASE,
        "B_resp+bpresp+dev11+dev20": BASE + PDEV,
        "B+dev20only": V1 + Hv + PDEV,
    }
    results: dict = {"half_life": HALF_LIFE, "new_cols": len(PDEV)}
    print(f"{'set':30}{'acc':>8}{'cR2':>9}{'lamp':>8}{'QWK':>8}{'BP':>9}")
    for tag, feats in sets.items():
        r = evaluate(tr, te, feats)
        results[tag] = r
        print(f"{tag:30}{r['acc']['mae']:8.3f}{r['acc']['centered_r2']:+9.3f}"
              f"{r['lamp']['ord_mae']:8.3f}{r['lamp']['qwk']:8.3f}"
              f"{r['bp']['raw_mae']:9.1f}")

    # capacity control on the 20 new columns
    rs = np.random.RandomState(0)
    sh = d.copy()
    for c in PDEV:
        sh[c] = rs.permutation(sh[c].values)
    r = evaluate(sh[sh["phase"] == "train"], sh[sh["phase"] == "test"],
                 sets["B_resp+bpresp+dev11+dev20"])
    results["dev20_shuffled"] = r
    print(f"{'  dev20 shuffled':30}{r['acc']['mae']:8.3f}"
          f"{r['acc']['centered_r2']:+9.3f}{r['lamp']['ord_mae']:8.3f}"
          f"{r['lamp']['qwk']:8.3f}{r['bp']['raw_mae']:9.1f}")

    best = min(sets, key=lambda t: results[t]["acc"]["mae"])
    if best != "B_resp+bpresp+dev11":
        from common import hgb_fit_predict
        from scipy import stats as _st
        errs = {}
        for tag in ("B_resp+bpresp+dev11", best):
            pr = hgb_fit_predict(tr, te, sets[tag], tr["acc"].values)
            errs[tag] = np.abs(te["acc"].values - pr)
        diff = errs["B_resp+bpresp+dev11"] - errs[best]
        tt = _st.ttest_1samp(diff, 0.0)
        rng = np.random.RandomState(0)
        boot = diff[rng.randint(0, len(diff), size=(2000, len(diff)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        results["paired"] = {"vs": "B_resp+bpresp+dev11",
                             "mean": round(float(diff.mean()), 4),
                             "ci95": [round(float(lo), 4), round(float(hi), 4)],
                             "t": round(float(tt.statistic), 3),
                             "p": float(f"{tt.pvalue:.3g}"),
                             "helped": round(float((diff > 0).mean()), 4)}
        print(f"\npaired vs dev11: +{diff.mean():.4f} (95% CI {lo:.4f}..{hi:.4f}) "
              f"t={tt.statistic:.2f} p={tt.pvalue:.3g} helped {(diff > 0).mean():.1%}")

    results["features_used"] = feature_manifest(*sets.values())
    json.dump(results, open(OUT / "response_devfull.json", "w"), indent=2, default=str)
    print("saved ->", OUT / "response_devfull.json")


if __name__ == "__main__":
    main()
