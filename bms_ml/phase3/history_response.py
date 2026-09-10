"""Phase 3.6: per-axis PERSONAL RESPONSE PROFILE (history side).

Why this exists
---------------
The 12 hand-crafted history features contain exactly ONE chart-conditioned player
feature: `h_knn_acc` (mean acc over the 20 nearest played charts in the 27-dim stat
space). PHASE3_5_REVIEW §4.3 flags this as the most valuable thing to upgrade: the
player side is otherwise blind to *how* a player's performance depends on chart
properties, and `h_knn_acc` is a local average in a 27-dim space — it needs a lot
of history to be stable and it cannot express direction ("this player collapses on
density but is fine on scratch").

This module gives each player an explicit, per-axis RESPONSE CURVE fitted on their
own strictly-prior plays (exactly the same causal window as data.py's history):

    for axis a in {nps, ln, scratch, dur, chord, jack}:
        acc ~ alpha + beta * a        (OLS on that player's history up to this target)
        h_resp_<a>   = clip(alpha + beta * a_target, 0, 100)
        h_slope_<a>  = beta_hat * sd(a over all charts)     (acc points per 1 SD of a)

`h_resp_*` is chart-conditioned (uses the TARGET's own axis value, which is known);
`h_slope_*` is a pure player trait — the kind of individually identifiable signal
the transfer study said is scarce (`PHASE3_5_REVIEW §3.4`: own-data marginal value
~0.29 acc MAE), which is exactly why a smooth 1-parameter-per-axis fit is a better
estimator than a 27-dim kNN mean. `h_resp_mean` / `h_resp_std` summarise the profile.

Leakage discipline: only plays strictly BEFORE the target's own first play enter the
fit (prefix cumulative sums, shifted by one row). Feature values are therefore
identical to what data.py's build_history_features would produce.

Output: bms_ml/output/phase3/dataset/history_response.parquet, keyed (player, sha256).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from chart_repr import HISTORY_RESPONSE_COLS, RESPONSE_AXES

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "bms_ml" / "output" / "phase3" / "dataset"

MIN_HISTORY = 20      # below this an OLS slope is noise; feature stays NaN (imputed)


def _prefix(a: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(a)])


def build(fp: pd.DataFrame) -> pd.DataFrame:
    """fp: firstplays.parquet, must contain player / sha256 / time / acc / <axes>."""
    n = len(fp)
    out = {c: np.full(n, np.nan) for c in HISTORY_RESPONSE_COLS}
    axes = list(RESPONSE_AXES.items())
    y_all = fp["acc"].values.astype(np.float64)
    sd = {name: float(np.nanstd(fp[col].values.astype(np.float64))) for name, col in axes}

    for _player, idx in fp.groupby("player", sort=False).indices.items():
        pos = np.asarray(idx)
        y = y_all[pos]
        k = ~np.isnan(y)
        resp_cols = []
        for name, col in axes:
            x = fp[col].values[pos].astype(np.float64)
            kn = k & ~np.isnan(x)
            xk, yk = np.where(kn, x, 0.0), np.where(kn, y, 0.0)
            # cumulative sums shifted by one row => strictly-prior history
            cn, sx = _prefix(kn.astype(float)), _prefix(xk)
            sy, sxy, sxx = _prefix(yk), _prefix(xk * yk), _prefix(xk * xk)
            nn = cn[:-1]
            denom = nn * sxx[:-1] - sx[:-1] ** 2
            ok = (nn >= MIN_HISTORY) & (denom > 1e-9)
            beta = np.full(len(pos), np.nan)
            beta[ok] = ((nn * sxy[:-1] - sx[:-1] * sy[:-1]) / denom)[ok]
            alpha = np.full(len(pos), np.nan)
            alpha[ok] = ((sy[:-1] - beta * sx[:-1]) / nn)[ok]
            pred = np.clip(alpha + beta * x, 0.0, 100.0)
            out[f"h_resp_{name}"][pos] = pred
            out[f"h_slope_{name}"][pos] = beta * sd[name]
            resp_cols.append(pred)
        R = np.vstack(resp_cols)                       # (n_axes, n_rows_of_player)
        out["h_resp_mean"][pos] = np.nanmean(R, axis=0)
        out["h_resp_std"][pos] = np.nanstd(R, axis=0)
    return pd.DataFrame(out)


def main() -> None:
    fp = pd.read_parquet(DS / "firstplays.parquet")
    # v2 axes live in a side table keyed 1:1 on sha256 (built by chart_stats_v2.py)
    v2 = pd.read_parquet(DS / "chart_stats_v2.parquet")
    need = [c for c in set(RESPONSE_AXES.values()) if c not in fp.columns]
    if need:
        fp = fp.merge(v2[["sha256"] + [c for c in need if c in v2.columns]],
                      on="sha256", how="left")
    # chronological within player is required for the causal prefix sums
    fp = fp.sort_values(["player", "time"], kind="stable").reset_index(drop=True)
    feats = build(fp)
    out = pd.concat([fp[["player", "sha256", "phase", "time"]], feats], axis=1)
    out.to_parquet(DS / "history_response.parquet")
    cov = feats["h_resp_mean"].notna().mean()
    print(f"rows {len(out)} | h_resp_mean coverage {cov:.3f} -> {DS / 'history_response.parquet'}")
    print(feats[HISTORY_RESPONSE_COLS].describe().loc[["mean", "std", "min", "max"]]
          .round(3).T.to_string())


if __name__ == "__main__":
    main()
