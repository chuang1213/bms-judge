"""Phase 3.6: the shared per-axis personal response estimator.

Single source of truth for `h_resp_*` / `h_slope_*` (see chart_repr.RESPONSE_AXES).
Two callers with deliberately different windows share this math:

  history_response.py  window=None            full causal history (protocol baseline)
  transfer_eval.py     window=RESP_WIN (+rng) last N events, random truncation when
                                              training (few-shot must match the eval
                                              estimator, see below)

The estimator: for row i of a player's chronological frame, take the LAST n valid
(acc-known) events strictly before i, fit acc ~ alpha + beta * axis, and emit

  h_resp_<axis>  = clip(alpha + beta * x_i, 0, 100)   chart-conditioned, x_i is known
  h_slope_<axis> = beta * sd_axis                     pure player trait, scale-free

Why `window` and `rng` exist (this is the whole reason the module is shared)
--------------------------------------------------------------------------
A few-shot row at k has only k prefix events, while a protocol training row has its
entire archive. Fitting the OLS on "everything available" therefore uses a DIFFERENT
estimator on the two sides — the same train/eval incomparability that bit the
recency features in PHASE3_4_TRANSFER (they land outside the training p99 and once
faked a collapse of the M2 curve). Fix: cap both sides at the last `window` events
and, when training, draw n ~ Uniform{min_n..min(window, i)} per row so the model
sees every fill level (the deployment-style truncation augmentation used in
c0_fewshot.build_seqs). Evaluation uses n = min(window, k), no augmentation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from chart_repr import HISTORY_RESPONSE_COLS, RESPONSE_AXES


def _cum(a: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(a)])


def response_columns(X: np.ndarray, y: np.ndarray, sd: dict, min_n: int = 20,
                     window: int | None = None, rng: np.random.RandomState | None = None,
                     shrink: float = 0.0) -> dict:
    """X: (m, n_axes) axis values, y: (m,) acc, sd: axis -> global std (for the slope).

    `window=None` -> all prior events; `rng` given -> random truncation (training).
    `shrink` = lambda: the slope is shrunk toward 0 by ne/(ne+lambda), i.e. a zero-slope
    prior worth `lambda` events, and the intercept is re-anchored on the window means.
    Needed in few-shot: at k=5 an unshrunk OLS slope is pure noise, yet the truncation
    augmentation teaches the model to trust it (measured: k=5 got WORSE than k=1)."""
    names = list(RESPONSE_AXES)
    m = len(y)
    out = {c: np.full(m, np.nan) for c in HISTORY_RESPONSE_COLS}
    yv = ~np.isnan(y)
    idx = np.arange(m)
    cap = idx if window is None else np.minimum(idx, window)
    if rng is None:
        n = cap.copy()
    else:
        # n ~ U{min_n..cap} where cap >= min_n, else n = cap (cannot augment a short history)
        lo = np.minimum(min_n, cap)
        n = np.clip(lo + (rng.random_sample(m) * (cap - lo + 1)).astype(np.int64),
                    lo, cap)
    hi_i = idx
    lo_i = np.clip(idx - n, 0, None)
    resp = np.full((len(names), m), np.nan)
    for a, name in enumerate(names):
        x = X[:, a]
        v = yv & ~np.isnan(x)
        xv, yv2 = np.where(v, x, 0.0), np.where(v, y, 0.0)
        P = np.stack([_cum(v.astype(float)), _cum(xv), _cum(yv2),
                      _cum(xv * yv2), _cum(xv * xv)], axis=1)      # (m+1, 5)
        S = P[hi_i] - P[lo_i]                                      # sums over the window
        ne, sx, sy, sxy, sxx = (S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4])
        denom = ne * sxx - sx * sx
        ok = (ne >= min_n) & (denom > 1e-9)
        beta = np.full(m, np.nan)
        alpha = np.full(m, np.nan)
        raw = ((ne * sxy - sx * sy) / denom)[ok]
        beta[ok] = raw * (ne[ok] / (ne[ok] + shrink))
        alpha[ok] = ((sy - beta * sx) / ne)[ok]
        r = np.clip(alpha + beta * x, 0.0, 100.0)
        resp[a] = r
        out[f"h_resp_{name}"] = r
        out[f"h_slope_{name}"] = beta * sd[name]
    out["h_resp_mean"] = np.nanmean(resp, axis=0)
    with np.errstate(invalid="ignore"):
        out["h_resp_std"] = np.nanstd(resp, axis=0)
    return out


def build_table(fp_chronological: pd.DataFrame, sd: dict, min_n: int = 20,
                window: int | None = None, rng: np.random.RandomState | None = None,
                shrink: float = 0.0) -> pd.DataFrame:
    """Per-player application of `response_columns`. `fp_chronological` must be sorted
    by (player, time) — the causal prefix sums depend on it."""
    names = list(RESPONSE_AXES)
    cols = {c: np.full(len(fp_chronological), np.nan) for c in HISTORY_RESPONSE_COLS}
    for _p, pos in fp_chronological.groupby("player", sort=False).indices.items():
        pos = np.asarray(pos)
        y = fp_chronological["acc"].values[pos].astype(np.float64)
        X = np.stack([fp_chronological[RESPONSE_AXES[n]].values[pos].astype(np.float64)
                      for n in names], axis=1)
        r = response_columns(X, y, sd, min_n=min_n, window=window, rng=rng,
                             shrink=shrink)
        for c in HISTORY_RESPONSE_COLS:
            cols[c][pos] = r[c]
    return pd.DataFrame(cols)


def prefix_response(prefix: pd.DataFrame, targets: pd.DataFrame, sd: dict,
                    min_n: int = 5, window: int = 50,
                    shrink: float = 0.0) -> pd.DataFrame:
    """Response features for `targets`, fitted on `prefix` ONLY.

    This is NOT build_table(prefix + targets): that frame lets an early target become
    the history of a later one, so the curve is fitted on outcomes we are supposed to
    be predicting (it showed up as a spurious gain even at k=0, where the prefix is
    empty and every response feature must be NaN). The few-shot protocol says the
    history input is the prefix and nothing else, so the OLS is fitted once on the
    last `min(window, len(prefix))` prefix events and then evaluated at each target's
    own axis value.
    """
    names = list(RESPONSE_AXES)
    m = len(targets)
    n = min(window, len(prefix))
    pre = prefix.iloc[len(prefix) - n:] if n else prefix.iloc[:0]
    y = pre["acc"].values.astype(np.float64) if n else np.zeros(0)
    out = {}
    resp = np.full((len(names), m), np.nan)
    for a, name in enumerate(names):
        col = RESPONSE_AXES[name]
        xt = targets[col].values.astype(np.float64)
        r = np.full(m, np.nan)
        sl = np.full(m, np.nan)
        if n:
            x = pre[col].values.astype(np.float64)
            v = ~np.isnan(y) & ~np.isnan(x)
            ne = int(v.sum())
            if ne >= min_n:
                xv, yv = x[v], y[v]
                sx, sy = xv.sum(), yv.sum()
                sxy, sxx = float((xv * yv).sum()), float((xv * xv).sum())
                den = ne * sxx - sx * sx
                if den > 1e-9:
                    beta = ((ne * sxy - sx * sy) / den) * (ne / (ne + shrink))
                    alpha = (sy - beta * sx) / ne
                    r = np.clip(alpha + beta * xt, 0.0, 100.0)
                    sl = np.full(m, beta * sd[name])
        out[f"h_resp_{name}"] = r
        out[f"h_slope_{name}"] = sl
        resp[a] = r
    out["h_resp_mean"] = np.nanmean(resp, axis=0)
    with np.errstate(invalid="ignore"):
        out["h_resp_std"] = np.nanstd(resp, axis=0)
    return pd.DataFrame(out, index=targets.index)[HISTORY_RESPONSE_COLS]


def axis_sd(fp: pd.DataFrame) -> dict:
    return {n: float(np.nanstd(fp[col].values.astype(np.float64)))
            for n, col in RESPONSE_AXES.items()}
