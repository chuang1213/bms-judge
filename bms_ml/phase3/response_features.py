"""Phase 3.6: the shared per-axis personal response estimator.

Single source of truth for the response blocks (see chart_repr.RESPONSE_AXES and
`_response_cols`). Three callers share this math:

  history_response.py  window=None            full causal history (protocol baseline)
  transfer_eval.py     window=RESP_WIN (+rng) last N events, random truncation when
                                              training (few-shot must match the eval
                                              estimator, see below)

The estimator: for row i of a player's chronological frame, take the LAST n valid
(target-known) events strictly before i, fit target ~ alpha + beta * axis, and emit

  <p>resp_<axis>  = clip(alpha + beta * x_i, clip)   chart-conditioned, x_i is known
  <p>slope_<axis> = beta * sd_axis                   pure player trait, scale-free

A block is parameterised by (target column, name prefix): `h_` scores acc, `l_` scores
lamp. They are different objects - a player's per-axis accuracy profile is not their
per-axis survival profile - and lamp was the one target the acc block slightly hurt.

Why `window` and `rng` exist (this is the whole reason the module is shared)
--------------------------------------------------------------------------
A few-shot row at k has only k prefix events, while a protocol training row has its
entire archive. Fitting the OLS on "everything available" therefore uses a DIFFERENT
estimator on the two sides - the same train/eval incomparability that bit the recency
features in PHASE3_4_TRANSFER. Fix: cap both sides at the last `window` events and,
when training, draw n ~ Uniform{min_n..min(window, i)} per row so the model sees every
fill level (the deployment-style truncation augmentation of c0_fewshot.build_seqs).
Evaluation uses n = min(window, k), no augmentation.

`shrink` = lambda is a zero-slope prior worth `lambda` events, applied to beta with the
intercept re-anchored on the window means. Needed in few-shot: at k=5 an unshrunk slope
is pure noise, yet the augmentation teaches the model to trust it (measured: k=5 got
worse than k=1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:                                    # run as a script: bms_ml/phase3 is on sys.path
    from chart_repr import RESPONSE_AXES, _response_cols
except ModuleNotFoundError:             # imported as bms_ml.phase3.response_features
    from .chart_repr import RESPONSE_AXES, _response_cols


def _cum(a: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(a)])


def response_columns(X: np.ndarray, y: np.ndarray, sd: dict, min_n: int = 20,
                     window: int | None = None, rng: np.random.RandomState | None = None,
                     shrink: float = 0.0, prefix: str = "h_",
                     clip: tuple[float, float] = (0.0, 100.0),
                     half_life: float | None = None,
                     t_days: np.ndarray | None = None) -> dict:
    """X: (m, n_axes) axis values, y: (m,) target, sd: axis -> global std (for slope).

    `window=None` -> all prior events; `rng` given -> random truncation (training).

    `half_life` (days) makes the fit RECENCY-WEIGHTED: when fitting row i, event j gets
    weight 0.5 ** ((t_i - t_j) / half_life). None = the original uniform fit over the
    whole archive. Motivation: archives span a median of 959 days (max 2000) and
    within-player acc drifts by +8.4pp on average (sd 11.8, >5pp on 13/18 players), yet
    the fit was uniform - so the profile described the player's LIFETIME average response
    while every target sits in their most recent quartile. `t_days` gives each row's time
    in days (any origin; only differences are used).
    """
    names = list(RESPONSE_AXES)
    m = len(y)
    out = {c: np.full(m, np.nan) for c in _response_cols(prefix)}
    yv = ~np.isnan(y)
    idx = np.arange(m)
    cap = idx if window is None else np.minimum(idx, window)
    if rng is None:
        n = cap.copy()
    else:
        lo = np.minimum(min_n, cap)
        n = np.clip(lo + (rng.random_sample(m) * (cap - lo + 1)).astype(np.int64), lo, cap)
    lo_i = np.clip(idx - n, 0, None)

    # Exponential forgetting: the weight of event j when fitting row i is
    # 0.5 ** ((t_i - t_j) / half_life), i.e. exp(u_j - u_i). It factorises into a
    # per-event factor exp(u_j) and a per-row factor exp(-u_i), so the windowed sums can
    # still be done with prefix sums. beta and alpha are invariant to per-row rescaling
    # (numerator and denominator both scale by the square of it), but keeping it makes
    # `ne` an EFFECTIVE sample size in O(1) units - which is what the min_n / shrink /
    # denom thresholds below assume. u is measured from the player's first event so
    # u >= 0. Note the direction: exp(+u) on the event, exp(-u) on the row. Getting it
    # backwards silently UP-WEIGHTS the distant past (a unit test catches it).
    if half_life is None:
        gw = np.ones(m)
        rescale = np.ones(m)
    else:
        if t_days is None:
            raise ValueError("half_life requires t_days")
        td = np.asarray(t_days, dtype=np.float64)
        if not np.isfinite(td).all():
            raise ValueError("t_days must be finite for a decayed fit")
        u = np.log(2.0) * (td - td[0]) / float(half_life)
        gw = np.exp(u)
        rescale = np.exp(-u)

    resp = np.full((len(names), m), np.nan)
    for a, name in enumerate(names):
        x = X[:, a]
        v = yv & ~np.isnan(x)
        w = gw * v
        # x and y must be NaN-free BEFORE multiplying by w: 0 * nan is nan, so a single
        # missing axis value would propagate through cumsum and blank every later row
        # (this made the 4 v2 axes 86% NaN until it was caught by an equivalence check
        # against the unweighted path).
        xv = np.where(np.isnan(x), 0.0, x)
        yv2 = np.where(np.isnan(y), 0.0, y)
        P = np.stack([_cum(w), _cum(w * xv), _cum(w * yv2),
                      _cum(w * xv * yv2), _cum(w * xv * xv)], axis=1)   # (m+1, 5)
        S = P[idx] - P[lo_i]                                            # sums over the window
        ne, sx, sy, sxy, sxx = (S[:, 0] * rescale, S[:, 1] * rescale, S[:, 2] * rescale,
                                S[:, 3] * rescale, S[:, 4] * rescale)
        denom = ne * sxx - sx * sx
        ok = (ne >= min_n) & (denom > 1e-9)
        beta = np.full(m, np.nan)
        alpha = np.full(m, np.nan)
        raw = ((ne * sxy - sx * sy) / denom)[ok]
        beta[ok] = raw * (ne[ok] / (ne[ok] + shrink))
        alpha[ok] = ((sy - beta * sx) / ne)[ok]
        r = np.clip(alpha + beta * x, clip[0], clip[1])
        resp[a] = r
        out[f"{prefix}resp_{name}"] = r
        out[f"{prefix}slope_{name}"] = beta * sd[name]
    out[f"{prefix}resp_mean"] = np.nanmean(resp, axis=0)
    with np.errstate(invalid="ignore"):
        out[f"{prefix}resp_std"] = np.nanstd(resp, axis=0)
    return out


def build_table(fp_chronological: pd.DataFrame, sd: dict, min_n: int = 20,
                window: int | None = None, rng: np.random.RandomState | None = None,
                shrink: float = 0.0, target: str = "acc", prefix: str = "h_",
                clip: tuple[float, float] = (0.0, 100.0),
                half_life: float | None = None) -> pd.DataFrame:
    """Per-player application of `response_columns`. `fp_chronological` must be sorted
    by (player, time) — the causal prefix sums depend on it. `half_life` (days) needs a
    `time` column and is forwarded to `response_columns`."""
    names = list(RESPONSE_AXES)
    cols = _response_cols(prefix)
    out = {c: np.full(len(fp_chronological), np.nan) for c in cols}
    tcol = None
    if half_life is not None:
        tv = fp_chronological["time"].values
        if np.issubdtype(tv.dtype, np.datetime64):
            tv = tv.astype("datetime64[s]").astype(np.float64)
        tcol = np.asarray(tv, dtype=np.float64) / 86400.0
    for _p, pos in fp_chronological.groupby("player", sort=False).indices.items():
        pos = np.asarray(pos)
        y = fp_chronological[target].values[pos].astype(np.float64)
        X = np.stack([fp_chronological[RESPONSE_AXES[n]].values[pos].astype(np.float64)
                      for n in names], axis=1)
        r = response_columns(X, y, sd, min_n=min_n, window=window, rng=rng,
                             shrink=shrink, prefix=prefix, clip=clip,
                             half_life=half_life,
                             t_days=None if tcol is None else tcol[pos])
        for c in cols:
            out[c][pos] = r[c]
    return pd.DataFrame(out)


def prefix_response(prefix_df: pd.DataFrame, targets: pd.DataFrame, sd: dict,
                    min_n: int = 5, window: int = 50, shrink: float = 0.0,
                    target: str = "acc", prefix: str = "h_",
                    clip: tuple[float, float] = (0.0, 100.0)) -> pd.DataFrame:
    """Response features for `targets`, fitted on `prefix_df` ONLY.

    This is NOT build_table(prefix + targets): that frame lets an early target become
    the history of a later one, so the curve is fitted on outcomes we are supposed to
    be predicting (it showed up as a spurious gain even at k=0, where the prefix is
    empty and every response feature must be NaN). The few-shot protocol says the
    history input is the prefix and nothing else, so the OLS is fitted once on the
    last `min(window, len(prefix))` prefix events and evaluated at each target's own
    axis value.
    """
    names = list(RESPONSE_AXES)
    m = len(targets)
    n = min(window, len(prefix_df))
    pre = prefix_df.iloc[len(prefix_df) - n:] if n else prefix_df.iloc[:0]
    y = pre[target].values.astype(np.float64) if n else np.zeros(0)
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
                    r = np.clip(alpha + beta * xt, clip[0], clip[1])
                    sl = np.full(m, beta * sd[name])
        out[f"{prefix}resp_{name}"] = r
        out[f"{prefix}slope_{name}"] = sl
        resp[a] = r
    out[f"{prefix}resp_mean"] = np.nanmean(resp, axis=0)
    with np.errstate(invalid="ignore"):
        out[f"{prefix}resp_std"] = np.nanstd(resp, axis=0)
    return pd.DataFrame(out, index=targets.index)[_response_cols(prefix)]


def axis_sd(fp: pd.DataFrame) -> dict:
    return {n: float(np.nanstd(fp[col].values.astype(np.float64)))
            for n, col in RESPONSE_AXES.items()}
