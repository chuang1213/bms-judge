"""Phase 4 metrics.

Absolute MAE is dominated by how strong each player is (measured spread across
players is ~4.8pp, far larger than any model-vs-baseline gap), so a model can
"win" on overall MAE by reproducing player averages better while learning nothing
about charts. PHASE4_PROTOCOL §5.2 therefore requires player-internal metrics
next to the absolute ones:

  player_centered_mae   each test row's actual AND prediction minus that player's
                        TRAIN mean, then MAE. This removes the player level
                        entirely: it asks "given how good this player is, can you
                        tell which chart they will do better or worse on?"
  player_rank_spearman  within one player, Spearman(model ranking, actual ranking)
  player_topk_ndcg      within one player, NDCG@K of the model's top-K

Both a global view and a per-player view are always produced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOP_K = (5, 10, 20)


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def rmse(y, p) -> float:
    d = np.asarray(y, float) - np.asarray(p, float)
    return float(np.sqrt(np.mean(d ** 2)))


def bias(y, p) -> float:
    return float(np.mean(np.asarray(p, float) - np.asarray(y, float)))


def spearman(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < 3 or np.std(p) == 0 or np.std(y) == 0:
        return None
    import warnings
    ry, rp = pd.Series(y).rank().to_numpy(), pd.Series(p).rank().to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.corrcoef(ry, rp)[0, 1])


def player_train_means(tr: pd.DataFrame, target: str) -> dict:
    """Per-player mean over TRAINING rows; missing player -> global train mean."""
    return tr.groupby("player")[target].mean().to_dict()


def centered(y, p, te: pd.DataFrame, train_means: dict, target: str):
    """Player deviations, and it is easy to get this WRONG.

    Returns (y - player_train_mean, p - player_train_mean): the actual and the
    predicted deviation from the same player-level baseline.

    NOTE: a MAE of these two arrays is NOT a player-independent metric, because
    |(y-base) - (p-base)| == |y - p| exactly. `evaluate_predictions` therefore
    de-means BOTH sides within the test player before measuring. This helper is
    kept for callers that want the raw deviations.
    """
    base = te["player"].map(train_means).astype(float).to_numpy()
    return np.asarray(y, float) - base, np.asarray(p, float) - base


def calibration(y, p, n_bins: int = 10) -> list[dict]:
    """Prediction-decile bins: mean predicted vs mean actual (a diagnostic)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    if len(y) < n_bins * 2 or np.std(p) == 0:
        return []
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b = np.digitize(p, edges[1:-1])
    out = []
    for i in range(n_bins):
        m = b == i
        if m.sum() == 0:
            continue
        out.append({"bin": i, "n": int(m.sum()),
                    "pred_mean": round(float(p[m].mean()), 3),
                    "actual_mean": round(float(y[m].mean()), 3),
                    "gap": round(float(p[m].mean() - y[m].mean()), 3),
                    "frac_within_5pp": round(float((np.abs(p[m] - y[m]) <= 5).mean()), 3)})
    return out


def _ndcg_at_k(order: np.ndarray, y: np.ndarray, k: int) -> float | None:
    kk = min(k, len(order))
    disc = 1.0 / np.log2(np.arange(2, kk + 2))
    dcg = float((y[order[:kk]] * disc).sum())
    ideal = np.sort(y)[::-1][:kk]
    idcg = float((ideal * disc).sum())
    return (dcg / idcg) if idcg > 0 else None


def ranking_by_player(te: pd.DataFrame, pred: np.ndarray, target: str,
                      ks=TOP_K) -> dict:
    """Per-player Spearman / Precision@K / NDCG@K over the test rows."""
    d = te[["player", target]].copy()
    d["pred"] = np.asarray(pred, float)
    rows = {}
    for player, g in d.groupby("player", sort=True):
        y, p, n = g[target].to_numpy(float), g["pred"].to_numpy(float), len(g)
        r = {"n": int(n), "spearman": spearman(y, p),
             "pred_mean": round(float(p.mean()), 3),
             "actual_mean": round(float(y.mean()), 3),
             "mae": round(mae(y, p), 3)}
        if n >= 5:
            order = np.argsort(-p, kind="stable")
            for k in ks:
                kk = min(k, n)
                r[f"precision@{k}"] = round(
                    float(len(set(order[:kk]) & set(np.argsort(-y, kind="stable")[:kk])) / kk), 4)
                r[f"ndcg@{k}"] = _ndcg_at_k(order, y, k)
        rows[str(player)] = r
    agg = {}
    for name in ("spearman", "mae", *[f"precision@{k}" for k in ks],
                 *[f"ndcg@{k}" for k in ks]):
        vals = [v[name] for v in rows.values() if v.get(name) is not None]
        if vals:
            agg[name] = {"mean": round(float(np.mean(vals)), 4),
                         "median": round(float(np.median(vals)), 4),
                         "min": round(float(np.min(vals)), 4),
                         "max": round(float(np.max(vals)), 4),
                         "n_players": len(vals)}
    return {"per_player": rows, "aggregate": agg}


def evaluate_predictions(te: pd.DataFrame, pred: np.ndarray, target: str,
                         train_means: dict, *, want_ranking: bool = False,
                         with_calibration: bool = False) -> dict:
    """Everything the protocol asks for, for one model on one test frame.

    Player-internal metrics: "player centred" means the ACTUAL and the PREDICTED
    values are each centred on that player's training mean, and then we measure how
    well the model reproduces the player's within-player deviations:

        dev_actual    = y - player_train_mean
        dev_predicted = p - player_train_mean
        player_centered_mae = MAE(dev_predicted, dev_actual)     (= MAE(cp, cy))

    Note the argument order above: the metric compares the two DEVIATIONS. A first
    version compared `|cy - cp|`, which collapses to `|y - p|` and is therefore
    exactly the plain MAE - the patch 2.2 metric would have been a no-op.
    """
    y = te[target].to_numpy(float)
    p = np.asarray(pred, float)
    # Player-internal signal: de-mean BOTH the actual and the predicted values
    # within each player. This is the quantity patch 2.2 asks for - "how well does
    # the model rank charts within one player" - and it is invariant to any constant
    # per-player offset and to any global shift, so it cannot be won by reproducing
    # player levels. (Subtracting the training mean from both sides alone is NOT
    # enough: |dy - dp| = |y - p|, i.e. byte-identical to the plain MAE.)
    dev_y = y - pd.Series(y).groupby(te["player"].values).transform("mean").to_numpy()
    dev_p = p - pd.Series(p).groupby(te["player"].values).transform("mean").to_numpy()
    cy, cp = centered(y, p, te, train_means, target)
    out = {
        "mae": round(mae(y, p), 4),
        "rmse": round(rmse(y, p), 4),
        "bias": round(bias(y, p), 4),
        "player_centered_mae": round(mae(dev_y, dev_p), 4),
        "player_centered_rmse": round(rmse(dev_y, dev_p), 4),
        "player_centered_bias": round(bias(dev_y, dev_p), 4),
        "spearman": (lambda s: round(s, 4) if s is not None else None)(spearman(y, p)),
        "player_centered_spearman": (
            lambda s: round(s, 4) if s is not None else None)(spearman(dev_y, dev_p)),
        "per_player_mae": {str(k): round(float(v), 4) for k, v in
                           pd.Series(np.abs(y - p)).groupby(te["player"].values).mean().items()},
        "per_player_centered_mae": {
            str(k): round(float(v), 4) for k, v in
            pd.Series(np.abs(dev_y - dev_p)).groupby(te["player"].values).mean().items()},
        "per_player_n": {str(k): int(v) for k, v in
                         te["player"].value_counts().sort_index().items()},
    }
    if with_calibration:
        out["calibration"] = calibration(y, p)
    if want_ranking:
        out["ranking"] = ranking_by_player(te, p, target)
        # aggregate the per-player rank metrics at the top level too: the M3 gate
        # is stated in terms of player-internal ranking, so it must be readable
        # without digging into nested structures.
        r = out["ranking"]["aggregate"]
        out["player_rank_spearman"] = r.get("spearman", {}).get("mean")
        out["player_topk_ndcg"] = {f"ndcg@{k}": r.get(f"ndcg@{k}", {}).get("mean")
                                   for k in TOP_K}
    return out
