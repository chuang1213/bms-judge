"""Phase 4 held-out split definitions.

Single source of truth for splits: `evaluate_matrix.py` (baselines) and
`run_m3.py` (matrix completion) both import from here, so a model can never be
scored under a quietly different split than the baseline it is compared against.

The seed→split mapping is deterministic and independent of PYTHONHASHSEED (see
`_rng`). Splits are defined on a per-client frame; clients are never pooled.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

# split names understood by split_mask()
WARM_SPLITS = ("random_interaction", "loo_chart_per_player")
COLD_SPLITS = ("cold_player", "cold_chart")


def _rng(seed: int, salt: str) -> np.random.Generator:
    h = hashlib.sha256(f"{salt}|{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def loo_k_name(k: int) -> str:
    return f"loo_{k}_charts_per_player"


def parse_split(name: str) -> tuple[str, int | None]:
    """'loo_5_charts_per_player' -> ('loo_k_charts_per_player', 5)."""
    if name.startswith("loo_") and name.endswith("_charts_per_player"):
        k = int(name[len("loo_"):-len("_charts_per_player")])
        return "loo_k_charts_per_player", k
    return name, None


def all_split_names(loo_ks: tuple[int, ...] = (1, 5, 10)) -> list[str]:
    return ["random_interaction", *[loo_k_name(k) for k in loo_ks],
            "cold_player", "cold_chart"]


def split_mask(df: pd.DataFrame, kind: str, seed: int, frac: float) -> np.ndarray:
    """Boolean TEST mask over `df` rows. Deterministic in (kind, seed, frac).

    The k=1 leave-one-chart-out case keeps its original salt ("loo") so splits
    recorded in `evaluate_matrix.json` before the loo_k patch still reproduce.
    """
    n = len(df)
    base, k = parse_split(kind)

    if base == "random_interaction":
        return _rng(seed, "ri").random(n) < frac

    if base == "loo_k_charts_per_player":
        assert k is not None
        if k == 1:
            order = np.argsort(_rng(seed, "loo").random(n), kind="stable")
            test = np.zeros(n, dtype=bool)
            pick = (pd.DataFrame({"i": order, "p": df["player"].values[order]})
                    .groupby("p", sort=True)["i"].first())
            test[pick.to_numpy()] = True
            return test

        # k charts per player, sampled without replacement from that player's
        # observed charts. Per-player RNG so one player's size does not shift
        # another player's draw.
        test = np.zeros(n, dtype=bool)
        for player, idx in df.groupby("player", sort=True).indices.items():
            charts = np.unique(df["sha256"].values[idx])
            kk = min(k, len(charts))
            r = _rng(seed, f"loo{k}|{player}").random(len(charts))
            held = set(charts[np.argsort(r, kind="stable")[:kk]])
            test[idx] = np.isin(df["sha256"].values[idx], list(held))
        return test

    if base in ("cold_player", "cold_chart"):
        group = "player" if base == "cold_player" else "sha256"
        vals = np.sort(df[group].unique())
        r = _rng(seed, base).random(len(vals))
        n_held = max(1, int(round(len(vals) * frac)))
        held = set(vals[np.argsort(r, kind="stable")[:n_held]])
        return df[group].isin(held).to_numpy()

    raise ValueError(f"unknown split {kind!r}")


def holdout_mask(n_or_df, seed: int, frac: float = 0.1) -> np.ndarray:
    """BOOLEAN mask over the rows of `train_df`, marking a validation slice.

    Returns a boolean array (not indices) so it composes with the other masks in
    this module. Callers must index POSITIONALLY (`np.flatnonzero(...)`) - a
    boolean mask applied as `.iloc`/`df[mask]` on a frame whose index was reset
    will silently select the wrong rows.
    """
    n = len(n_or_df)
    return _rng(seed, "inner_val").random(n) < frac


def holdout_indices(n_or_df, seed: int, frac: float = 0.1) -> np.ndarray:
    """Positional row indices of a random-CELL inner validation slice."""
    return np.flatnonzero(holdout_mask(n_or_df, seed, frac))


def inner_validation_indices(train_df: pd.DataFrame, kind: str, seed: int,
                             frac: float = 0.1) -> np.ndarray:
    """Positional row indices of an inner validation slice OF `train_df`.

    The slice MIRRORS THE STRUCTURE of the outer split, because that is what the
    hyper-parameters have to generalise to:

      * random_interaction: a per-player random-INTERACTION holdout, charts stay
        visible in the inner fit. This mirrors exactly what the outer split does
        (one random interaction per player), only at a larger fraction.
      * loo_k_charts_per_player: k whole charts per player, as in the outer split.
      * cold_player / cold_chart: the same group-disjoint structure as the split.

    Everything is drawn from TRAIN rows only, so the outer test set is untouched.
    Returns POSITIONAL indices, ready for `train_df.iloc[idx]`.

    MEASURED LIMITATION (2026-09-12): this still does NOT track the outer test
    closely enough to pick the regularisation. Inner validation prefers weak
    regularisation (reg ~ 0.1-0.3) under every variant tried - random cells,
    whole charts, per-player interactions, scored by RMSE or MAE - while the real
    test split prefers reg ~ 1-3. The inner task is intrinsically easier (the
    evaluated rows were part of the frame the hyper-parameters were fitted
    against), so it rewards fitting harder. `run_m3.py` therefore reports the
    selection procedure's choice AND the full grid, and never presents the
    test-best grid point as if the procedure had found it.
    """
    base, k = parse_split(kind)
    n = len(train_df)
    if base == "random_interaction":
        idx = _per_player_random_rows(train_df, seed, frac)
    elif base == "loo_k_charts_per_player":
        idx = np.flatnonzero(split_mask(train_df, loo_k_name(k or 1), seed, frac))
    elif base == "cold_player":
        idx = np.flatnonzero(split_mask(train_df, "cold_player", seed, frac))
    else:
        idx = np.flatnonzero(split_mask(train_df, "cold_chart", seed, frac))
    if len(idx) == 0 or len(idx) == n:            # degenerate -> random cells
        idx = holdout_indices(train_df, seed, frac)
    return idx


def _per_player_random_rows(train_df: pd.DataFrame, seed: int, frac: float) -> np.ndarray:
    """Mask `frac` of EACH player's rows; the charts stay visible in the fit."""
    rng = _rng(seed, "ri_inner")
    picks = []
    for _, rows in train_df.groupby("player", sort=True).indices.items():
        k = max(1, int(round(len(rows) * frac)))
        picks.append(rng.choice(rows, size=min(k, len(rows)), replace=False))
    return np.sort(np.concatenate(picks)) if picks else np.array([], dtype=int)


def split_report(df: pd.DataFrame, test: np.ndarray, kind: str) -> dict:
    te, tr = df[test], df[~test]
    base, k = parse_split(kind)
    rep = {
        "kind": kind,
        "loo_k": k,
        "n_test": int(test.sum()),
        "n_train": int((~test).sum()),
        "test_frac_rows": round(float(test.mean()), 4),
        "test_players": int(te["player"].nunique()),
        "test_charts": int(te["sha256"].nunique()),
        "train_players": int(tr["player"].nunique()),
        "train_charts": int(tr["sha256"].nunique()),
        "cold_players": sorted(set(te["player"]) - set(tr["player"])),
        "n_cold_charts": int(len(set(te["sha256"]) - set(tr["sha256"]))),
        "test_obs_per_player": {str(p): int(v) for p, v in
                                te["player"].value_counts().sort_index().items()},
    }
    if base in WARM_SPLITS:
        rep["note"] = ("every test player and chart is seen in train; measures "
                       "interpolation, not cold start")
    else:
        rep["note"] = ("held-out groups are absent from train; only global/chart or "
                       "global/player structure can help")
    # Loo splits are the small-k end: say out loud how many rows that is.
    if base == "loo_k_charts_per_player":
        rep["mean_test_rows_per_player"] = (
            round(float(te.groupby("player").size().mean()), 2) if len(te) else 0.0)
    return rep
