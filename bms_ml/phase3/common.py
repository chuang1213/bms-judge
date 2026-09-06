"""Phase 3.5: shared evaluation primitives for the first-play prediction protocol.

Why this module exists
----------------------
Every Phase 3 experiment fits the same model family on the same sample space and
reports the same metrics, but before 3.5 each script re-declared them: 5 copies of
the 27-dim chart feature list, 5 copies of the 12-dim history list, 5 copies of
`Imputer`/`mae`/`r2`/`centered_r2`/`hgb`. Adding one feature therefore meant
editing five files — which is exactly how `c_jrank` (2026-09-05) ended up in every
experiment script but *not* in `chart_repr.OBJECTIVE_STAT_COLS`, the module that
PROTOCOL.md §3 designates as the single feature registry.

Scope discipline
----------------
This module holds **only primitives that are provably identical across all
callers** — each one was diffed against the pre-refactor implementations and
verified to reproduce `compare_nolevel.json` byte-for-byte. Per-experiment metric
*composition* (which dicts get built, what gets rounded into the results JSON)
stays in each script, so no experiment's reported numbers change.

Feature lists live in `chart_repr.py` (the registry). Do not re-declare them here
or in experiment scripts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "bms_ml" / "output" / "phase3" / "dataset"

# HGB hyper-parameters shared by every Phase 3 experiment (fixed since 3.1).
# Deterministic given `seed`; PROTOCOL.md §6 requires a fixed random_state.
HGB_KW = dict(max_iter=300, learning_rate=0.06, max_depth=3)


class Imputer:
    """Median imputation fitted on the TRAIN frame, applied to any frame.

    NaN -> train column median (0.0 for all-NaN columns). Fitting on train only is
    part of the protocol: test-set statistics must never enter the pipeline.
    """

    def fit(self, X):
        X = np.asarray(X, float)
        self.med = np.nan_to_num(np.nanmedian(X, axis=0))
        return self

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def transform(self, X):
        X = np.asarray(X, float).copy()
        return np.where(np.isnan(X), self.med, X)


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def r2(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def centered_r2(d: pd.DataFrame, pred, col: str = "acc") -> float:
    """R2 of WITHIN-PLAYER deviations.

    The single most important metric in the project: it removes each player's own
    mean, so a model that only knows "this player is strong" scores ~0. Positive
    centered R2 is the evidence that the model captures player x chart interaction
    rather than two marginals (PROTOCOL.md §5).
    """
    pred = np.asarray(pred, float)
    a = d[col].values.astype(float) - d.groupby("player")[col].transform("mean").values
    b = pred - d.groupby("player")[col].transform("mean").values
    den = np.sum((a - a.mean()) ** 2)
    return float(1 - np.sum((a - b) ** 2) / den) if den > 0 else float("nan")


def hgb_fit_predict(tr: pd.DataFrame, te: pd.DataFrame, feats, y, seed: int = 0) -> np.ndarray:
    """Fit HGB on `tr[feats] -> y` (impute on train, scale on train) and predict `te`."""
    imp = Imputer()
    sc = StandardScaler().fit(imp.fit_transform(tr[feats]))
    m = HistGradientBoostingRegressor(random_state=seed, **HGB_KW)
    m.fit(sc.transform(imp.fit_transform(tr[feats])), y)
    return m.predict(sc.transform(imp.transform(te[feats])))


def difficulty_region(row) -> str:
    """Coarse difficulty coordinate: SL / ST0-3 / ST4-7 / ST8+ / insane(★).

    Difficulty-table level is a REPORTING COORDINATE ONLY — never a feature
    (PROTOCOL.md §1).
    """
    t, lv = row["table"], row["level"]
    if t == "satellite":
        return "SL"
    if t == "stella":
        return "ST0-3" if lv <= 3 else ("ST4-7" if lv <= 7 else "ST8+")
    return "insane(★)"


def add_region(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a `region` column (vectorised, unlike row.apply)."""
    out = df.copy()
    lv = out["level"]
    out["region"] = np.select(
        [out["table"] == "satellite",
         (out["table"] == "stella") & (lv <= 3),
         (out["table"] == "stella") & (lv <= 7),
         out["table"] == "stella"],
        ["SL", "ST0-3", "ST4-7", "ST8+"],
        default="insane(★)",
    )
    return out


def load_samples() -> pd.DataFrame:
    """Protocol samples (train/test bands only; history rows excluded)."""
    return pd.read_parquet(DS / "samples.parquet")


def load_firstplays() -> pd.DataFrame:
    """All in-scope first-play events (used as history and for coverage)."""
    return pd.read_parquet(DS / "firstplays.parquet")
