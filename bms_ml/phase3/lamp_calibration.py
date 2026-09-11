"""Phase 3.6: post-hoc calibration of the lamp head (Roxy's ordinal-calibration lesson).

Roxy's meta head is calibrated on the 0.5-grid ordinal dan scale because difficulty
output is consumed as an ordinal. Our lamp head is a plain regression on the integer
lamp, evaluated with ordinal MAE and QWK. If the regression is biased on the ordinal
scale (e.g. compressed toward the middle by the squared loss), a monotone recalibration
fitted on held-out predictions can reduce ordinal MAE for free - no retraining, no
protocol change, just a 1-D isotonic map between raw prediction and expected lamp.

Honesty about the calibration split: the isotonic map is fitted on a TIME-based held-out
slice of the train band (each player's latest 20% of train rows), never on the test band,
so the test numbers stay out-of-sample in the protocol sense.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "bms_ml" / "output" / "phase3" / "dataset"
sys.path.insert(0, str(ROOT / "bms_ml" / "phase3"))
from chart_repr import BEST_FEATURES, HISTORY_FEATURES, OBJECTIVE_STAT_COLS  # noqa: E402
from common import HGBModel, load_samples, mae  # noqa: E402


def qwk(y, p) -> float:
    return float(cohen_kappa_score(y, np.clip(np.round(p), 1, 9).astype(int),
                                   weights="quadratic", labels=list(range(1, 10))))


def main() -> None:
    hr = pd.read_parquet(DS / "history_response.parquet")
    static = OBJECTIVE_STAT_COLS + HISTORY_FEATURES
    need = [c for c in BEST_FEATURES if c not in static]
    df = (load_samples()
          .merge(hr[["player", "sha256"] + [c for c in need if c in hr.columns]],
                 on=["player", "sha256"], how="left"))
    tr, te = df[df["phase"] == "train"], df[df["phase"] == "test"]

    # per-player time split of the train band: 80% fit / 20% calibration
    fit_idx, cal_idx = [], []
    for _p, g in tr.groupby("player"):
        g = g.sort_values("time")
        k = max(1, int(len(g) * 0.8))
        fit_idx.extend(g.index[:k])
        cal_idx.extend(g.index[k:])
    fit, cal = tr.loc[fit_idx], tr.loc[cal_idx]
    print(f"fit {len(fit)} / calib {len(cal)} (per-player time split of the train band)")

    model = HGBModel(fit, BEST_FEATURES, fit["lamp"].values.astype(float))
    raw_cal = model.predict(cal)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1.0, y_max=9.0)
    iso.fit(raw_cal, cal["lamp"].values.astype(float))

    raw_te = model.predict(te)
    cal_te = np.clip(iso.predict(raw_te), 1, 9)
    y = te["lamp"].values

    print(f"\n{'variant':18}{'ordMAE':>9}{'QWK':>9}")
    for tag, p in (("raw", raw_te), ("isotonic", cal_te),
                   ("isotonic+0.5grid", np.round(cal_te * 2) / 2)):
        print(f"{tag:18}{mae(y, p):9.3f}{qwk(y, p):9.3f}")

    grid = np.array([2, 3, 4, 5, 6, 7, 8], dtype=float)
    shift = iso.predict(grid) - grid
    print("\nisotonic shift by raw prediction level:")
    for g, s in zip(grid, shift):
        print(f"  raw {g:.0f} -> {g + s:.2f}  ({s:+.2f})")


if __name__ == "__main__":
    main()
