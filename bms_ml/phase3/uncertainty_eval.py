"""Phase 3.6: prediction uncertainty for the fused model - can we give honest bands?

The variance diagnostic (2026-09-11) showed per-player error is mostly that player's own
behavioural variance (corr(MAE, within-player std) = +0.882, MAE/std median 0.45). A
single point prediction hides that: for yangtao, "48" means somewhere in 20-80. This
script tests whether HGB quantile heads produce honest, feature-conditioned bands, and a
binary pass model produces usable P(lamp >= 4).

Everything trains on the train band with the fused feature set (BEST_FEATURES), exactly
like the shipped configuration. The validity check is COVERAGE on the test band: the
[q10, q90] interval should contain ~80% of the actual outcomes; a band whose coverage is
far off is decoration, not information. The value check is HETEROSCEDASTICITY: band
width should track each player's own behavioural variance (reiaki narrow, yangtao wide) -
a constant-width band would carry no per-player information.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "bms_ml" / "output" / "phase3" / "dataset"
sys.path.insert(0, str(ROOT / "bms_ml" / "phase3"))
from chart_repr import BEST_FEATURES, HISTORY_FEATURES, OBJECTIVE_STAT_COLS  # noqa: E402
from common import HGB_KW, load_samples, mae  # noqa: E402

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


class QHead:
    """Impute+scale+quantile-HGB, fit-once/predict-many, same preprocessing family as
    common.HGBModel but with a quantile loss."""

    def __init__(self, tr, feats, y, q: float, seed: int = 0):
        self.feats = list(feats)
        self.imp = SimpleImputer(strategy="median")
        self.sc = StandardScaler()
        X = self.sc.fit_transform(self.imp.fit_transform(tr[self.feats]))
        self.m = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                               random_state=seed, **HGB_KW)
        self.m.fit(X, y)

    def predict(self, te) -> np.ndarray:
        return self.m.predict(self.sc.transform(self.imp.transform(te[self.feats])))


def main() -> None:
    hr = pd.read_parquet(DS / "history_response.parquet")
    static = OBJECTIVE_STAT_COLS + HISTORY_FEATURES
    need = [c for c in BEST_FEATURES if c not in static]
    df = (load_samples()
          .merge(hr[["player", "sha256"] + [c for c in need if c in hr.columns]],
                 on=["player", "sha256"], how="left"))
    tr, te = df[df["phase"] == "train"], df[df["phase"] == "test"]
    print(f"train {len(tr)} / test {len(te)} | features {len(BEST_FEATURES)}")

    # quantile regression with finite samples + squared-tuned hyperparameters is known
    # to UNDER-cover, so every interval is conformalised (CQR, Romano et al.): the
    # correction Q is the 80th percentile of the calibration nonconformity
    # max(q10 - y, y - q90), fitted on a per-player time split of the train band and
    # never on the test band.
    fit_idx, cal_idx = [], []
    for _p, g in tr.groupby("player"):
        g = g.sort_values("time")
        k = max(1, int(len(g) * 0.8))
        fit_idx.extend(g.index[:k])
        cal_idx.extend(g.index[k:])
    fit, cal = tr.loc[fit_idx], tr.loc[cal_idx]
    print(f"conformal split: fit {len(fit)} / calib {len(cal)}")

    # ---- acc: conformalised quantile band ---------------------------------------
    qh = {q: QHead(fit, BEST_FEATURES, fit["acc"].values, q) for q in (0.1, 0.5, 0.9)}
    med = qh[0.5].predict(te)
    lo_cal, hi_cal = qh[0.1].predict(cal), qh[0.9].predict(cal)
    scores = np.maximum(lo_cal - cal["acc"].values, cal["acc"].values - hi_cal)
    qq = min(1.0, np.ceil((len(scores) + 1) * 0.8) / len(scores))
    Q = float(np.quantile(scores, qq))
    lo, hi = qh[0.1].predict(te) - Q, qh[0.9].predict(te) + Q
    cov = float(((te["acc"] >= lo) & (te["acc"] <= hi)).mean())
    width = hi - lo
    print(f"\nacc  MAE(median) {mae(te['acc'], med):.3f} | conformal Q {Q:.2f}")
    print(f"  [q10-Q, q90+Q] coverage {cov:.3f} (nominal 0.80) | "
          f"mean width {width.mean():.2f} | median width {np.median(width):.2f}")
    ps = te.groupby("player")["acc"].std()
    w_by = pd.Series(width).groupby(te["player"].values).median()
    j = pd.concat([ps.rename("std"), w_by.rename("width")], axis=1).dropna()
    print(f"  corr(player std, median band width) = {j['std'].corr(j['width']):+.3f}")
    print("  per-player width vs std:")
    print(j.sort_values("std").round(2).to_string())

    # ---- lamp: pass probability (SHARED fitted preprocessing, not per-frame) ----
    ytr_pass = (fit["lamp"].values >= 4).astype(int)
    imp = SimpleImputer(strategy="median").fit(fit[BEST_FEATURES])
    sc = StandardScaler().fit(imp.transform(fit[BEST_FEATURES]))
    clf = HistGradientBoostingClassifier(random_state=0, **HGB_KW)
    clf.fit(sc.transform(imp.transform(fit[BEST_FEATURES])), ytr_pass)
    p_pass = clf.predict_proba(sc.transform(imp.transform(te[BEST_FEATURES])))[:, 1]
    y_pass = (te["lamp"].values >= 4).astype(int)
    print(f"\nlamp pass model: test base rate {y_pass.mean():.3f} | "
          f"fit-band base rate {ytr_pass.mean():.3f}")
    print(f"  brier {brier_score_loss(y_pass, p_pass):.3f} | "
          f"AUC {roc_auc_score(y_pass, p_pass):.3f} | "
          f"mean p {p_pass.mean():.3f}")
    dec = pd.qcut(p_pass, 5, duplicates="drop")
    cal_tab = pd.DataFrame({"p": p_pass, "y": y_pass, "dec": dec}).groupby(
        "dec", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("p", "size"))
    print("  calibration by predicted-probability quintile:")
    print(cal_tab.round(3).to_string())
    lqh = {q: QHead(fit, BEST_FEATURES, fit["lamp"].values.astype(float), q)
           for q in (0.25, 0.5, 0.75)}
    print(f"  lamp ord MAE(median) "
          f"{mae(te['lamp'], lqh[0.5].predict(te)):.3f}")

    # ---- bp: conformalised quantile band ---------------------------------------
    bq = {q: QHead(fit, BEST_FEATURES, np.log1p(fit["bp"].values), q)
          for q in (0.1, 0.9)}
    blo_c, bhi_c = bq[0.1].predict(cal), bq[0.9].predict(cal)
    ys = np.log1p(cal["bp"].values)
    bscores = np.maximum(blo_c - ys, ys - bhi_c)
    bQ = float(np.quantile(bscores, qq))
    blo = np.expm1(np.clip(bq[0.1].predict(te) - bQ, 0, 12))
    bhi = np.expm1(np.clip(bq[0.9].predict(te) + bQ, 0, 12))
    bcov = float(((te["bp"] >= blo) & (te["bp"] <= bhi)).mean())
    print(f"\nbp   [q10-Q, q90+Q] coverage {bcov:.3f} (nominal 0.80, log space) | "
          f"median width {np.median(bhi - blo):.1f}")


if __name__ == "__main__":
    main()
