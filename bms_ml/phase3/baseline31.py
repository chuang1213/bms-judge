"""[DEPRECATED 2026-09-03] uses difficulty-table-derived features (h_level_acc, level_norm, table one-hots) which are removed from the current pipeline; kept for history. Use compare_nolevel.py / c_chart_aware.py instead. See PROTOCOL.md.

Phase 3.1 baselines: three targets (acc / lamp / BP) x feature sets
(A chart-only / H history-statistics-only / B chart+history), same strict time
split as v0 (train targets in (q50,q75], test targets after q75).

Lamp is ordinal (beatoraja ClearType: FAILED=1 < ASSIST=2 < LASSIST=3 < EASY=4 <
NORMAL=5 < HARD=6 < EXHARD=7 < FC=8 < PERFECT=9); compared as multiclass
classification vs plain regression, judged by ordinal MAE and quadratic-weighted kappa.

BP is run on two tracks (raw BP and notes-normalized BP=bp/notes), each with a
raw and a log1p variant; every variant is evaluated on BOTH native scales
(raw MAE in BP units, ratio MAE in misses/note) for head-to-head comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"
LAMP_MIN, LAMP_MAX = 1, 9

C_FEATS = None  # filled in main from data columns
H_FEATS = ["h_level_acc", "h_n_firstplays", "h_acc_mean", "h_acc_std", "h_acc_last10",
           "h_bp_mean", "h_bp_ratio_mean", "h_fail_rate", "h_fc_rate",
           "h_days_since_active", "h_plays_last30d", "h_days_span"]


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def medae(y, p):
    return float(np.median(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


class Imputer:
    def fit(self, X):
        X = np.asarray(X, float)
        self.med = np.nan_to_num(np.nanmedian(X, axis=0))
        return self

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def transform(self, X):
        X = np.asarray(X, float).copy()
        return np.where(np.isnan(X), self.med, X)


def fit_hgb(tr, te, feats, ytr, kind="reg", seed=0):
    imp = Imputer()
    Xtr, Xte = imp.fit_transform(tr[feats]), imp.transform(te[feats])
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if kind == "reg":
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=3,
                                          random_state=seed)
    else:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_depth=3,
                                           random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict(Xte), m


def eval_by_player(te, pred):
    return {p: round(mae(g["acc"], pred[te["player"].values == p]), 3)
            for p, g in te.groupby("player")}


def main() -> None:
    df = pd.read_parquet(DS / "samples.parquet")
    global C_FEATS
    C_FEATS = [c for c in df.columns if c.startswith("c_")]
    C_FEATS += ["level_norm", "table_satellite", "table_stella", "table_insane"]
    tr, te = df[df["phase"] == "train"], df[df["phase"] == "test"]
    res: dict = {"n_train": len(tr), "n_test": len(te)}
    sets = {"A": C_FEATS, "H": H_FEATS, "B": C_FEATS + H_FEATS}

    # ---------------- acc ----------------
    res["acc"] = {}
    for tag, feats in sets.items():
        pred, _ = fit_hgb(tr, te, feats, tr["acc"].values)
        res["acc"][tag] = {"mae": round(mae(te["acc"], pred), 3),
                           "r2": round(r2(te["acc"], pred), 3),
                           "per_player": eval_by_player(te, pred)}

    # ---------------- lamp (ordinal) ----------------
    lamp_tr, lamp_te = tr["lamp"].values, te["lamp"].values
    res["lamp"] = {}

    def lamp_metrics(pred):
        return {"ord_mae": round(mae(lamp_te, pred), 3),
                "qwk": round(float(cohen_kappa_score(lamp_te, np.round(pred).clip(LAMP_MIN, LAMP_MAX),
                                                     weights="quadratic",
                                                     labels=list(range(LAMP_MIN, LAMP_MAX + 1)))), 3)}

    for tag, feats in sets.items():
        pred_reg, _ = fit_hgb(tr, te, feats, lamp_tr.astype(float))
        pred_cls, _ = fit_hgb(tr, te, feats, lamp_tr, kind="cls")
        res["lamp"][tag] = {"regression": lamp_metrics(pred_reg),
                            "classification": lamp_metrics(pred_cls),
                            "regression_per_player": {
                                p: round(mae(g["lamp"], pred_reg[te["player"].values == p]), 3)
                                for p, g in te.groupby("player")}}

    # ---------------- BP two tracks x two transforms ----------------
    # every variant is scored on both native scales:
    #   raw-MAE (BP units) against raw truth, ratio-MAE (misses/note) against ratio truth
    raw_truth = te["bp"].values
    ratio_truth = (te["bp"] / te["notes"]).values
    tracks = {
        "raw": {"variants": {"raw": tr["bp"].values, "log1p": np.log1p(tr["bp"].values)}},
        "ratio": {"variants": {"raw": (tr["bp"] / tr["notes"]).values,
                               "log1p": np.log1p(tr["bp"] / tr["notes"]).values}},
    }
    res["bp"] = {}
    for track, spec in tracks.items():
        for tag, feats in sets.items():
            for vname, ytr in spec["variants"].items():
                pred, _ = fit_hgb(tr, te, feats, ytr)
                if track == "raw":
                    p_raw = np.expm1(pred) if vname == "log1p" else pred
                    p_ratio = p_raw / te["notes"].values
                else:
                    p_ratio = np.expm1(pred) if vname == "log1p" else pred
                    p_raw = p_ratio * te["notes"].values
                res["bp"].setdefault(track, {}).setdefault(tag, {})[vname] = {
                    "raw_mae": round(mae(raw_truth, p_raw), 2),
                    "raw_medae": round(medae(raw_truth, p_raw), 2),
                    "ratio_mae_x1000": round(mae(ratio_truth, p_ratio) * 1000, 2),
                }
                if track == "raw" and vname == "log1p":
                    res["bp"][track][tag][vname]["raw_per_player"] = {
                        p: round(mae(g["bp"], p_raw[te["player"].values == p]), 2)
                        for p, g in te.groupby("player")}

    json.dump(res, open(OUT / "baseline31_results.json", "w"), indent=2)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
