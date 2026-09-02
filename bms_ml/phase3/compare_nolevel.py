"""Post-ablation comparison: objective features (no difficulty-table anywhere)
vs the v0 table-based setup, on the rebuilt dataset (table filter dropped).

Two evaluation scopes:
  table — train&test restricted to in-table rows: same sample space as the old
          pipeline, isolates the FEATURE change (h_knn_acc vs h_level_acc, no
          table chart feats)
  all   — the full rebuilt sample space: shows the benefit of dropping the
          table restriction (44% more samples)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"

STAT = [f"c_{n}" for n in [
    "total_notes", "ln_ratio", "duration_sec", "measures", "initial_bpm", "min_bpm",
    "max_bpm", "bpm_change_count", "stop_count", "stop_total_sec", "lane0_scratch",
    "lane1", "lane2", "lane3", "lane4", "lane5", "lane6", "lane7", "scratch_ratio",
    "avg_nps", "peak_nps_1s", "peak_measure_nps", "chord_count", "chord2_count",
    "chord3plus_count", "jack_count"]]
H_FEATS = ["h_knn_acc", "h_n_firstplays", "h_acc_mean", "h_acc_std", "h_acc_last10",
           "h_bp_mean", "h_bp_ratio_mean", "h_fail_rate", "h_fc_rate",
           "h_days_since_active", "h_plays_last30d", "h_days_span"]


class Imp:
    def fit(self, X):
        X = np.asarray(X, float)
        self.m = np.nan_to_num(np.nanmedian(X, axis=0))
        return self

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def transform(self, X):
        X = np.asarray(X, float).copy()
        return np.where(np.isnan(X), self.m, X)


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def centered_r2(d, pred, col):
    a = d[col].values - d.groupby("player")[col].transform("mean").values
    b = pred - d.groupby("player")[col].transform("mean").values
    den = np.sum((a - a.mean()) ** 2)
    return float(1 - np.sum((a - b) ** 2) / den)


def main() -> None:
    df = pd.read_parquet(DS / "samples.parquet")
    results: dict = {}
    for scope in ["table", "all"]:
        if scope == "table":
            d = df[df["table"].notna()]
        else:
            d = df
        tr, te = d[d["phase"] == "train"], d[d["phase"] == "test"]
        res = {"n_train": len(tr), "n_test": len(te)}

        def hgb(feats, y):
            imp = Imp()
            sc = StandardScaler().fit(imp.fit_transform(tr[feats]))
            Xtr = sc.transform(imp.fit_transform(tr[feats]))
            Xte = sc.transform(imp.transform(te[feats]))
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                              max_depth=3, random_state=0)
            m.fit(Xtr, y)
            return m.predict(Xte)

        sets = {"A": STAT, "H": H_FEATS, "B": STAT + H_FEATS}
        for tag, feats in sets.items():
            acc_p = hgb(feats, tr["acc"].values)
            lamp_p = hgb(feats, tr["lamp"].values.astype(float))
            bp_p = np.expm1(hgb(feats, np.log1p(tr["bp"].values)))
            res[tag] = {
                "acc": {"mae": round(mae(te["acc"], acc_p), 3),
                        "r2": round(r2(te["acc"], acc_p), 3),
                        "centered_r2": round(centered_r2(te, acc_p, "acc"), 3),
                        "per_player": {p: round(mae(g["acc"], acc_p[te["player"].values == p]), 3)
                                       for p, g in te.groupby("player")}},
                "lamp": {"ord_mae": round(mae(te["lamp"], lamp_p), 3),
                         "qwk": round(float(cohen_kappa_score(
                             te["lamp"], np.clip(np.round(lamp_p), 1, 9).astype(int),
                             weights="quadratic", labels=list(range(1, 10)))), 3)},
                "bp": {"raw_mae": round(mae(te["bp"], bp_p), 2),
                       "ratio_mae_x1000": round(mae(te["bp"] / te["notes"],
                                                    bp_p / te["notes"]) * 1000, 2)},
            }
        from chart_repr import feature_manifest, OBJECTIVE_STAT_COLS, HISTORY_FEATURES
        import sys; sys.path.insert(0, str(Path(__file__).parent))
        res['features_used'] = feature_manifest(STAT, H_FEATS)
        results[scope] = res
        print(f"[{scope}] n={len(tr)}/{len(te)}",
              {t: round(res[t]["acc"]["mae"], 2) for t in sets})

    json.dump(results, open(OUT / "compare_nolevel.json", "w"), indent=2)
    print("saved ->", OUT / "compare_nolevel.json")


if __name__ == "__main__":
    main()
