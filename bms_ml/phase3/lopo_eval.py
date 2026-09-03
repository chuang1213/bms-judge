"""Phase 3.5: Leave-one-player-out (LOPO) cold-start evaluation.

User proposal: remove player P from training, train on the remaining players, then
predict a masked score from P's own archive. Feasibility note: for first-play
targets the masking is structural — features only contain events STRICTLY BEFORE
the target's own first play, so the target outcome can never appear in its input
(the 2026-09-04 leak-fix semantics, PROTOCOL.md §1).

Protocol:
  for each player P (10 players):
    train H and B (HGB, deterministic) on ALL rows of the other 9 players
    (every row is a valid causal feature->outcome pair; using others' full timeline
    maximizes data and does not leak P's information)
    evaluate on ALL of P's rows (their whole first-play timeline = the cold-start
    scenario; early samples with thin history are included and visible per-player)
  targets: acc MAE / centered-R2-within-P, lamp ordinal MAE + QWK, BP raw MAE.
  baselines reported alongside: h_acc_mean alone, and the in-protocol B numbers.

Runs on whichever scope samples.parquet currently holds (flag in data.py).
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


def hgb(tr, te, feats, y):
    imp = Imp()
    sc = StandardScaler().fit(imp.fit_transform(tr[feats]))
    Xtr = sc.transform(imp.fit_transform(tr[feats]))
    Xte = sc.transform(imp.transform(te[feats]))
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=3,
                                      random_state=0)
    m.fit(Xtr, y)
    return m.predict(Xte)


def main() -> None:
    df = pd.read_parquet(DS / "samples.parquet")
    scope = "survival" if (df["lamp"].isin([1]).sum() == 0 and
                           (df["acc"] < 50).sum() == 0) else "full"
    players = sorted(df["player"].unique())
    results: dict = {"scope": scope, "n_rows": len(df), "players": players, "lopo": {}}

    for P in players:
        tr, te = df[df["player"] != P], df[df["player"] == P]
        res = {"n_eval": int(len(te))}
        for tag, feats in [("H", H_FEATS), ("B", STAT + H_FEATS),
                           ("mean_only", ["h_acc_mean"])]:
            acc_p = hgb(tr, te, feats, tr["acc"].values) if tag != "mean_only" \
                else te["h_acc_mean"].fillna(tr["h_acc_mean"].mean()).values
            if tag != "mean_only":
                lamp_p = hgb(tr, te, feats, tr["lamp"].values.astype(float))
                bp_p = np.expm1(hgb(tr, te, feats, np.log1p(tr["bp"].values)))
            else:
                lamp_p = bp_p = None  # no causal lamp/BP statistic stored for v0
            a = te["acc"].values - te["acc"].mean()
            b = acc_p - te["acc"].mean()
            den = float(np.sum((a - a.mean()) ** 2))
            res[tag] = {
                "acc_mae": round(mae(te["acc"], acc_p), 3),
                "acc_within_r2": round(float(1 - np.sum((a - b) ** 2) / den), 3) if den else None,
                "lamp_mae": round(mae(te["lamp"], lamp_p), 3) if lamp_p is not None else None,
                "lamp_qwk": round(float(cohen_kappa_score(
                    te["lamp"], np.clip(np.round(lamp_p), 1, 9).astype(int),
                    weights="quadratic", labels=list(range(1, 10)))), 3) if lamp_p is not None else None,
                "bp_mae": round(mae(te["bp"], bp_p), 2) if bp_p is not None else None,
            }
        results["lopo"][P] = res
        print(f"{P:9} n={res['n_eval']:4} | H acc {res['H']['acc_mae']:.2f} "
              f"B acc {res['B']['acc_mae']:.2f} mean-only {res['mean_only']['acc_mae']:.2f} "
              f"| B lamp {res['B']['lamp_mae']:.2f} BP {res['B']['bp_mae']:.1f}")

    # aggregate
    def wavg(tag, key, weight_key="n_eval"):
        num, den = 0.0, 0
        for p in players:
            v = results["lopo"][p][tag][key]
            if v is None:
                continue
            num += v * results["lopo"][p][weight_key]
            den += results["lopo"][p][weight_key]
        return round(num / den, 3) if den else None

    results["summary"] = {
        tag: {"acc_mae": wavg(tag, "acc_mae"), "lamp_mae": wavg(tag, "lamp_mae"),
              "bp_mae": wavg(tag, "bp_mae")}
        for tag in ["H", "B", "mean_only"]}

    json.dump(results, open(OUT / "lopo_results.json", "w"), indent=2)
    print("\nweighted summary:", json.dumps(results["summary"], indent=1))


if __name__ == "__main__":
    main()
