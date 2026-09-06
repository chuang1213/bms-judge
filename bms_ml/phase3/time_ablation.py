"""Phase 3.4: Time Ablation / Timestamp Necessity Study (independent ablation).

Research question: how much does player-state modelling actually depend on
timestamp information? (Protocol per PROTOCOL.md; the main Phase 3 results are
not modified.)

Variants — identical rows, targets, split, imputer, HGB budget; only the history
features differ. History MEMBERSHIP (events before the cutoff) is part of sample
construction and identical everywhere; what is ablated is the use of time WITHIN
the history:
  H_time        current H (12 features incl. recency/calendar terms)
  H_no_time     set statistics only: no absolute date, no recency, no windows,
                no chronological position. 8 features.
  H_order_only  H_no_time + h_acc_last10 (play ORDER from timestamps — a weakened
                form of temporal information, declared as such)
  H_masked_XX   timestamps of a random XX% of each player's scorelog rows are
                hidden and the 4 time features are recomputed from the remaining
                dates; membership unchanged. XX=100 must collapse to H_no_time
                (sanity anchor).

Metrics: acc MAE / centered R2, lamp ordinal MAE / QWK, BP raw / ratio; overall,
per player, per difficulty region (SL / ST0-3 / ST4-7 / ST8+ / insane — coordinate
only, never a feature), fixed-chart cross-player subset.

Growth diagnostics: per-player drift (Spearman acc~time), drift-bucketed
H_time-minus-H_no_time gap, and h_acc_mean vs h_acc_last10 univariate comparison.

All HGB fits run with 3 seeds; results reported as mean±std.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from common import (add_region, centered_r2, hgb_fit_predict, load_firstplays,
                    load_samples, mae, r2)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "玩家资料"
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"
SEEDS = [0, 1, 2]

SET_FEATURES = ["h_knn_acc", "h_n_firstplays", "h_acc_mean", "h_acc_std",
                "h_bp_mean", "h_bp_ratio_mean", "h_fail_rate", "h_fc_rate"]
TIME_FEATURES = ["h_acc_last10", "h_days_since_active", "h_plays_last30d", "h_days_span"]
FEATURES = {
    "H_time": SET_FEATURES + TIME_FEATURES,
    "H_no_time": list(SET_FEATURES),
    "H_order_only": SET_FEATURES + ["h_acc_last10"],
}


def load_log_counts() -> pd.DataFrame:
    import json
    roster = json.load(open(Path(__file__).resolve().parent / "players.json", encoding="utf-8"))
    rows = []
    for name, e in roster["players"].items():
        if not e.get("include"):
            continue
        con = sqlite3.connect(ROOT / e["dir"] / "scorelog.db")
        df = pd.read_sql_query("SELECT date FROM scorelog", con)
        con.close()
        rows.append(pd.DataFrame({"player": name,
                                  "time": pd.to_datetime(df["date"], unit="s")}))
    return pd.concat(rows, ignore_index=True)


def masked_time_features(samples: pd.DataFrame, fp: pd.DataFrame,
                         log: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Recompute the 4 time features with a random `frac` of each player's scorelog
    dates hidden. Membership (events before cutoff) unchanged; masked events lose
    their DATES for feature purposes only.

    Vectorized: per player, precompute visible (unmasked) arrays sorted by time,
    then answer each sample with searchsorted. At frac=1.0 all four features are
    constant (NaN -> imputer median), collapsing to H_no_time by construction."""
    rng = np.random.RandomState(seed)
    prep = {}
    for player in log["player"].unique():
        lt = log.loc[log["player"] == player, "time"].values.astype("datetime64[s]").astype(np.int64)
        order = np.argsort(lt, kind="stable")
        lt_sorted = lt[order]
        hidden_idx = rng.choice(len(lt), size=int(round(frac * len(lt))), replace=False)
        hidden_times = lt[hidden_idx]
        vis_log_mask = ~np.isin(lt_sorted, hidden_times)
        lt_vis = lt_sorted[vis_log_mask]                      # visible log times, sorted
        f = fp[fp["player"] == player]
        ft = f["time"].values.astype("datetime64[s]").astype(np.int64)
        acc = f["acc"].values
        f_order = np.argsort(ft, kind="stable")
        ft_s, acc_s = ft[f_order], acc[f_order]
        f_vis = ~np.isin(ft_s, hidden_times)
        vf_t_all = ft_s[f_vis]                                # visible fp, any acc
        known = f_vis & ~np.isnan(acc)
        prep[player] = (ft_s[known], acc_s[known], vf_t_all, lt_vis)
    out = np.full((len(samples), 4), np.nan)
    # boundary = the target chart's own first-play time; history is strictly prior
    for i, (player, t_end) in enumerate(zip(samples["player"], samples["time"])):
        vf_t_known, vf_acc, vf_t_all, lt_vis = prep[player]
        cut = int(t_end.timestamp())
        n = np.searchsorted(vf_t_known, cut)
        if n:
            out[i, 0] = vf_acc[max(0, n - 10):n].mean()
            out[i, 3] = (cut - vf_t_all[0]) / 86400.0 if vf_t_all[0] <= cut else np.nan
        hi = np.searchsorted(lt_vis, cut, side="right")
        lo = np.searchsorted(lt_vis, cut - 30 * 86400, side="left")
        out[i, 2] = float(hi - lo)
        if hi:
            out[i, 1] = (cut - lt_vis[hi - 1]) / 86400.0
    return pd.DataFrame(out, columns=TIME_FEATURES, index=samples.index)


def main() -> None:
    df = add_region(load_samples())
    fp = load_firstplays()
    log = load_log_counts()
    tr, te = df[df["phase"] == "train"], df[df["phase"] == "test"]

    results: dict = {"n_train": len(tr), "n_test": len(te), "seeds": SEEDS}

    def hgb_pred(feats, y, seed, trX=tr, teX=te):
        return hgb_fit_predict(trX, teX, feats, y, seed=seed)

    def evaluate(preds: dict, tag: str) -> dict:
        acc_p, lamp_p, bp_p = preds["acc"], preds["lamp"], preds["bp"]
        out = {
            "acc": {"mae": round(mae(te["acc"], acc_p), 3),
                    "r2": round(r2(te["acc"], acc_p), 3),
                    "centered_r2": round(centered_r2(te, acc_p), 3)},
            "lamp": {"ord_mae": round(mae(te["lamp"], lamp_p), 3),
                     "qwk": round(float(cohen_kappa_score(
                         te["lamp"], np.clip(np.round(lamp_p), 1, 9).astype(int),
                         weights="quadratic", labels=list(range(1, 10)))), 3)},
            "bp": {"raw_mae": round(mae(te["bp"], bp_p), 2),
                   "ratio_mae_x1000": round(mae(te["bp"] / te["notes"],
                                                bp_p / te["notes"]) * 1000, 2)},
        }
        out["_per_player"] = {p: {"acc_mae": round(mae(g["acc"], acc_p[te["player"].values == p]), 3),
                                  "bp_mae": round(mae(g["bp"], bp_p[te["player"].values == p]), 2)}
                              for p, g in te.groupby("player")}
        multi = (te["sha256"].map(te.groupby("sha256")["player"].nunique()) >= 2).values
        out["_fixed_chart"] = {"n_charts": int(te.loc[multi, "sha256"].nunique()),
                               "acc_mae": round(mae(te["acc"][multi], acc_p[multi]), 3),
                               "bp_mae": round(mae(te["bp"][multi], bp_p[multi]), 2)}
        regions = {}
        for reg, sub in te.groupby("region"):
            m = sub.index.values
            pos = np.where(te.index.isin(m))[0]
            regions[reg] = {
                "n": int(len(sub)),
                "acc_mae": round(mae(sub["acc"], acc_p[pos]), 3),
                "centered_r2": round(centered_r2(sub, acc_p[pos]), 3),
                "lamp_mae": round(mae(sub["lamp"], lamp_p[pos]), 3),
                "bp_mae": round(mae(sub["bp"], bp_p[pos]), 2),
            }
        out["_regions"] = regions
        return out

    runs: dict = {}

    # static variants
    for tag, feats in FEATURES.items():
        runs[tag] = []
        for seed in SEEDS:
            runs[tag].append({
                "acc": hgb_pred(feats, tr["acc"].values, seed),
                "lamp": hgb_pred(feats, tr["lamp"].values.astype(float), seed),
                "bp": np.expm1(hgb_pred(feats, np.log1p(tr["bp"].values), seed)),
            })

    # masked variants (features recomputed per frac/seed; 3 feature seeds x 3 hgb seeds)
    for frac in [0.25, 0.5, 0.75, 1.0]:
        tag = f"H_masked_{int(frac*100)}"
        runs[tag] = []
        for fseed in SEEDS:
            mf = masked_time_features(df, fp, log, frac, fseed)
            # drop the original time columns so the model sees ONLY masked versions
            d2 = pd.concat([df.drop(columns=TIME_FEATURES), mf], axis=1)
            tr2, te2 = d2[d2["phase"] == "train"], d2[d2["phase"] == "test"]
            feats = FEATURES["H_time"]  # same 12 columns, 4 of them masked
            for seed in SEEDS:
                runs[tag].append({
                    "acc": hgb_pred(feats, tr2["acc"].values, seed, tr2, te2),
                    "lamp": hgb_pred(feats, tr2["lamp"].values.astype(float), seed, tr2, te2),
                    "bp": np.expm1(hgb_pred(feats, np.log1p(tr2["bp"].values), seed, tr2, te2)),
                })

    def agg(tag):
        n = len(runs[tag])
        mean_preds = {k: np.mean([r[k] for r in runs[tag]], axis=0) for k in ("acc", "lamp", "bp")}
        out = evaluate(mean_preds, tag)
        # seed spread on the headline metrics
        out["_seed_std"] = {
            "acc_mae": round(float(np.std([mae(te["acc"], r["acc"]) for r in runs[tag]])), 3),
            "lamp_mae": round(float(np.std([mae(te["lamp"], r["lamp"]) for r in runs[tag]])), 3),
            "bp_mae": round(float(np.std([mae(te["bp"], r["bp"]) for r in runs[tag]])), 2),
        }
        out["_n_runs"] = n
        return out

    for tag in runs:
        results[tag] = agg(tag)

    # ---------------- growth / drift diagnostics ----------------
    diag: dict = {}
    drift = {}
    for player, g in fp.groupby("player"):
        g = g.sort_values("time").dropna(subset=["acc"])
        rho = spearmanr(g["acc"], g["time"].astype(np.int64)).statistic
        drift[player] = round(float(rho), 3)
    diag["per_player_drift_spearman_acc_time"] = drift
    # H_time vs H_no_time gap per player
    acc_t = np.mean([r["acc"] for r in runs["H_time"]], axis=0)
    acc_n = np.mean([r["acc"] for r in runs["H_no_time"]], axis=0)
    gap = {}
    for p, g in te.groupby("player"):
        m = te["player"].values == p
        gap[p] = {"drift": drift.get(p), "mae_H_time": round(mae(g["acc"], acc_t[m]), 3),
                  "mae_H_no_time": round(mae(g["acc"], acc_n[m]), 3),
                  "gap": round(mae(g["acc"], acc_n[m]) - mae(g["acc"], acc_t[m]), 3)}
    diag["time_gap_by_player"] = gap
    # univariate: mean vs last10 as single-feature predictors of future acc
    uni = {}
    for name, feats in [("h_acc_mean", ["h_acc_mean"]), ("h_acc_last10", ["h_acc_last10"]),
                        ("mean+last10", ["h_acc_mean", "h_acc_last10"])]:
        p = hgb_pred(feats, tr["acc"].values, 0)
        uni[name] = {"acc_mae": round(mae(te["acc"], p), 3)}
    diag["mean_vs_last10_univariate"] = uni
    results["diagnostics"] = diag

    results["features_used"] = {
        "H_time": FEATURES["H_time"], "H_no_time": FEATURES["H_no_time"],
        "H_order_only": FEATURES["H_order_only"],
        "uses_timestamp": {"H_time": True, "H_no_time": False,
                           "H_order_only": "order_only",
                           "H_masked_*": f"masked fraction of dates (seeds {SEEDS})"},
        "history_membership": "events <= cutoff (protocol-fixed, identical for all variants)",
    }

    json.dump(results, open(OUT / "time_ablation_results.json", "w"), indent=2, default=str)

    # console summary
    print(f"{'variant':14}{'acc':>8}{'cR2':>8}{'lamp':>8}{'QWK':>7}{'BP':>8}{'ratio1k':>9}")
    for tag in ["H_time", "H_order_only", "H_no_time", "H_masked_25",
                "H_masked_50", "H_masked_75", "H_masked_100"]:
        d = results[tag]
        print(f"{tag:14}{d['acc']['mae']:8.2f}{d['acc']['centered_r2']:8.3f}"
              f"{d['lamp']['ord_mae']:8.3f}{d['lamp']['qwk']:7.3f}"
              f"{d['bp']['raw_mae']:8.1f}{d['bp']['ratio_mae_x1000']:9.1f}")
    print("\ndrift:", diag["per_player_drift_spearman_acc_time"])
    print("time gap by player:", json.dumps(diag["time_gap_by_player"], indent=1))
    print("univariate:", diag["mean_vs_last10_univariate"])


if __name__ == "__main__":
    main()
