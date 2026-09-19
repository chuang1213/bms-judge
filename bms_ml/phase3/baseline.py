"""Phase 3 v0 baselines: predict first-play accuracy (and first-play failure) from
chart-only (A) vs chart + player history statistics (B).

Models
  M0 global-mean / M1 player-mean / M2 level-mean / M3 additive player+level:
     sanity baselines answering "how far do marginal quantities get us?"
  A  chart features only (26 dims + table/level)         -> answers Baseline A
  B  chart features + player history statistics          -> answers Baseline B
Time split (see data.py): train targets in (q50,q75], test targets after q75.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"

C_FEATS = [c for c in pd.read_parquet(DS / "samples.parquet").columns if c.startswith("c_")]
C_FEATS += ["level_norm", "table_satellite", "table_stella", "table_insane"]
H_FEATS = ["h_level_acc", "h_n_firstplays", "h_acc_mean", "h_acc_std", "h_acc_last10",
           "h_bp_mean", "h_fail_rate", "h_fc_rate", "h_days_since_active",
           "h_plays_last30d", "h_days_span"]


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def main() -> None:
    df = pd.read_parquet(DS / "samples.parquet")
    tr, te = df[df["phase"] == "train"], df[df["phase"] == "test"]
    ytr, yte = tr["acc"].values, te["acc"].values

    results: dict = {"n_train": len(tr), "n_test": len(te)}

    # ---------- sanity / marginal baselines ----------
    gmean = float(np.mean(ytr))
    results["M0_global_mean"] = {"test_mae": mae(yte, gmean), "test_r2": r2(yte, np.full(len(yte), gmean))}
    results["M1_player_mean"] = {"test_mae": mae(yte, te["h_acc_mean"])}
    lvl_mean = tr.groupby(["table", "level"])["acc"].mean()
    results["M2_level_mean"] = {
        "test_mae": mae(yte, [lvl_mean.get((t, l), gmean) for t, l in zip(te["table"], te["level"])])}
    player_off = tr.groupby("player")["acc"].mean() - gmean
    lvl_off = tr.groupby(["table", "level"])["acc"].mean() - gmean
    add = [gmean + player_off.get(p, 0) + lvl_off.get((t, l), 0)
           for p, t, l in zip(te["player"], te["table"], te["level"])]
    results["M3_player+level_additive"] = {"test_mae": mae(yte, add), "test_r2": r2(yte, add)}

    # ---------- feature-based models ----------
    def fit_pred(feats, model):
        imp = SimpleImputerMedian()
        Xtr = imp.fit_transform(tr[feats])
        Xte = imp.transform(te[feats])
        sc = StandardScaler().fit(Xtr)
        model.fit(sc.transform(Xtr), ytr)
        return model.predict(sc.transform(Xte))

    class SimpleImputerMedian:
        def fit(self, X):
            X = np.asarray(X, float)
            self.med = np.nanmedian(X, axis=0)
            self.med = np.where(np.isnan(self.med), 0.0, self.med)
            return self

        def fit_transform(self, X):
            return self.fit(X).transform(X)

        def transform(self, X):
            X = np.asarray(X, float).copy()
            return np.where(np.isnan(X), self.med, X)

    for tag, feats in [("A_chart_only", C_FEATS), ("H_history_only", H_FEATS),
                       ("B_chart+history", C_FEATS + H_FEATS)]:
        results[tag] = {
            "ridge": {"test_mae": mae(yte, fit_pred(feats, Ridge(alpha=10.0)))},
            "hgb": {"test_mae": mae(yte, fit_pred(
                feats, HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                                     max_depth=3, random_state=0)))},
        }
        results[tag]["ridge"]["test_r2"] = None

    # per-player MAE for the strongest simple model vs player-mean baseline
    preds = {}
    for tag in ["M1_player_mean", "A_chart_only", "H_history_only", "B_chart+history"]:
        if tag == "M1_player_mean":
            pred = te["h_acc_mean"].values
        else:
            feats = {"A_chart_only": C_FEATS, "H_history_only": H_FEATS,
                     "B_chart+history": C_FEATS + H_FEATS}[tag]
            pred = fit_pred(feats, HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.06, max_depth=3, random_state=0))
        preds[tag] = pred
        if tag != "M1_player_mean":
            results[tag]["hgb"]["test_r2"] = r2(yte, pred)
        results.setdefault("_per_player_mae", {})[tag] = {
            p: round(mae(g["acc"], pred[te["player"].values == p]), 3)
            for p, g in te.groupby("player")}

    # ---------- first-play failure (lamp==FAILED) AUC ----------
    ftr, fte = (tr["lamp"] == 1).astype(int).values, (te["lamp"] == 1).astype(int).values
    results["_fail_auc"] = {}
    for tag, feats in [("A_chart_only", C_FEATS), ("H_history_only", H_FEATS),
                       ("B_chart+history", C_FEATS + H_FEATS)]:
        imp = SimpleImputerMedian()
        sc = StandardScaler().fit(imp.fit_transform(tr[feats]))
        Xtr = sc.transform(imp.fit_transform(tr[feats]))
        Xte = sc.transform(imp.transform(te[feats]))
        lr = LogisticRegression(max_iter=2000, C=0.5).fit(Xtr, ftr)
        results["_fail_auc"][tag] = round(float(roc_auc_score(fte, lr.predict_proba(Xte)[:, 1])), 4)

    # ---------- same-chart-different-player subset ----------
    multi = te["sha256"].map(te.groupby("sha256")["player"].nunique())
    sub = te[multi >= 2]
    results["_same_chart_multiplayer"] = {
        "n_charts": int(sub["sha256"].nunique()), "n_rows": int(len(sub)),
        "M1_player_mean_mae": mae(sub["acc"], sub["h_acc_mean"]),
        "A_chart_only_mae": mae(sub["acc"], preds["A_chart_only"][multi >= 2]),
        "H_history_only_mae": mae(sub["acc"], preds["H_history_only"][multi >= 2]),
        "B_chart+history_mae": mae(sub["acc"], preds["B_chart+history"][multi >= 2]),
    }

    json.dump(results, open(OUT / "baseline_results.json", "w"), indent=2)

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = {
        "M0 global mean": results["M0_global_mean"]["test_mae"],
        "M1 player mean": results["M1_player_mean"]["test_mae"],
        "M2 level mean": results["M2_level_mean"]["test_mae"],
        "M3 player+level": results["M3_player+level_additive"]["test_mae"],
        "A chart-only (HGB)": results["A_chart_only"]["hgb"]["test_mae"],
        "H history-only (HGB)": results["H_history_only"]["hgb"]["test_mae"],
        "B chart+history (HGB)": results["B_chart+history"]["hgb"]["test_mae"],
    }
    ax.barh(list(bars), list(bars.values()), color=["#bbb"] * 4 + ["#e74c3c", "#2ecc71", "#3498db"])
    ax.invert_yaxis()
    ax.set_xlabel("test MAE (first-play accuracy, 0-100)")
    ax.set_title(f"first-play acc prediction, n_test={len(te)} (time split, 4 players)")
    for i, v in enumerate(bars.values()):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "baseline_mae.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharex=True, sharey=True)
    for ax, p in zip(axes, ["chuang", "muiclac", "tzh", "nanji"]):
        m = te["player"] == p
        ax.scatter(te.loc[m, "acc"], preds["B_chart+history"][m], s=8, alpha=0.4)
        lim = [40, 100]
        ax.plot(lim, lim, "r--", lw=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_title(f"{p} (MAE {mae(te.loc[m,'acc'], preds['B_chart+history'][m]):.1f})")
        ax.set_xlabel("actual acc")
    axes[0].set_ylabel("predicted acc")
    fig.suptitle("B chart+history (HGB): predicted vs actual first-play acc")
    fig.tight_layout()
    fig.savefig(OUT / "baseline_scatter.png", dpi=140)
    plt.close(fig)

    print(json.dumps(results, indent=2, default=str))
    print("plots ->", OUT)


if __name__ == "__main__":
    main()
