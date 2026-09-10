"""Phase 3.6: is k=20 the right neighbourhood for `h_knn_acc`?

`h_knn_acc` (mean acc of the k nearest previously-played charts in the standardized
27-dim stat space) is the single most valuable history feature and the ONLY
chart-conditioned one before the response profile. Its k has been 20 since data.py was
written - PHASE3_3_READINESS §5 lists "k=20 拍脑袋" as outstanding debt and asks for a
k in {10,20,50} sensitivity check.

This recomputes the feature for several k with data.py's exact algorithm (strictly
prior rows only, chunked distances, acc averaged WITHOUT dropping unknown-acc
neighbours - so the variants differ from the shipped column only in k) and refits B on
each, so the comparison is apples-to-apples on one row set.

Also reports the k=20 recomputation's agreement with the shipped column as a
reproducibility check: it must match to floating-point noise, otherwise the two
implementations have drifted and the sweep would be measuring the wrong thing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

from chart_repr import (HISTORY_FEATURES, HISTORY_RESPONSE_COLS,
                        OBJECTIVE_STAT_COLS)
from common import centered_r2, hgb_fit_predict, load_samples, mae

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"
KS = [5, 10, 20, 50, 100]


def knn_feature(fp: pd.DataFrame, k: int) -> np.ndarray:
    """data.py's build_history_features kNN block, verbatim, for an arbitrary k."""
    stat_cols = [c for c in fp.columns if c.startswith("c_")]
    scaler = StandardScaler().fit(fp[stat_cols].values)
    Z = np.nan_to_num(scaler.transform(fp[stat_cols].values)).astype(np.float64)
    acc = fp["acc"].values
    out = np.full(len(fp), np.nan)
    for _p, idx in fp.groupby("player", sort=False).indices.items():
        pos = np.asarray(idx)
        r = len(pos)
        Zi = Z[pos]
        lo0 = 0
        while lo0 < r:
            hi0 = min(lo0 + 256, r)
            D = np.sqrt(((Zi[lo0:hi0, None, :] - Zi[None, :hi0, :]) ** 2).sum(-1))
            for j in range(lo0, hi0):
                drow = D[j - lo0, :j]
                if len(drow) == 0:
                    continue
                kk = min(k, len(drow))
                near = np.argpartition(drow, kk - 1)[:kk]
                out[pos[j]] = np.mean(acc[pos][near])
            lo0 = hi0
    return out


def main() -> None:
    fp = pd.read_parquet(DS / "firstplays.parquet").sort_values(
        ["player", "time"], kind="stable").reset_index(drop=True)
    shipped = fp["h_knn_acc"].values.copy()

    variants = {}
    for k in KS:
        v = knn_feature(fp, k)
        variants[k] = v
        if k == 20:
            both = ~np.isnan(v) & ~np.isnan(shipped)
            print(f"k=20 recompute vs shipped: n={int(both.sum())} "
                  f"max|diff|={float(np.nanmax(np.abs(v[both] - shipped[both]))):.2e}")

    df = load_samples()
    tr, te = df[df["phase"] == "train"].copy(), df[df["phase"] == "test"].copy()
    key = fp[["player", "sha256"]].copy()
    for k in KS:
        key[f"knn{k}"] = variants[k]
    # one sha256 can belong to several players -> key on (player, sha256)
    tr = tr.merge(key, on=["player", "sha256"], how="left")
    te = te.merge(key, on=["player", "sha256"], how="left")

    base = [c for c in HISTORY_FEATURES if c != "h_knn_acc"]
    # the same sweep must also be run WITH the response profile, because that is the
    # configuration we would actually ship - the optimum need not transfer.
    hr = pd.read_parquet(DS / "history_response.parquet")
    prof = HISTORY_RESPONSE_COLS
    tr = tr.merge(hr[["player", "sha256"] + prof], on=["player", "sha256"], how="left")
    te = te.merge(hr[["player", "sha256"] + prof], on=["player", "sha256"], how="left")

    results: dict = {"n_train": len(tr), "n_test": len(te), "ks": KS, "sweep": {},
                     "sweep_with_profile": {}}
    for tag, extra in (("sweep", []), ("sweep_with_profile", prof)):
        print(f"\n--- {tag} ---\n{'k':>5}{'acc':>8}{'cR2':>9}{'lamp':>8}{'QWK':>7}{'BP':>8}")
        for k in KS:
            feats = OBJECTIVE_STAT_COLS + base + [f"knn{k}"] + extra
            acc_p = hgb_fit_predict(tr, te, feats, tr["acc"].values)
            lamp_p = hgb_fit_predict(tr, te, feats, tr["lamp"].values.astype(float))
            bp_p = np.expm1(hgb_fit_predict(tr, te, feats, np.log1p(tr["bp"].values)))
            r = {"acc_mae": round(mae(te["acc"], acc_p), 3),
                 "cR2": round(centered_r2(te, acc_p, "acc"), 3),
                 "lamp": round(mae(te["lamp"], lamp_p), 3),
                 "qwk": round(float(cohen_kappa_score(
                     te["lamp"], np.clip(np.round(lamp_p), 1, 9).astype(int),
                     weights="quadratic", labels=list(range(1, 10)))), 3),
                 "bp": round(mae(te["bp"], bp_p), 1)}
            results[tag][str(k)] = r
            print(f"{k:5d}{r['acc_mae']:8.3f}{r['cR2']:+9.3f}{r['lamp']:8.3f}"
                  f"{r['qwk']:7.3f}{r['bp']:8.1f}")
    json.dump(results, open(OUT / "knn_k_sensitivity.json", "w"), indent=2, default=str)
    print("saved ->", OUT / "knn_k_sensitivity.json")


if __name__ == "__main__":
    main()
