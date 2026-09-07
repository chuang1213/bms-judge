"""Phase 3.5 (task 1b): learned player-state AS FEATURES, not as an end-to-end model.

c0_fewshot.py (seed 0) showed the end-to-end GRU conditioning loses the sparse
regime badly (k=1: 27.2 vs M2 23.9) while winning at k=200 (10.44 vs 11.11).
Diagnosis: the bottleneck is the chart pathway (a small MLP is far weaker than
HGB on 27-dim tabular data) and the joint data budget, not the state encoder —
held-out validation reaches B-level MAE when sequences are full.

So this script keeps M2's HGB (which is strong on tabular) and ADDS the learned
64-dim state as extra features:

    HGB(27 chart + 9 HIST_FEW + 64 learned-state)  vs  M2 = HGB(27 + 9 HIST_FEW)

The state encoder is trained per held-out player (LOPO, leakage-safe) on the acc
head only (one head keeps the cost at 1/3 of c0_fewshot); its pooled GRU state is
then extracted for every evaluation row and appended to the HGB feature matrix.
Everything else — LOPO, few-shot k curve, prefix-only histories — matches
transfer_eval / c0_fewshot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from chart_repr import HISTORY_FEW_FEATURES, OBJECTIVE_STAT_COLS
from common import Imputer, hgb_fit_predict, load_firstplays, mae
from c_chart_aware import event_matrix
from c0_fewshot import build_seqs, train_head
from transfer_eval import prefix_hist_features

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"
KS = [0, 1, 5, 10, 20, 50, 100, 200]
W = 150
K = 100
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def pooled_state(model, S, M, C):
    """64-dim GRU pooled state for each row."""
    with torch.no_grad():
        h, _ = model.gru(torch.as_tensor(S, dtype=torch.float32, device=DEV))
        idx = (torch.as_tensor(M, device=DEV).sum(1).long() - 1).clamp(min=0)
        return torch.cat(
            [h[torch.arange(h.size(0)), idx],
             model.chart(torch.as_tensor(C, dtype=torch.float32, device=DEV))],
            dim=1).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    fp = pd.read_parquet(DS / "firstplays.parquet").sort_values(
        ["player", "time"]).reset_index(drop=True)
    times_days = fp["time"].values.astype("datetime64[s]").astype(np.int64) / 86400.0
    E = event_matrix(fp, "C0", None)
    pos_by_player = {p: np.where(fp["player"].values == p)[0]
                     for p in fp["player"].unique()}
    players = sorted(fp["player"].unique())

    print("building augmented training sequences ...", flush=True)
    SEQ = {p: build_seqs(pos_by_player[p], pos_by_player[p], times_days, E,
                         rng=np.random.RandomState(seed * 1000 + abs(hash(p)) % 997))
           for p in players}

    HGB_FEATS_M2 = OBJECTIVE_STAT_COLS + HISTORY_FEW_FEATURES
    results: dict = {"seed": seed, "ks": KS, "lopo": {}}

    for D in players:
        # few-shot history features MUST be recomputed from the prefix only
        # (transfer_eval semantics); the data.py h_* of D's rows include events
        # after the prefix and would leak deployment-unavailable information.
        scaler = StandardScaler().fit(
            fp[fp["player"] != D][OBJECTIVE_STAT_COLS].values)

        tr_pos = np.concatenate([pos_by_player[p] for p in players if p != D])
        rows_tr = fp.loc[tr_pos]
        S_tr = np.concatenate([SEQ[p][0] for p in players if p != D])
        M_tr = np.concatenate([SEQ[p][1] for p in players if p != D])
        assert len(rows_tr) == S_tr.shape[0]

        g = rows_tr.groupby("player", sort=False)
        val_idx = np.concatenate([gg.index[-max(1, int(round(len(gg) * 0.1))):]
                                  for _, gg in g])
        is_val = rows_tr.index.isin(val_idx)
        imp = Imputer()
        C_all = imp.fit_transform(rows_tr[OBJECTIVE_STAT_COLS].values)
        sc = StandardScaler().fit(C_all)

        model = train_head("acc", S_tr[~is_val], M_tr[~is_val], sc.transform(C_all[~is_val]),
                           rows_tr["acc"].values[~is_val] / 100.0,
                           S_tr[is_val], M_tr[is_val], sc.transform(C_all[is_val]),
                           rows_tr["acc"].values[is_val] / 100.0,
                           len(OBJECTIVE_STAT_COLS), E.shape[1], seed)

        # state for the TRAINING rows too, so HGB sees the same schema at train/eval
        Z_tr = pooled_state(model, S_tr, M_tr, C_all)

        mine = fp[fp["player"] == D].sort_values("time").reset_index(drop=True)
        pos_D = pos_by_player[D]
        curve = []
        for k in KS:
            if k >= len(mine) - 1:
                continue
            targets = mine.iloc[k:k + W]
            if not len(targets):
                continue
            pool = pos_D[:k]
            tt = targets["time"].values.astype("datetime64[s]").astype(np.int64) / 86400.0
            assert k == 0 or (tt > times_days[pool[-1]]).all()
            S_te = np.zeros((len(targets), K, E.shape[1]), dtype=np.float32)
            M_te = np.zeros((len(targets), K), dtype=np.float32)
            if k:
                n = min(K, k)
                S_te[:, K - n:] = E[pool[-n:]][None, :, :]
                M_te[:, K - n:] = 1.0
            C_te = sc.transform(imp.transform(targets[OBJECTIVE_STAT_COLS].values))
            Z_te = pooled_state(model, S_te, M_te, C_te)

            # prefix-limited history features (transfer_eval semantics, k=0 -> NaN)
            if k:
                hf = prefix_hist_features(mine.iloc[:k], targets, scaler)
            else:
                hf = pd.DataFrame(np.nan, index=targets.index,
                                  columns=HISTORY_FEW_FEATURES)
            tr_hgb = pd.DataFrame(np.hstack([C_all, rows_tr[HISTORY_FEW_FEATURES].values, Z_tr]),
                                  columns=[f"f{i}" for i in range(C_all.shape[1] + 9 + Z_tr.shape[1])])
            te_hgb = pd.DataFrame(
                np.hstack([C_te, hf[HISTORY_FEW_FEATURES].values, Z_te]),
                columns=tr_hgb.columns)
            tr_hgb["acc"] = rows_tr["acc"].values
            te_hgb["acc"] = targets["acc"].values

            p_m2 = hgb_fit_predict(tr_hgb, te_hgb, [f"f{i}" for i in range(C_all.shape[1] + 9)],
                                   tr_hgb["acc"].values, seed=seed)
            p_st = hgb_fit_predict(tr_hgb, te_hgb, list(tr_hgb.columns[:-1]),
                                   tr_hgb["acc"].values, seed=seed)
            curve.append({"k": k, "n_eval": int(len(targets)),
                          "M2_acc": round(mae(targets["acc"], p_m2), 3),
                          "M2state_acc": round(mae(targets["acc"], p_st), 3)})
        results["lopo"][D] = {"n_events": int(len(mine)), "fewshot": curve}
        print(D, {c["k"]: c["M2state_acc"] for c in curve}, flush=True)

    agg = []
    for k in KS:
        rows = [c for D in players for c in results["lopo"][D]["fewshot"] if c["k"] == k]
        if not rows:
            continue
        w = np.array([c["n_eval"] for c in rows], float)
        agg.append({"k": k, "n_eval": int(w.sum()),
                    "M2_acc": round(float(np.average([c["M2_acc"] for c in rows], weights=w)), 3),
                    "M2state_acc": round(float(np.average([c["M2state_acc"] for c in rows], weights=w)), 3)})
    results["fewshot_aggregate"] = agg
    json.dump(results, open(OUT / f"c0_state_hgb_s{seed}.json", "w"), indent=2, default=str)
    print("\nlearned-state-as-features vs M2 (seed %d):" % seed)
    for e in agg:
        print(f"  k={e['k']:3}: M2 {e['M2_acc']:.2f}  +state {e['M2state_acc']:.2f}  "
              f"delta {e['M2state_acc'] - e['M2_acc']:+.2f}")


if __name__ == "__main__":
    main()
