"""Phase 3.2: chart-aware history encoder ladder (C0 -> C3) with interaction analysis.

Variants differ ONLY in what each history event carries:
  C0  outcome + time-stamp info          (3 + 2 dt dims)
  C1  + difficulty (table one-hot, level)            == Phase 3.1 v1 event set
  C2  + 26D chart statistics
  C3  + Phase2A T1 representation (64d, mean-pooled windows); target side gets it too

Target chart features are ALWAYS the full 26D stats + difficulty (+ rep for C3), so
the ladder isolates the value of chart context *in the history*.

H / B (hand-crafted statistics baselines, HGB) are re-fit here so every model is
evaluated on identical rows with identical metrics:
  - overall MAE/R2 (acc), ordinal MAE/QWK (lamp), raw+ratio MAE (BP, log1p training)
  - within-player centered R2 (acc): chart-conditioning beyond player strength
  - fixed-chart cross-player MAE: charts first-played by >=2 players in test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"
K = 100
SEED = 0
STAT_COLS = [f"c_{n}" for n in [
    "total_notes", "ln_ratio", "duration_sec", "measures", "initial_bpm", "min_bpm",
    "max_bpm", "bpm_change_count", "stop_count", "stop_total_sec", "lane0_scratch",
    "lane1", "lane2", "lane3", "lane4", "lane5", "lane6", "lane7", "scratch_ratio",
    "avg_nps", "peak_nps_1s", "peak_measure_nps", "chord_count", "chord2_count",
    "chord3plus_count", "jack_count"]]

torch.manual_seed(SEED)
np.random.seed(SEED)


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


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def centered_r2(te: pd.DataFrame, pred: np.ndarray, col: str) -> float:
    """R2 of within-player deviations: does the model explain chart-relative
    variation after removing each player's own mean (player strength)?"""
    a = te[col].values.astype(float) - te.groupby("player")[col].transform("mean").values
    b = pred - te.groupby("player")[col].transform("mean").values
    denom = np.sum((a - a.mean()) ** 2)
    return float(1 - np.sum((a - b) ** 2) / denom) if denom > 0 else float("nan")


def event_matrix(fp_sorted: pd.DataFrame, variant: str, reps: pd.DataFrame | None) -> np.ndarray:
    """Objective ladder: C0 outcome+time, C1 +26D stats, C2 +Phase2A rep."""
    blocks = [np.stack([fp_sorted["lamp"].values / 9.0,
                        fp_sorted["acc"].values / 100.0,
                        np.log1p(fp_sorted["bp"].values)], axis=1)]
    if variant in ("C1", "C2"):
        blocks.append(fp_sorted[STAT_COLS].values.astype(float))
    if variant == "C2":
        R = reps.set_index("sha256").reindex(fp_sorted["sha256"].values)
        mat = R[[f"r{i}" for i in range(64)]].values.astype(float)
        blocks.append(np.nan_to_num(mat))
        blocks.append(R["n_windows"].notna().values.reshape(-1, 1).astype(float))
    return np.concatenate(blocks, axis=1).astype(np.float32)


def build_sequences(samples: pd.DataFrame, fp_sorted: pd.DataFrame,
                    E: np.ndarray, times_days: np.ndarray, pos_by_player: dict):
    """seqs [n, K, d+2], masks [n, K], last_dt [n, 1] — events before each cutoff."""
    seqs, masks, lasts = [], [], []
    dim = E.shape[1] + 2
    for player, cutoff in zip(samples["player"], samples["cutoff"]):
        pos = pos_by_player[player]
        t = times_days[pos]                      # this player's event times
        cut = cutoff.timestamp() / 86400.0
        n_before = int(np.searchsorted(t, cut, side="right"))
        lo, hi = max(0, n_before - K), n_before
        n = hi - lo
        cols = np.zeros((K, dim), dtype=np.float32)
        m = np.zeros(K, dtype=np.float32)
        if n:
            ev = E[pos[lo:hi]]
            dt_prev = np.diff(np.concatenate([[t[lo] - 30], t[lo:hi]]))
            dt_cut = cut - t[lo:hi]
            extra = np.stack([np.log1p(np.maximum(dt_prev, 0)),
                              np.log1p(np.maximum(dt_cut, 0))], axis=1)
            cols[K - n:] = np.concatenate([ev, extra], axis=1)
            m[K - n:] = 1.0
            lasts.append(np.log1p(max(dt_cut[-1], 0)))
        else:
            lasts.append(0.0)
        seqs.append(cols)
        masks.append(m)
    return np.stack(seqs), np.stack(masks), np.asarray(lasts, dtype=np.float32).reshape(-1, 1)


class Model(nn.Module):
    def __init__(self, seq_dim: int, chart_dim: int, heads=("acc", "lamp", "bp")):
        super().__init__()
        self.gru = nn.GRU(seq_dim, 64, batch_first=True)
        self.chart = nn.Sequential(nn.Linear(chart_dim, 64), nn.ReLU())
        self.body = nn.Sequential(nn.Linear(64 + 64 + 1, 64), nn.ReLU())
        self.heads = nn.ModuleDict({"acc": nn.Linear(64, 1), "bp": nn.Linear(64, 1),
                                    "lamp": nn.Linear(64, 9)})

    def forward(self, seq, mask, chart, last_dt):
        h, _ = self.gru(seq)
        idx = (mask.sum(1).long() - 1).clamp(min=0)
        pooled = h[torch.arange(h.size(0)), idx]
        z = self.body(torch.cat([pooled, self.chart(chart), last_dt], dim=1))
        return {k: self.heads[k](z).squeeze(-1) for k in ("acc", "lamp", "bp")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="C0,C1,C2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    global SEED
    SEED = args.seed
    variants = args.variants.split(",")

    fp = pd.read_parquet(DS / "firstplays.parquet")
    reps = pd.read_parquet(DS / "chart_repr_t1.parquet") if "C2" in variants else None
    df = pd.read_parquet(DS / "samples.parquet")
    tr_all, te = df[df["phase"] == "train"], df[df["phase"] == "test"]

    fp_sorted = fp.sort_values(["player", "time"]).reset_index(drop=True)
    times_days = fp_sorted["time"].values.astype("datetime64[s]").astype(np.int64) / 86400.0
    pos_by_player = {p: np.where(fp_sorted["player"].values == p)[0]
                     for p in fp_sorted["player"].unique()}

    chart_cols = list(STAT_COLS)  # objective features only (table levels dropped)
    results: dict = {"n_train": len(tr_all), "n_test": len(te)}

    # temporal val split (per player, last ~10% of train targets)
    tr_all = tr_all.sort_values("time")
    val = tr_all.groupby("player").tail(max(1, len(tr_all) // 40))
    trn = tr_all.drop(val.index)

    def seqs_for(d: pd.DataFrame, variant: str):
        E = event_matrix(fp_sorted, variant, reps)
        return build_sequences(d, fp_sorted, E, times_days, pos_by_player)

    def eval_preds(preds: dict) -> dict:
        out = {
            "acc": {"mae": round(mae(te["acc"], preds["acc"]), 3),
                    "r2": round(r2(te["acc"], preds["acc"]), 3),
                    "centered_r2": round(centered_r2(te, preds["acc"], "acc"), 3),
                    "per_player": {p: round(mae(g["acc"], preds["acc"][te["player"].values == p]), 3)
                                   for p, g in te.groupby("player")}},
            "lamp": {"ord_mae": round(mae(te["lamp"], preds["lamp"]), 3),
                     "qwk": round(float(cohen_kappa_score(
                         te["lamp"], np.clip(np.round(preds["lamp"]), 1, 9).astype(int),
                         weights="quadratic", labels=list(range(1, 10)))), 3)},
            "bp": {"raw_mae": round(mae(te["bp"], preds["bp"]), 2),
                   "raw_medae": round(float(np.median(np.abs(te["bp"].values - preds["bp"]))), 2),
                   "ratio_mae_x1000": round(mae(te["bp"] / te["notes"],
                                                preds["bp"] / te["notes"]) * 1000, 2)},
        }
        multi = te["sha256"].map(te.groupby("sha256")["player"].nunique())
        sub = multi >= 2
        out["_interaction"] = {
            "n_multi_charts": int(te.loc[sub, "sha256"].nunique()),
            "acc_mae_multi": round(mae(te.loc[sub, "acc"], preds["acc"][sub.values]), 3),
            "acc_centered_r2_multi": round(centered_r2(te[sub], preds["acc"][sub.values], "acc"), 3),
            "bp_raw_mae_multi": round(mae(te.loc[sub, "bp"], preds["bp"][sub.values]), 2),
        }
        return out

    # ---------- H / B baselines (HGB) ----------
    H_FEATS = ["h_knn_acc", "h_n_firstplays", "h_acc_mean", "h_acc_std", "h_acc_last10",
               "h_bp_mean", "h_bp_ratio_mean", "h_fail_rate", "h_fc_rate",
               "h_days_since_active", "h_plays_last30d", "h_days_span"]

    def hgb(feats, y):
        imp = Imputer()
        Xtr = StandardScaler().fit_transform(imp.fit_transform(tr_all[feats]))
        Xte = StandardScaler().fit(imp.fit_transform(tr_all[feats])).transform(imp.transform(te[feats]))
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=3,
                                          random_state=SEED)
        m.fit(Xtr, y)
        return m.predict(Xte)

    results["H"] = eval_preds({"acc": hgb(H_FEATS, tr_all["acc"].values),
                               "lamp": hgb(H_FEATS, tr_all["lamp"].values.astype(float)),
                               "bp": np.expm1(hgb(H_FEATS, np.log1p(tr_all["bp"].values)))})
    results["B"] = eval_preds({"acc": hgb(chart_cols + H_FEATS, tr_all["acc"].values),
                               "lamp": hgb(chart_cols + H_FEATS, tr_all["lamp"].values.astype(float)),
                               "bp": np.expm1(hgb(chart_cols + H_FEATS,
                                                  np.log1p(tr_all["bp"].values)))})

    # ---------- C variants (per-target independent encoders — same protocol as 3.1 v1,
    # where the shared multi-task head was shown to hurt acc/BP) ----------
    for variant in variants:
        S_tr, M_tr, D_tr = seqs_for(trn, variant)
        S_va, M_va, D_va = seqs_for(val, variant)
        S_te, M_te, D_te = seqs_for(te, variant)
        seq_dim = S_tr.shape[2]
        chart_dim = len(chart_cols) + (64 if variant == "C2" else 0)

        def chart_mat(d: pd.DataFrame) -> np.ndarray:
            X = d[chart_cols].values.astype(float)
            if variant == "C2":
                R = reps.set_index("sha256").reindex(d["sha256"].values)
                X = np.concatenate([X, np.nan_to_num(R[[f"r{i}" for i in range(64)]].values)], axis=1)
            return X

        Xall = chart_mat(tr_all)                     # may contain NaN for missing reps
        cmu = np.nan_to_num(np.nanmedian(Xall, axis=0))
        csd = np.nan_to_num(np.nanstd(Xall, axis=0))
        csd[csd == 0] = 1.0

        def cmat(d):
            return np.nan_to_num((chart_mat(d) - cmu) / csd).astype(np.float32)

        S_va_t = (torch.tensor(S_va), torch.tensor(M_va), torch.tensor(cmat(val)),
                  torch.tensor(D_va))

        def train_head(head: str):
            torch.manual_seed(SEED)
            model = Model(seq_dim, chart_dim)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            if head == "acc":
                ytr = torch.tensor(trn["acc"].values / 100.0, dtype=torch.float32)
                yva = torch.tensor(val["acc"].values / 100.0, dtype=torch.float32)
            elif head == "bp":
                ytr = torch.tensor(np.log1p(trn["bp"].values), dtype=torch.float32)
                yva = torch.tensor(np.log1p(val["bp"].values), dtype=torch.float32)
            else:
                ytr = torch.tensor(trn["lamp"].values - 1)
                yva = torch.tensor(val["lamp"].values - 1)
            Xt, Mt, DT, CT = (torch.tensor(S_tr), torch.tensor(M_tr),
                              torch.tensor(D_tr), torch.tensor(cmat(trn)))
            lossf = nn.functional.cross_entropy if head == "lamp" else nn.functional.huber_loss
            best, best_state, patience = np.inf, None, 0
            for ep in range(60):
                model.train()
                perm = torch.randperm(len(ytr))
                for i in range(0, len(ytr), 64):
                    b = perm[i:i + 64]
                    out = model(Xt[b], Mt[b], CT[b], DT[b])
                    loss = lossf(out[head], ytr[b])
                    opt.zero_grad(); loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    outv = model(*S_va_t)
                    vloss = float(lossf(outv[head], yva))
                if vloss < best - 1e-4:
                    best, patience = vloss, 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= 6:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)
            return model

        preds = {}
        for head in ("acc", "lamp", "bp"):
            model = train_head(head)
            model.eval()
            with torch.no_grad():
                out = model(torch.tensor(S_te), torch.tensor(M_te),
                            torch.tensor(cmat(te)), torch.tensor(D_te))
            if head == "acc":
                preds["acc"] = out["acc"].numpy() * 100.0
            elif head == "bp":
                bp_log = out["bp"].numpy().clip(0.0, float(np.log1p(tr_all["bp"].max())))
                preds["bp"] = np.expm1(bp_log)
            else:
                preds["lamp"] = torch.softmax(out["lamp"], dim=1).numpy() @ np.arange(1, 10)
        results[variant] = eval_preds(preds)
        print(variant, "| acc", results[variant]["acc"]["mae"],
              "| lamp", results[variant]["lamp"]["ord_mae"],
              "| bp", results[variant]["bp"]["raw_mae"])

    # history length diagnostic (nanji stress test context): events available per
    # player at their train cutoff (capped at K)
    results["_history_len"] = {}
    for p in ["chuang", "muiclac", "tzh", "nanji"]:
        pos = pos_by_player[p]
        t = times_days[pos]
        cut = tr_all.loc[tr_all["player"] == p, "cutoff"].iloc[0].timestamp() / 86400.0
        results["_history_len"][p] = {"events_at_train_cutoff": int(
            min(np.searchsorted(t, cut, side="right"), K))}

    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from chart_repr import feature_manifest
    results["features_used"] = {
        "target_chart": chart_cols + ([f"r{i}" for i in range(64)] if "C2" in variants else []),
        "event_ladder": {"C0": "outcome+time", "C1": "+26D stats", "C2": "+Phase2A rep"},
        "history_stats": H_FEATS,
        "uses_difficulty_table_features": False,
    }
    suffix = "" if SEED == 0 else f"_s{SEED}"
    json.dump(results, open(OUT / f"phase32_obj_results{suffix}.json", "w"), indent=2, default=str)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
