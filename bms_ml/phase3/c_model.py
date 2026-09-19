"""[DEPRECATED 2026-09-03] uses difficulty-table-derived features (h_level_acc, level_norm, table one-hots) which are removed from the current pipeline; kept for history. Use compare_nolevel.py / c_chart_aware.py instead. See PROTOCOL.md.

Phase 3.1 model C: history encoder (GRU) + chart features -> 3 performance heads.

Question (brief section 6/8): can a learned player state from the raw first-play
event sequence beat hand-crafted history statistics (H), and does a SHARED
multi-task encoder (3 heads) beat three INDEPENDENT encoders?

Sequence per sample: the player's last K=100 first-play events strictly before the
sample's cutoff (no target leakage; target event is after cutoff by construction).
Event features: table one-hot(3), level_norm, lamp/9, acc/100, log1p(bp), log1p(days
since previous event), log1p(days to cutoff) => 9 dims + validity mask.

Heads: acc regression (standardized), lamp 9-class CE (ordinal reported), BP
regression on log1p (standardized). Run twice: --mode independent (one head per
run) and --mode shared (joint loss).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"
K = 100
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)


def build_sequences(samples: pd.DataFrame, fp: pd.DataFrame):
    """X [n, K, 9], mask [n, K] — last K first-play events before each sample's cutoff."""
    fp = fp.sort_values("time")
    by_player = {p: g for p, g in fp.groupby("player")}
    seqs, masks, last_dt = [], [], []
    for player, cutoff in zip(samples["player"], samples["cutoff"]):
        g = by_player[player]
        ev = g[g["time"] <= cutoff].tail(K)
        t = ev["time"].values.astype("datetime64[s]").astype(np.int64) / 86400.0
        dt_prev = np.diff(t, prepend=t[0] - 30) if len(t) else np.array([])
        dt_cut = cutoff.timestamp() / 86400.0 - t if len(t) else np.array([])
        cols = np.stack([
            (ev["table"] == "satellite").values.astype(float),
            (ev["table"] == "stella").values.astype(float),
            (ev["table"] == "insane").values.astype(float),
            ev["level_norm"].values,
            ev["lamp"].values / 9.0,
            ev["acc"].values / 100.0,
            np.log1p(ev["bp"].values),
            np.log1p(np.maximum(dt_prev, 0)),
            np.log1p(np.maximum(dt_cut, 0)),
        ], axis=1)
        n = len(cols)
        if n == 0:
            seqs.append(np.zeros((K, 9), dtype=np.float32))
            masks.append(np.zeros(K, dtype=np.float32))
            last_dt.append(np.nan)
            continue
        pad = np.zeros((K - n, 9), dtype=np.float32)
        seqs.append(np.concatenate([pad, cols], axis=0).astype(np.float32))
        m = np.zeros(K, dtype=np.float32); m[K - n:] = 1.0
        masks.append(m)
        last_dt.append(np.log1p(max(dt_cut[-1], 0)))
    return (np.stack(seqs), np.stack(masks),
            np.asarray(last_dt, dtype=np.float32).reshape(-1, 1))


class Model(nn.Module):
    def __init__(self, n_chart_feats: int, heads: list[str]):
        super().__init__()
        self.gru = nn.GRU(9, 64, batch_first=True)
        self.chart = nn.Sequential(nn.Linear(n_chart_feats, 64), nn.ReLU())
        self.body = nn.Sequential(nn.Linear(64 + 64 + 1, 64), nn.ReLU())
        self.heads = nn.ModuleDict({
            "acc": nn.Linear(64, 1),
            "bp": nn.Linear(64, 1),
            "lamp": nn.Linear(64, 9),
        })
        self.active = heads

    def forward(self, seq, mask, chart, last_dt):
        h, _ = self.gru(seq)
        idx = (mask.sum(1).long() - 1).clamp(min=0)
        pooled = h[torch.arange(h.size(0)), idx]          # hidden at last valid event
        z = self.body(torch.cat([pooled, self.chart(chart), last_dt], dim=1))
        return {k: self.heads[k](z).squeeze(-1) for k in self.active}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["independent", "shared"], required=True)
    args = ap.parse_args()

    fp = pd.read_parquet(DS / "firstplays.parquet")
    df = pd.read_parquet(DS / "samples.parquet")
    chart_cols = [c for c in df.columns if c.startswith("c_")] + \
        ["level_norm", "table_satellite", "table_stella", "table_insane"]
    tr, te = df[df["phase"] == "train"], df[df["phase"] == "test"]

    Xtr, Mtr, Dtr = build_sequences(tr, fp)
    Xte, Mte, Dte = build_sequences(te, fp)
    chart_mu, chart_sd = tr[chart_cols].mean(), tr[chart_cols].std().replace(0, 1)
    acc_mu, acc_sd = tr["acc"].mean(), tr["acc"].std()
    bp_mu, bp_sd = np.log1p(tr["bp"].values).mean(), np.log1p(tr["bp"].values).std()
    CT = lambda d: torch.tensor(((d[chart_cols] - chart_mu) / chart_sd).values.astype(np.float32))
    Xt, Mt, DT, CTtr = torch.tensor(Xtr), torch.tensor(Mtr), torch.tensor(Dtr), CT(tr)
    Xe, Me, DE, CTte = torch.tensor(Xte), torch.tensor(Mte), torch.tensor(Dte), CT(te)

    heads = ["acc", "lamp", "bp"]
    results = {"mode": args.mode}

    def make_model(active):
        torch.manual_seed(SEED)
        return Model(len(chart_cols), active)

    # temporal val split inside train (per player, last 25% of train targets) for early stop
    tr = tr.sort_values("time")
    val = tr.groupby("player").tail(max(1, int(len(tr) // 40)))
    trn = tr.drop(val.index)

    # bounded targets: acc/100, log1p(bp) raw scale (no z-scaling), lamp CE
    def tgt(d, head):
        if head == "acc":
            return torch.tensor(d["acc"].values / 100.0, dtype=torch.float32)
        if head == "bp":
            return torch.tensor(np.log1p(d["bp"].values), dtype=torch.float32)
        return torch.tensor(d["lamp"].values - 1, dtype=torch.long)

    def train(active):
        torch.manual_seed(SEED)
        model = make_model(active)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        ys, yv = {h: tgt(trn, h) for h in active}, {h: tgt(val, h) for h in active}
        CTn, CTv = CT(trn), CT(val)
        Xtn, Mtn, DTn = torch.tensor(build_sequences(trn, fp)[0]), torch.tensor(
            build_sequences(trn, fp)[1]), torch.tensor(build_sequences(trn, fp)[2])
        Xv, Mv, DTv = torch.tensor(build_sequences(val, fp)[0]), torch.tensor(
            build_sequences(val, fp)[1]), torch.tensor(build_sequences(val, fp)[2])
        best, best_state, patience = np.inf, None, 0
        for ep in range(60):
            model.train()
            perm = torch.randperm(len(trn))
            for i in range(0, len(trn), 64):
                b = perm[i:i + 64]
                out = model(Xtn[b], Mtn[b], CTn[b], DTn[b])
                loss = sum(
                    nn.functional.cross_entropy(out[h], ys[h][b]) if h == "lamp"
                    else nn.functional.huber_loss(out[h], ys[h][b])
                    for h in active) / len(active)
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                outv = model(Xv, Mv, CTv, DTv)
                vloss = sum(
                    nn.functional.cross_entropy(outv[h], yv[h]) if h == "lamp"
                    else nn.functional.huber_loss(outv[h], yv[h])
                    for h in active) / len(active)
            if float(vloss) < best - 1e-4:
                best, patience = float(vloss), 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= 6:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    if args.mode == "shared":
        model = train(heads)
        model.eval()
        with torch.no_grad():
            out = model(Xe, Me, CTte, DE)
        acc_pred = out["acc"].numpy() * 100.0
        bp_log = out["bp"].numpy().clip(0.0, float(np.log1p(tr["bp"].max())))
        bp_pred = np.expm1(bp_log)
        lamp_pred = torch.softmax(out["lamp"], dim=1).numpy() @ np.arange(1, 10)
    else:
        acc_pred = bp_pred = lamp_pred = None
        for head in ["acc", "bp", "lamp"]:
            model = train([head])  # independent: fresh encoder per target
            model.eval()
            with torch.no_grad():
                out = model(Xe, Me, CTte, DE)
            if head == "acc":
                acc_pred = out["acc"].numpy() * 100.0
            elif head == "bp":
                bp_log = out["bp"].numpy().clip(0.0, float(np.log1p(tr["bp"].max())))
                bp_pred = np.expm1(bp_log)
            else:
                lamp_pred = torch.softmax(out["lamp"], dim=1).numpy() @ np.arange(1, 10)

    mae = lambda y, p: float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))
    results["acc"] = {"mae": round(mae(te["acc"], acc_pred), 3),
                      "per_player": {p: round(mae(g["acc"], acc_pred[te["player"].values == p]), 3)
                                     for p, g in te.groupby("player")}}
    results["lamp"] = {"ord_mae": round(mae(te["lamp"], lamp_pred), 3),
                       "qwk": round(float(cohen_kappa_score(
                           te["lamp"], np.clip(np.round(lamp_pred), 1, 9).astype(int),
                           weights="quadratic", labels=list(range(1, 10)))), 3),
                       "per_player": {p: round(mae(g["lamp"], lamp_pred[te["player"].values == p]), 3)
                                      for p, g in te.groupby("player")}}
    results["bp"] = {"raw_mae": round(mae(te["bp"], bp_pred), 2),
                     "ratio_mae_x1000": round(mae(te["bp"] / te["notes"],
                                                  bp_pred / te["notes"]) * 1000, 2),
                     "per_player": {p: round(mae(g["bp"], bp_pred[te["player"].values == p]), 2)
                                    for p, g in te.groupby("player")}}

    with open(OUT / f"c_model_{args.mode}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
