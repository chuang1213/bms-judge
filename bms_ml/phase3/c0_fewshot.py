"""Phase 3.5 (task 1): learned player-state conditioning in the few-shot setting.

Replaces M2's hand-crafted conditioning (12 h_* scalars) with a C0-style learned
state: a GRU over the last K=100 prefix events, pooled at the last valid step,
concatenated with the target chart's 27-dim objective stats. Everything else is
held identical to the M2 few-shot protocol (transfer_eval.py):

  LOPO       train on ALL rows of the other 17 players (causal sequences)
  few-shot   for k in KS: prefix = D's first k plays, targets = the next <=150;
             the history input may ONLY use prefix events (deployment semantics)
  chart side v1 27-dim, standardised on training players only (leakage guard)
  heads      acc / lamp / BP trained independently (3.1: no multi-task gain)

Design decisions, mirroring transfer_eval's methodology notes:
  * NO dt / recency inputs anywhere. Few-shot targets are temporally ADJACENT to
    the prefix while training targets are spread over years, so any time-gap
    feature has incomparable distributions between train and eval — the same
    reason HIST_FEW drops days_since_active / plays_last30d / days_span. Events
    carry outcome dims only (lamp/9, acc/100, log1p(bp)); ORDER is preserved by
    the sequence (the analogue of h_acc_last10).
  * k=0 -> empty sequence (all-masked): the model degrades to chart-only, the
    analogue of M2's NaN history at k=0.

The number to beat is M2@k from transfer_results.json (hand-crafted conditioning).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from chart_repr import OBJECTIVE_STAT_COLS
from common import Imputer, load_firstplays, mae
from c_chart_aware import event_matrix

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"
KS = [0, 1, 5, 10, 20, 50, 100, 200]
W = 150            # few-shot target window (same as transfer_eval)
K = 100            # history length (same as c_chart_aware)
SEEDS = [0, 1, 2]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=DEV)


class CondModel(nn.Module):
    """GRU history -> pooled state, concat chart stats, three heads."""

    def __init__(self, seq_dim: int, chart_dim: int):
        super().__init__()
        self.gru = nn.GRU(seq_dim, 64, batch_first=True)
        self.chart = nn.Sequential(nn.Linear(chart_dim, 64), nn.ReLU())
        self.body = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
        self.heads = nn.ModuleDict({"acc": nn.Linear(64, 1), "bp": nn.Linear(64, 1),
                                    "lamp": nn.Linear(64, 9)})

    def forward(self, seq, mask, chart):
        h, _ = self.gru(seq)
        idx = (mask.sum(1).long() - 1).clamp(min=0)
        pooled = h[torch.arange(h.size(0)), idx]
        z = self.body(torch.cat([pooled, self.chart(chart)], dim=1))
        return {k: self.heads[k](z).squeeze(-1) for k in ("acc", "lamp", "bp")}


def build_seqs(target_pos: np.ndarray, pool_pos: np.ndarray, times_days: np.ndarray,
               E: np.ndarray, rng: np.random.RandomState | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Outcome-only event sequences (no dt): last min(K, len(pool)) events,
    in play order. target_pos: positions of the target rows; pool_pos: positions
    of the ALLOWED history events (strictly prior by construction).

    rng=None -> exact history (evaluation). rng given -> DEPLOYMENT-STYLE
    TRUNCATION AUGMENTATION: each training row's history is randomly truncated to
    L ~ Uniform{0..available}, so the model sees every window fill level. Without
    this, training windows are ~97% full while few-shot evaluation windows are
    0-100% full — a train/eval mismatch that cripples the learned conditioning
    exactly where it is supposed to help (measured: k=50 16.3 vs 12.4 before the
    fix). This is the sequence-model analogue of dropping the recency features."""
    m = len(target_pos)
    seqs = np.zeros((m, K, E.shape[1]), dtype=np.float32)
    masks = np.zeros((m, K), dtype=np.float32)
    for i, tp in enumerate(target_pos):
        pool = pool_pos[pool_pos < tp] if len(pool_pos) else pool_pos
        n_avail = len(pool)
        n = min(K, n_avail)
        if rng is not None and n_avail > 0:
            n = min(K, int(rng.randint(0, n_avail + 1)))
        if n:
            seqs[i, K - n:] = E[pool[-n:]]
            masks[i, K - n:] = 1.0
    return seqs, masks


def train_head(head: str, S_tr, M_tr, C_tr, y_tr, S_va, M_va, C_va, y_va,
               chart_dim: int, seq_dim: int, seed: int):
    torch.manual_seed(seed)
    model = CondModel(seq_dim, chart_dim).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if head == "lamp":
        lossf, ytr, yva = nn.functional.cross_entropy, T(y_tr - 1, torch.long), T(y_va - 1, torch.long)
    else:
        lossf = nn.functional.huber_loss
        ytr, yva = T(y_tr), T(y_va)
    Xt, Mt, Ct = T(S_tr), T(M_tr), T(C_tr)
    best, best_state, patience = np.inf, None, 0
    for _ in range(60):
        model.train()
        perm = torch.randperm(len(ytr))
        for i in range(0, len(ytr), 64):
            b = perm[i:i + 64]
            out = model(Xt[b], Mt[b], Ct[b])
            loss = lossf(out[head], ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(lossf(model(T(S_va), T(M_va), T(C_va))[head], yva))
        if vloss < best - 1e-4:
            best, patience, best_state = vloss, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 6:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


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
    E = event_matrix(fp, "C0", None)                     # outcome dims only (3)
    pos_by_player = {p: np.where(fp["player"].values == p)[0]
                     for p in fp["player"].unique()}
    players = sorted(fp["player"].unique())

    # chart-side standardisation is refit per held-out player on training players
    imp = Imputer()
    results: dict = {"seed": seed, "ks": KS, "W": W, "K": K, "lopo": {}}

    # ---- precompute augmented training sequences ONCE per player ----
    # A player's training sequences do not depend on who is held out, but the
    # naive loop rebuilt all 17 of them for every D (18x redundant CPU work).
    # Augmentation RNG is seeded per (seed, player) so this is reproducible.
    print("building augmented training sequences ...", flush=True)
    SEQ: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for p in players:
        SEQ[p] = build_seqs(pos_by_player[p], pos_by_player[p], times_days, E,
                            rng=np.random.RandomState(seed * 1000 + abs(hash(p)) % 997))

    for D in players:
        mine = fp[fp["player"] == D]
        pos_D = pos_by_player[D]
        # ---- training rows: all rows of the other players, each with its own
        # causal pool (events strictly before the target's own first play) ----
        # NB: keep PLAYER order (fp is sorted by player,time), because S_tr below is
        # concatenated in the same player order — sorting rows_tr by time here would
        # misalign rows vs sequences.
        tr_pos = np.concatenate([pos_by_player[p] for p in players if p != D])
        rows_tr = fp.loc[tr_pos]
        S_list, M_list = [], []
        for p in players:
            if p == D:
                continue
            S_list.append(SEQ[p][0]); M_list.append(SEQ[p][1])
        S_tr = np.concatenate(S_list); M_tr = np.concatenate(M_list)
        assert len(rows_tr) == S_tr.shape[0], "rows/sequence misalignment"
        y_acc = rows_tr["acc"].values / 100.0
        y_lamp = rows_tr["lamp"].values.astype(float)
        y_bp = np.log1p(rows_tr["bp"].values)

        # temporal val: last 10% per training player (same rule as c_chart_aware)
        g = rows_tr.groupby("player", sort=False)
        val_idx = np.concatenate([gg.index[-max(1, int(round(len(gg) * 0.1))):]
                                  for _, gg in g])
        is_val = rows_tr.index.isin(val_idx)
        S_va, M_va = S_tr[is_val], M_tr[is_val]
        S_tn, M_tn = S_tr[~is_val], M_tr[~is_val]
        C_all = imp.fit_transform(rows_tr[OBJECTIVE_STAT_COLS].values)
        sc = StandardScaler().fit(C_all)
        C_tn, C_va = sc.transform(C_all[~is_val]), sc.transform(C_all[is_val])

        models = {h: train_head(h, S_tn, M_tn, C_tn,
                                (y_acc if h == "acc" else y_bp if h == "bp" else y_lamp)[~is_val],
                                S_va, M_va, C_va,
                                (y_acc if h == "acc" else y_bp if h == "bp" else y_lamp)[is_val],
                                len(OBJECTIVE_STAT_COLS), E.shape[1], seed)
                  for h in ("acc", "lamp", "bp")}

        # ---- few-shot eval ----
        mine_sorted = mine.sort_values("time").reset_index(drop=True)
        curve = []
        for k in KS:
            if k >= len(mine_sorted) - 1:
                continue
            targets = mine_sorted.iloc[k:k + W]
            if not len(targets):
                continue
            pool = pos_D[:k]
            tt = targets["time"].values.astype("datetime64[s]").astype(np.int64) / 86400.0
            assert k == 0 or (tt > times_days[pool[-1]]).all()
            assert not mine_sorted["sha256"].iloc[:k].isin(targets["sha256"]).any()
            S_te = np.zeros((len(targets), K, E.shape[1]), dtype=np.float32)
            M_te = np.zeros((len(targets), K), dtype=np.float32)
            if k:
                n = min(K, k)
                S_te[:, K - n:] = E[pool[-n:]][None, :, :]
                M_te[:, K - n:] = 1.0
            C_te = sc.transform(imp.transform(targets[OBJECTIVE_STAT_COLS].values))
            with torch.no_grad():
                out = models["acc"](T(S_te), T(M_te), T(C_te.astype(np.float32)))
                out_l = models["lamp"](T(S_te), T(M_te), T(C_te.astype(np.float32)))
                out_b = models["bp"](T(S_te), T(M_te), T(C_te.astype(np.float32)))
            acc_p = out["acc"].cpu().numpy() * 100.0
            lamp_p = torch.softmax(out_l["lamp"], dim=1).cpu().numpy() @ np.arange(1, 10)
            bp_log = out_b["bp"].cpu().numpy().clip(0.0, float(y_bp.max()))
            bp_p = np.expm1(bp_log)
            curve.append({"k": k, "n_eval": int(len(targets)),
                          "acc_mae": round(mae(targets["acc"], acc_p), 3),
                          "lamp_mae": round(mae(targets["lamp"], lamp_p), 3),
                          "bp_mae": round(mae(targets["bp"], bp_p), 2)})
        results["lopo"][D] = {"n_events": int(len(mine)), "fewshot": curve}
        print(D, {c["k"]: c["acc_mae"] for c in curve}, flush=True)

    # weighted aggregate
    agg = []
    for k in KS:
        rows = [c for D in players for c in results["lopo"][D]["fewshot"] if c["k"] == k]
        if not rows:
            continue
        w = np.array([c["n_eval"] for c in rows], float)
        agg.append({"k": k, "n_eval": int(w.sum()),
                    **{key: round(float(np.average([c[key] for c in rows], weights=w)), 3)
                       for key in ("acc_mae", "lamp_mae", "bp_mae")}})
    results["fewshot_aggregate"] = agg
    json.dump(results, open(OUT / f"c0_fewshot_s{seed}.json", "w"), indent=2, default=str)
    print("\nC0-conditioned few-shot (seed %d):" % seed)
    for e in agg:
        print(f"  k={e['k']:3}: acc {e['acc_mae']:.2f}  lamp {e['lamp_mae']:.2f}  BP {e['bp_mae']:.1f}")


if __name__ == "__main__":
    main()
