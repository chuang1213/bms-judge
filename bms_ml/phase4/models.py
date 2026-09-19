"""Phase 4 predictors: the fixed baselines, biased matrix factorisation (M3), and
the cold-chart content baseline.

Model contract: `fit(train_df)` / `predict(test_df) -> np.ndarray` on the accuracy
scale. Every model sees ONLY its training frame, so a leak is impossible by
construction; `test_phase4_protocol.py` enforces the invariance.

Nothing here uses time, first plays, recency, difficulty-table levels or
hand-defined skill axes. Predictions are clipped to [0, 100] because the target is
a percentage; the clip rate is reported by the driver so silent saturation cannot
hide.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

ACC_LO, ACC_HI = 0.0, 100.0


class BaseModel:
    name = "base"

    def fit(self, tr: pd.DataFrame, target: str) -> "BaseModel":
        raise NotImplementedError

    def predict(self, te: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def config(self) -> dict:
        return {}


# --------------------------------------------------------------------------- #
# fixed baselines
# --------------------------------------------------------------------------- #
class BaselineModel(BaseModel):
    """global_mean / player_mean / chart_mean / player_chart_bias (+_shrunk).

    The additive decomposition is the M3 gate. Effects are centered by
    construction (group mean minus the global mean), and an unseen player or
    chart contributes 0, i.e. falls back to the global mean. When
    `shrink_k` is set the effects are empirical-Bayes shrunk towards 0 by
    n/(n+k); that variant is a model, never the gate.
    """

    def __init__(self, name: str, shrink_k: float | None = None):
        self.name = name
        self.shrink_k = shrink_k

    def fit(self, tr: pd.DataFrame, target: str) -> "BaselineModel":
        self.target = target
        self.mu = float(tr[target].mean())
        if self.shrink_k is None:
            self.player_eff = tr.groupby("player")[target].mean() - self.mu
            self.chart_eff = tr.groupby("sha256")[target].mean() - self.mu
        else:
            g = tr.groupby("player")[target].agg(["mean", "size"])
            self.player_eff = (g["mean"] - self.mu) * (g["size"] / (g["size"] + self.shrink_k))
            g = tr.groupby("sha256")[target].agg(["mean", "size"])
            self.chart_eff = (g["mean"] - self.mu) * (g["size"] / (g["size"] + self.shrink_k))
        return self

    def predict(self, te: pd.DataFrame) -> np.ndarray:
        p = te["player"].map(self.player_eff).astype(float).fillna(0.0).to_numpy()
        c = te["sha256"].map(self.chart_eff).astype(float).fillna(0.0).to_numpy()
        if self.name == "global_mean":
            return np.full(len(te), self.mu)
        if self.name == "player_mean":
            return self.mu + p
        if self.name == "chart_mean":
            return self.mu + c
        if self.name in ("player_chart_bias", "player_chart_bias_shrunk"):
            return self.mu + p + c
        raise ValueError(f"unknown baseline {self.name}")


# --------------------------------------------------------------------------- #
# M3: biased matrix factorisation (ALS)
# --------------------------------------------------------------------------- #
class MFModel(BaseModel):
    """pred(u,c) = mu + b_u + b_c + <p_u, q_c>.

    Fitted by alternating least squares on a sparse residual matrix. ALS (not SGD)
    because it needs no learning rate and is deterministic given the
    hyper-parameters, so a seed change only moves the factor init.

    Regularisation:
      * `reg` (lambda) multiplies the identity in BOTH normal equations;
      * identity init (`init_id`) starts the factors at a scaled identity block
        instead of random values, which is what makes 2-3 dimensions behave the
        same across seeds instead of collapsing into arbitrary rotations.
    """

    def __init__(self, dim: int = 8, reg: float = 3.0, reg_bias: float | None = None,
                 n_epochs: int = 25, seed: int = 0, init_id: bool = False,
                 verbose: bool = False, patience: int = 4, damp: float = 0.5):
        self.name = f"mf_d{dim}_r{reg}"
        self.dim = int(dim)
        self.reg = float(reg)
        self.reg_bias = float(reg if reg_bias is None else reg_bias)
        self.n_epochs = int(n_epochs)
        self.seed = int(seed)
        # Identity init is deliberately NOT the default: on this matrix it makes the
        # latent block start far from 0, the bias step then compensates for a
        # spurious signal, and the two blocks feed back into divergence
        # (train RMSE 6.0 -> 1e7 within ~15 epochs). Near-zero init keeps every
        # update small and matches what SGD reaches.
        self.init_id = init_id
        self.damp = float(damp)
        self.verbose = verbose
        self.patience = int(patience)
        self.clip_lo = self.clip_hi = 0.0
        self.history: list[float] = []      # train RMSE per epoch
        self.val_history: list[float] = []
        self.best_epoch: int | None = None
        self.diverged = False

    def _rmse(self, pu, ci, r) -> float:
        pred = (self.mu + self.bu[pu] + self.bc[ci]
                + np.einsum("ij,ij->i", self.P[pu], self.Q[ci]))
        return float(np.sqrt(np.mean((r - pred) ** 2)))

    # -- internals ---------------------------------------------------------- #
    @staticmethod
    def _uv(df: pd.DataFrame, target: str):
        """(player index, chart index, ratings) plus the index vocabularies."""
        p_codes, p_uniq = pd.factorize(df["player"], sort=True)
        c_codes, c_uniq = pd.factorize(df["sha256"], sort=True)
        return (p_codes.astype(np.int64), c_codes.astype(np.int64),
                df[target].to_numpy(float), p_uniq, c_uniq)

    def _batched_chol(self, G: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Solve (G^T G + reg I) x = G^T r for every entity in one batch.

        Entities are grouped by support size so their normal equations stack into a
        (n, d, d) tensor and go through one batched Cholesky instead of a Python
        per-entity loop (the loop was ~10k sparse solves per epoch on the chart
        side). Falls back to a per-entity solve if the batch cost would be silly.
        """
        n, d = len(G), self.dim
        # R arrives as one residual per (entity, interaction). Reshape to (n, size)
        # (NOT just its first column - an earlier guard did exactly that and silently
        # replaced every residual with the first one, zeroing the solution).
        R = np.asarray(R, dtype=float).reshape(n, -1)
        assert R.shape[1] == G.shape[1], (G.shape, R.shape)
        if n * d * d > 4_000_000:                      # ~65 MB in float64
            cut = max(1, 4_000_000 // (d * d))
            return np.vstack([self._batched_chol(G[i:i + cut], R[i:i + cut])
                              for i in range(0, n, cut)])
        lhs = np.einsum("nij,nik->njk", G, G)
        lhs[:, np.arange(d), np.arange(d)] += self.reg
        rhs = np.einsum("nij,ni->nj", G, R)
        try:
            L = np.linalg.cholesky(lhs)
            z = np.linalg.solve(L, rhs[..., None])[..., 0]
            return np.linalg.solve(np.swapaxes(L, -1, -2), z[..., None])[..., 0]
        except np.linalg.LinAlgError:                  # rank-deficient block
            out = np.zeros((n, d))
            for i in range(n):
                lhs[i, np.arange(d), np.arange(d)] += 1e-6
                out[i] = np.linalg.solve(lhs[i], rhs[i])
            return out

    @staticmethod
    def _group_by_support(A: sp.csr_matrix) -> list[np.ndarray]:
        """Entity ids grouped by identical support size (built once, reused)."""
        sizes = np.diff(A.indptr)
        order = np.argsort(sizes, kind="stable")
        bounds = np.flatnonzero(np.diff(sizes[order])) + 1
        return [g for g in np.split(order, bounds) if len(g) and sizes[g[0]] > 0]

    def _solve(self, resid: np.ndarray, F_other: np.ndarray, A: sp.csr_matrix,
               reg: float, groups=None) -> np.ndarray:
        """One ALS half-step: solve every row of A for its own factor vector.

        `F_other` is the FIXED opposite-side factor matrix and is never written to
        (an earlier version solved in place into it, which aliased the two factor
        blocks and made the objective oscillate instead of decreasing).
        """
        if groups is None:
            groups = self._group_by_support(A)
        F_new = np.zeros((A.shape[0], F_other.shape[1]))
        for grp in groups:
            size = int(A.indptr[grp[0] + 1] - A.indptr[grp[0]])
            cols = (A.indptr[grp][:, None] + np.arange(size)[None, :]).reshape(-1)
            idx = A.indices[cols]
            if size == 1:
                # one interaction => G^T G = g g^T, rank 1. Sherman-Morrison gives
                # x = g * (g^T r) / (reg + ||g||^2); the `reg`-only denominator is
                # wrong and silently scales the whole factor vector.
                g = F_other[idx.reshape(-1)]                    # (n_grp, dim)
                r1 = resid[cols].reshape(-1)                    # (n_grp,)
                F_new[grp] = (g * (r1 / (self.reg + (g ** 2).sum(axis=1)))[:, None])
                continue
            F_new[grp] = self._batched_chol(F_other[idx].reshape(len(grp), size, -1),
                                            resid[cols])
        return F_new

    def fit(self, tr: pd.DataFrame, target: str,
            val_idx: np.ndarray | None = None) -> "MFModel":
        """`val_idx` indexes rows OF `tr` used to pick the stopping epoch.

        Selection must NOT use the training objective: ALS fits the observed cells
        better almost every epoch, so "best training epoch" would systematically
        pick an underfit state. When `val_idx` is None the training objective is
        used and the result is flagged as selection-on-train in `config()`.
        """
        self.target = target
        self.clip_lo = self.clip_hi = 0
        pu, ci, r, self.players, self.charts = self._uv(tr, target)
        self.n_u, self.n_c = len(self.players), len(self.charts)
        self.mu = float(r.mean())
        self.bu = np.zeros(self.n_u)
        self.bc = np.zeros(self.n_c)
        rng = np.random.default_rng(self.seed)
        if self.init_id:
            # scaled identity block (deterministic, seed-independent) - kept for the
            # ablation only; see the constructor comment for why it is not default
            scale = np.sqrt(1.0 / max(self.dim, 1))
            P = np.zeros((self.n_u, self.dim))
            Q = np.zeros((self.n_c, self.dim))
            d = min(self.dim, self.n_u, self.n_c)
            P[np.arange(d), np.arange(d)] = 1.0
            Q[np.arange(d), np.arange(d)] = 1.0
            P *= scale
            Q *= scale
        else:
            P = rng.normal(scale=0.01, size=(self.n_u, self.dim))
            Q = rng.normal(scale=0.01, size=(self.n_c, self.dim))

        ru = sp.csr_matrix((np.ones(len(pu)), (pu, ci)), shape=(self.n_u, self.n_c))
        rc = sp.csr_matrix((np.ones(len(pu)), (ci, pu)), shape=(self.n_c, self.n_u))
        ru.sort_indices()
        rc.sort_indices()
        gu = self._group_by_support(ru)
        gc = self._group_by_support(rc)

        if val_idx is not None:
            keep = np.ones(len(tr), dtype=bool)
            keep[val_idx] = False
            if keep.sum() == 0 or len(val_idx) == 0:
                val_idx = None
        if val_idx is not None:
            vp, vc = pu[val_idx], ci[val_idx]
            vr = r[val_idx]
            self.selection = "inner_validation"
        else:
            self.selection = "train_objective"

        self.P, self.Q = P, Q
        best = np.inf
        best_state = None
        stale = 0
        for epoch in range(self.n_epochs):
            # --- biases: damped step toward the ridge-optimal residual mean.
            # Damping matters: the plain closed-form update moves the bias all the
            # way to the optimum given the CURRENT factors, which (together with the
            # factor step) can cycle or diverge. Damping trades a little speed for a
            # monotone-ish objective that the early-stopping rule can trust.
            res = r - self.mu - np.einsum("ij,ij->i", P[pu], Q[ci])
            num = np.bincount(pu, weights=res, minlength=self.n_u)
            cnt = np.bincount(pu, minlength=self.n_u).astype(float)
            new_bu = num / (cnt + self.reg_bias)
            num = np.bincount(ci, weights=res, minlength=self.n_c)
            cnt = np.bincount(ci, minlength=self.n_c).astype(float)
            new_bc = num / (cnt + self.reg_bias)
            self.bu = self.bu + self.damp * (new_bu - self.bu)
            self.bc = self.bc + self.damp * (new_bc - self.bc)

            # --- factors on the bias-corrected residual. P is updated first from the
            # previous Q, then Q from the NEW P (standard ALS); neither call mutates
            # the other block.
            res = r - self.mu - self.bu[pu] - self.bc[ci]
            P = self._solve(res, Q, ru, self.reg, groups=gu)
            res = r - self.mu - self.bu[pu] - self.bc[ci]
            Q = self._solve(res, P, rc, self.reg, groups=gc)
            self.P, self.Q = P, Q

            train_obj = self._rmse(pu, ci, r)
            if not np.isfinite(train_obj):
                # roll back to the best state seen and stop; a diverged epoch must
                # never become the returned model
                self.diverged = True
                break
            obj = self._rmse(vp, vc, vr) if val_idx is not None else train_obj
            self.history.append(round(train_obj, 6))
            self.val_history.append(round(obj, 6))
            if self.verbose:
                print(f"    epoch {epoch}: train_rmse={train_obj:.4f}"
                      + (f" val_rmse={obj:.4f}" if val_idx is not None else ""))

            if np.isfinite(obj) and obj < best - 1e-4:
                best, best_state, stale, self.best_epoch = obj, (P.copy(), Q.copy(),
                                                                 self.bu.copy(),
                                                                 self.bc.copy()), 0, epoch
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self.P, self.Q, self.bu, self.bc = best_state
        return self

    def predict(self, te: pd.DataFrame) -> np.ndarray:
        p_idx = self.players.get_indexer(te["player"])
        c_idx = self.charts.get_indexer(te["sha256"])
        out = np.full(len(te), self.mu)
        seen_p, seen_c = p_idx >= 0, c_idx >= 0
        out[seen_p] += self.bu[p_idx[seen_p]]
        out[seen_c] += self.bc[c_idx[seen_c]]
        both = seen_p & seen_c      # latent term needs BOTH sides known
        if both.any():
            out[both] += np.einsum("ij,ij->i", self.P[p_idx[both]], self.Q[c_idx[both]])
        raw = out.copy()
        out = np.clip(out, ACC_LO, ACC_HI)
        self.clip_lo += int((raw < ACC_LO).sum())
        self.clip_hi += int((raw > ACC_HI).sum())
        return out

    def config(self) -> dict:
        return {"dim": self.dim, "reg": self.reg, "reg_bias": self.reg_bias,
                "n_epochs_max": self.n_epochs, "best_epoch": self.best_epoch,
                "epoch_selection": getattr(self, "selection", "unknown"),
                "damp": self.damp, "diverged": self.diverged,
                "init": "identity" if self.init_id else "near_zero",
                "solver": "ALS",
                "epoch_history_train_rmse": self.history,
                "epoch_history_val_rmse": self.val_history}


# --------------------------------------------------------------------------- #
# cold-chart content baseline
# --------------------------------------------------------------------------- #
class ContentRidgeModel(BaseModel):
    """Chart content -> accuracy, for the cold-chart split (PROTOCOL §6.3).

    Uses ONLY the 26 objective chart statistics (`cs_*`, from the corpus parser)
    plus a per-player offset. It deliberately has no latent player/chart factors,
    because the cold-chart question is exactly "can chart content alone carry the
    signal when the matrix cell is absent?".

    A difficulty-table level is NOT used (red line §3.1).
    """

    name = "content_ridge"

    def __init__(self, alpha: float = 1.0, use_player: bool = True):
        self.alpha = float(alpha)
        self.use_player = bool(use_player)

    def fit(self, tr: pd.DataFrame, target: str) -> "ContentRidgeModel":
        from sklearn.linear_model import Ridge
        self.target = target
        self.feat_cols = [c for c in tr.columns if c.startswith("cs_")]
        X = tr[self.feat_cols].to_numpy(float)
        self.mu_x = np.nanmean(X, axis=0)
        self.sd_x = np.nanstd(X, axis=0)
        self.sd_x[self.sd_x == 0] = 1.0
        Xs = self._std(X)
        self.player_eff = (tr.groupby("player")[target].mean() - float(tr[target].mean())
                           if self.use_player else pd.Series(dtype=float))
        self.mu = float(tr[target].mean())
        off = tr["player"].map(self.player_eff).astype(float).fillna(0.0).to_numpy()
        self.ridge = Ridge(alpha=self.alpha).fit(Xs, tr[target].to_numpy(float) - self.mu - off)
        self.clip_lo = self.clip_hi = 0
        return self

    def _std(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu_x) / self.sd_x

    def predict(self, te: pd.DataFrame) -> np.ndarray:
        Xs = self._std(te[self.feat_cols].to_numpy(float))
        off = te["player"].map(self.player_eff).astype(float).fillna(0.0).to_numpy()
        raw = self.mu + off + self.ridge.predict(Xs)
        out = np.clip(raw, ACC_LO, ACC_HI)
        self.clip_lo += int((raw < ACC_LO).sum())
        self.clip_hi += int((raw > ACC_HI).sum())
        return out

    def config(self) -> dict:
        return {"alpha": self.alpha, "n_features": len(self.feat_cols),
                "use_player_offset": self.use_player, "features": "cs_* (26 objective stats)"}
