"""Phase 4B feature-matrix experiment — the two gates, over the content feature families.

Builds on the frozen Phase 4B protocol (PHASE4B_PLAN.md). This script DOES NOT
redefine the splits or the metrics: it imports `phase4/splits.py` and
`phase4/metrics.py` (the single sources of truth) and reuses the gate logic shape
from `phase4b/run_phase4b.py`.

WHAT IS COMPARED
  feature families : obj (26 objective stats) | msd (MinaCalc 7 axes) |
                     perm (5040-permutation geometry) | v2 (objective v2) |
                     and the ladders obj -> obj+msd -> obj+msd+perm -> +v2
  model classes    : ridge | gbdt
  coverage layers  : FULL  (every row the track can score)
                     COMMON (rows where EVERY track has features) <- main conclusion

The user's decision (2026-09-12) is explicit: all feature comparisons must run on the
same row set, both coverage layers must be reported, and the MAIN conclusion is the
common-coverage one. A track that only wins on an easier subset is not a win.

TWO GATES (same definitions as PHASE4B_REPORT.md)
  Gate 1 difficulty : predict a cold chart's mean residual (player skill removed) and
                      beat the in-protocol content_ridge baseline ON THE SAME ROWS.
                      >=3 seeds, gain > seed spread.
  Gate 2 interaction: `content_noninteractive` vs `content_x_player`, judged on
                      player-internal MAE / Spearman / NDCG@10, plus a permutation
                      check. Chart-level MAE improvement with flat player-internal
                      ordering = "difficulty only" = FAIL.

LR2 has 2 players: numbers are recorded, no statistical verdict is drawn.

Usage
  python bms_ml/phase4b/run_phase4b_features.py --steps msd,perm
  python bms_ml/phase4b/run_phase4b_features.py --families obj,obj+msd
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PHASE4 = ROOT / "bms_ml" / "phase4"
sys.path.insert(0, str(PHASE4))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_matrix import MIN_JUDGEMENTS, load_frame  # noqa: E402
from metrics import mae, player_train_means, ranking_by_player, spearman  # noqa: E402
from provenance import header, write_result  # noqa: E402
from splits import split_mask  # noqa: E402

DATASET = ROOT / "bms_ml/output/phase4/dataset"
CROSS_SECTION = DATASET / "cross_section.parquet"
TARGET = "acc"
SEEDS_DEFAULT = (0, 1, 2)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)      # chosen on train-only CV
GBDT_KW = dict(max_iter=300, learning_rate=0.06, max_depth=3)

# ladder = "each added layer must beat the previous one" (PROTOCOL sec 11.5)
LADDERS = {
    "obj": ["obj"],
    "obj+msd": ["obj", "msd"],
    "obj+msd+perm": ["obj", "msd", "perm"],
    "obj+msd+perm+v2": ["obj", "msd", "perm", "v2"],
}


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def load_features(msd_cap: float = 100.0) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """cross_section rows + the four feature families, joined per client row."""
    df, filt = load_frame(CROSS_SECTION, TARGET, MIN_JUDGEMENTS)
    info = {"phase4_filter": filt, "families": {}}

    obj_cols = sorted(c for c in df.columns if c.startswith("cs_"))
    info["families"]["obj"] = {"cols": obj_cols, "source": "cross_section cs_*"}

    msd_file = DATASET / ("msd_cap100.parquet" if msd_cap == 100.0 else "msd_cap40.parquet")
    msd = pd.read_parquet(msd_file)
    msd_cols = [c for c in msd.columns if c.startswith("msd_") and c != "msd_cap"]
    df = df.merge(msd[["sha256"] + msd_cols], on="sha256", how="left")
    info["families"]["msd"] = {"cols": msd_cols, "source": msd_file.name,
                               "cap": msd_cap, "axes": len(msd_cols)}

    perm_file = DATASET / "perm_space_full.parquet"
    if perm_file.exists():
        perm = pd.read_parquet(perm_file)
        perm_cols = sorted(c for c in perm.columns if c.startswith("ps_"))
        df = df.merge(perm[["sha256"] + perm_cols], on="sha256", how="left")
        info["families"]["perm"] = {"cols": perm_cols, "source": perm_file.name}
    else:
        info["families"]["perm"] = {"cols": [], "source": None,
                                    "note": "perm_space_full.parquet missing"}
        perm_cols = []

    v2_file = DATASET / "chart_stats_v2_full.parquet"
    if v2_file.exists():
        v2 = pd.read_parquet(v2_file)
        v2_cols = sorted(c for c in v2.columns if c.startswith("v2_"))
        df = df.merge(v2[["sha256"] + v2_cols], on="sha256", how="left")
        info["families"]["v2"] = {"cols": v2_cols, "source": v2_file.name}
    else:
        info["families"]["v2"] = {"cols": [], "source": None,
                                  "note": "chart_stats_v2_full.parquet not built yet"}
    return df, msd, info


def family_cols(family: str, info: dict) -> list[str]:
    if family in ("obj", "msd", "perm", "v2"):
        return list(info["families"].get(family, {}).get("cols", []))
    if family in LADDERS:
        cols = []
        for part in LADDERS[family]:
            cols += family_cols(part, info)
        return cols
    raise ValueError(f"unknown feature family {family!r}")


def coverage(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].notna().all(axis=1) if cols else pd.Series(True, index=df.index)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def fit_predict_ridge(Xtr, ytr, Xte, seed: int, folds: int = 5) -> np.ndarray:
    """Ridge with alpha chosen by K-fold CV INSIDE the training frame only."""
    from sklearn.linear_model import Ridge
    mu, sd = np.nanmean(Xtr, axis=0), np.nanstd(Xtr, axis=0)
    sd[sd == 0] = 1.0
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    Ztr = np.nan_to_num(Ztr, nan=0.0)
    Zte = np.nan_to_num(Zte, nan=0.0)
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, size=len(Ztr))
    best, best_err = RIDGE_ALPHAS[0], np.inf
    for a in RIDGE_ALPHAS:
        errs = []
        for f in range(folds):
            m = fold == f
            if m.sum() == 0 or (~m).sum() == 0:
                continue
            mdl = Ridge(alpha=a).fit(Ztr[~m], ytr[~m])
            errs.append(np.mean(np.abs(mdl.predict(Ztr[m]) - ytr[m])))
        if errs and np.mean(errs) < best_err:
            best, best_err = a, float(np.mean(errs))
    return Ridge(alpha=best).fit(Ztr, ytr).predict(Zte), {"alpha": best,
                                                          "cv_mae": round(best_err, 4)}


def fit_predict_gbdt(Xtr, ytr, Xte, seed: int) -> tuple[np.ndarray, dict]:
    """GBDT. Missing values are left as NaN on purpose: imputing them would be the
    thing we explicitly forbade. HistGradientBoosting handles NaN natively."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(random_state=seed, **GBDT_KW)
    return m.fit(Xtr, ytr).predict(Xte), {"params": GBDT_KW}


def fit_predict(model: str, Xtr, ytr, Xte, seed: int):
    if model == "ridge":
        return fit_predict_ridge(Xtr, ytr, Xte, seed)
    if model == "gbdt":
        return fit_predict_gbdt(Xtr, ytr, Xte, seed)
    raise ValueError(model)


def interaction_matrix(X: np.ndarray, players: np.ndarray) -> np.ndarray:
    """[X | one-hot(player) * X]: lets each player weight the content differently."""
    if X.shape[1] == 0:
        return X
    return np.hstack([X] + [X * (players == p).astype(float)[:, None]
                            for p in np.unique(players)])


# --------------------------------------------------------------------------- #
# evaluation pieces
# --------------------------------------------------------------------------- #
def row_metrics(te: pd.DataFrame, pred: np.ndarray, means: dict, y: np.ndarray) -> dict:
    base = te["player"].map(means).astype(float).fillna(float(np.mean(list(means.values()))))
    base = base.to_numpy()
    dev_y, dev_p = y - base, pred - base
    rk = ranking_by_player(te, pred, TARGET)
    return {
        "mae": round(mae(y, pred), 4),
        "player_centered_mae": round(mae(dev_y, dev_p), 4),
        "spearman": (lambda s: round(s, 4) if s is not None else None)(spearman(y, pred)),
        "player_rank_spearman": rk["aggregate"].get("spearman", {}).get("mean"),
        "ndcg@10": rk["aggregate"].get("ndcg@10", {}).get("mean"),
        "ndcg@20": rk["aggregate"].get("ndcg@20", {}).get("mean"),
        "n_rows": int(len(te)),
        "n_charts": int(te["sha256"].nunique()),
        "n_players": int(te["player"].nunique()),
    }


def build_chart_frame(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str],
                      means: dict, mu: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-chart frames for the difficulty gate: mean residual + rater composition.

    residual = acc - player_train_mean(player), so the model cannot win by learning
    which players rated the chart (their skill is removed and then added back at
    scoring time).
    """
    d = tr.copy()
    d["_pmean"] = d["player"].map(means).astype(float)
    d["_resid"] = d[TARGET] - d["_pmean"]
    g = d.groupby("sha256", sort=True)
    cd_tr = g.agg(mean_resid=("_resid", "mean")).join(g[cols].first()).reset_index()

    comp = te.groupby("sha256", sort=True)["player"].apply(
        lambda s: float(np.mean([means.get(p, mu) for p in s])))
    cd_te = comp.rename("mean_skill").to_frame().join(
        te.groupby("sha256", sort=True)[cols].first()).reset_index()
    return cd_tr, cd_te


def predict_from_chart_resid(te: pd.DataFrame, cd_te: pd.DataFrame,
                             pred_resid: np.ndarray, means: dict,
                             mu: float) -> np.ndarray:
    base = te["player"].map(means).astype(float).fillna(mu).to_numpy()
    resid_map = dict(zip(cd_te["sha256"], pred_resid))
    return np.clip(base + te["sha256"].map(resid_map).astype(float).fillna(0.0).to_numpy(),
                   0, 100)


def gate1_chart_difficulty(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str],
                           target: str, seed: int, model: str,
                           ref_cols: list[str] | None = None) -> dict:
    """Does chart content predict a COLD chart's difficulty, beating content_ridge?

    The reference is ALWAYS the same fixed baseline - Ridge on the objective stats -
    never "the first column of whatever family is being tested". Earlier this function
    built the reference from `cols[:1]`, which made the MSD/perm tracks compare against
    a one-feature straw man instead of the real baseline.
    """
    means = player_train_means(tr, target)
    mu = float(tr[target].mean())
    cd_tr, cd_te = build_chart_frame(tr, te, cols, means, mu)
    pred_resid, cfg = fit_predict(model, cd_tr[cols].to_numpy(float),
                                  cd_tr["mean_resid"].to_numpy(float),
                                  cd_te[cols].to_numpy(float), seed)
    pred = predict_from_chart_resid(te, cd_te, pred_resid, means, mu)
    y = te[target].to_numpy(float)
    base = te["player"].map(means).astype(float).fillna(mu).to_numpy()

    # ---- fixed in-protocol reference: content_ridge = 26 objective stats + player
    # offset, fitted on the SAME rows (per user decision 4: same row set for all).
    ref_pool = [c for c in (ref_cols or []) if c in tr.columns]
    if ref_pool:
        from sklearn.linear_model import Ridge
        cd_tr_ref, cd_te_ref = build_chart_frame(tr, te, ref_pool, means, mu)
        Xr = cd_tr_ref[ref_pool].to_numpy(float)
        Xr_te = cd_te_ref[ref_pool].to_numpy(float)
        keep_tr = ~np.isnan(Xr).any(axis=1)
        keep_te = ~np.isnan(Xr_te).any(axis=1)
        mu_r, sd_r = Xr[keep_tr].mean(axis=0), Xr[keep_tr].std(axis=0)
        sd_r[sd_r == 0] = 1.0
        ref = Ridge(alpha=1.0).fit((Xr[keep_tr] - mu_r) / sd_r,
                                   cd_tr_ref.loc[keep_tr, "mean_resid"].to_numpy(float))
        ref_resid = np.zeros(len(cd_te_ref))
        if keep_te.any():
            ref_resid[keep_te] = ref.predict((Xr_te[keep_te] - mu_r) / sd_r)
        ref_row = predict_from_chart_resid(te, cd_te_ref, ref_resid, means, mu)
    else:
        ref_row = base           # no objective stats on these rows -> player mean
    return {"model": model, "config": cfg,
            "row_mae": round(mae(y, pred), 4),
            "row_mae_within_player": round(mae(y - base, pred - base), 4),
            "content_ridge_row_mae": round(mae(y, ref_row), 4),
            "player_mean_row_mae": round(mae(y, base), 4),
            "gain_over_content_ridge": round(mae(y, ref_row) - mae(y, pred), 4),
            "gain_over_player_mean": round(mae(y, base) - mae(y, pred), 4),
            "n_test_rows": int(len(te)), "n_test_charts": int(te["sha256"].nunique())}


def gate2_interaction(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str],
                      target: str, seed: int, model: str) -> dict:
    """Can the content give DIFFERENT predictions for different players?"""
    means = player_train_means(tr, target)
    mu = float(tr[target].mean())
    base_tr = tr["player"].map(means).astype(float).fillna(mu).to_numpy()
    base_te = te["player"].map(means).astype(float).fillna(mu).to_numpy()
    ytr = tr[target].to_numpy(float)
    y = te[target].to_numpy(float)

    mu_x, sd_x = np.nanmean(tr[cols].to_numpy(float), axis=0), \
        np.nanstd(tr[cols].to_numpy(float), axis=0)
    sd_x[sd_x == 0] = 1.0
    Xtr = np.nan_to_num((tr[cols].to_numpy(float) - mu_x) / sd_x)
    Xte = np.nan_to_num((te[cols].to_numpy(float) - mu_x) / sd_x)

    out = {"player_mean": row_metrics(te, base_te, means, y)}
    pred_ni, cfg_ni = fit_predict(model, Xtr, ytr - base_tr, Xte, seed)
    p_ni = np.clip(base_te + pred_ni, 0, 100)
    out["content_noninteractive"] = row_metrics(te, p_ni, means, y)
    out["content_noninteractive"]["config"] = cfg_ni

    Itr = interaction_matrix(Xtr, tr["player"].to_numpy())
    Ite = interaction_matrix(Xte, te["player"].to_numpy())
    pred_xi, cfg_xi = fit_predict(model, Itr, ytr - base_tr, Ite, seed)
    p_xi = np.clip(base_te + pred_xi, 0, 100)
    out["content_x_player"] = row_metrics(te, p_xi, means, y)
    out["content_x_player"]["config"] = cfg_xi
    out["content_x_player"]["n_features"] = int(Itr.shape[1])

    # permutation guard: break the content<->player pairing and re-predict
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(te))
    pred_perm, _ = fit_predict(model, Itr, ytr - base_tr, Ite[perm], seed)
    p_perm = np.clip(base_te + pred_perm, 0, 100)
    out["content_x_player_permuted"] = row_metrics(te, p_perm, means, y)
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_client(cdf: pd.DataFrame, families: list[str], info: dict, splits: list[str],
               seeds: list[int], models: list[str], frac: float, verbose: bool,
               common_families: list[str] | None = None) -> dict:
    block = {"rows": int(len(cdf)), "players": int(cdf["player"].nunique()),
             "charts": int(cdf["sha256"].nunique()), "coverage": {}, "per_family": {}}

    for fam in families:
        cols = family_cols(fam, info)
        if not cols:
            block["per_family"][fam] = {"skipped": "no feature columns available"}
            continue
        cov = coverage(cdf, cols)
        block["coverage"][fam] = {
            "n_cols": len(cols),
            "rows_with_all_features": int(cov.sum()),
            "row_coverage": round(float(cov.mean()), 4),
            "charts_with_all_features": int(cdf.loc[cov, "sha256"].nunique()),
        }
    # COMMON rows = rows where every family IN THE COMMON SET has features.
    # `--common-families` defaults to the high-coverage families only. Including a
    # sparse family here would silently shrink the "same rows" layer: with v2 in the
    # set it fell to 26% of rows AND to an easier subset (reference MAE 6.60 vs 6.80).
    # So the sparse family is scored on its own rows and on this common set instead.
    common_set = common_families if common_families is not None else families
    common = pd.Series(True, index=cdf.index)
    for fam in common_set:
        cols = family_cols(fam, info)
        if cols:
            common &= coverage(cdf, cols)
    block["coverage"]["_common_all_families"] = {
        "rows": int(common.sum()), "row_coverage": round(float(common.mean()), 4),
        "charts": int(cdf.loc[common, "sha256"].nunique()),
        "computed_from_families": list(common_set),
        "note": "MAIN conclusion layer: every family scored on exactly these rows"}

    for fam in families:
        cols = family_cols(fam, info)
        if not cols:
            continue
        fam_block = {"n_cols": len(cols), "per_split": {}}
        for split in splits:
            sp_block = {"per_seed": {}, "layers": {}}
            for layer, mask in (("full", coverage(cdf, cols)), ("common", common)):
                sub = cdf[mask].reset_index(drop=True)
                if len(sub) < 100:
                    sp_block["layers"][layer] = {"skipped": "too few rows", "rows": len(sub)}
                    continue
                per_seed = {}
                for seed in seeds:
                    test = split_mask(sub, split, seed, frac)
                    if test.sum() < 10 or (~test).sum() < 10:
                        per_seed[str(seed)] = {"skipped": "split too small"}
                        continue
                    tr, te = sub[~test].reset_index(drop=True), sub[test].reset_index(drop=True)
                    ref_pool = info["families"]["obj"]["cols"]
                    entry = {}
                    for model in models:
                        g1 = gate1_chart_difficulty(tr, te, cols, TARGET, seed, model,
                                                    ref_cols=ref_pool)
                        g2 = gate2_interaction(tr, te, cols, TARGET, seed, model)
                        entry[model] = {"gate1": g1, "gate2": g2}
                    per_seed[str(seed)] = entry
                    if verbose:
                        r = per_seed[str(seed)][models[0]]
                        print(f"    {fam:<18} {split:<22} {layer:<7} seed={seed} "
                              f"g1_mae={r['gate1']['row_mae']:.3f} "
                              f"vs_ref={r['gate1']['content_ridge_row_mae']:.3f} "
                              f"xi_cmae={r['gate2']['content_x_player']['player_centered_mae']:.3f}")
                sp_block["layers"][layer] = {"rows": len(sub), "per_seed": per_seed}
            fam_block["per_split"][split] = sp_block
        block["per_family"][fam] = fam_block
    return block


def summarise(block: dict, families: list[str], splits: list[str],
              models: list[str]) -> dict:
    """Per (family, split, layer, model): mean/std over seeds + gate verdicts."""
    out = {}
    for fam, fb in block.get("per_family", {}).items():
        if "per_split" not in fb:
            continue
        for split, sp in fb["per_split"].items():
            for layer, lb in sp.get("layers", {}).items():
                if "per_seed" not in lb:
                    continue
                for model in models:
                    entries = [v[model] for v in lb["per_seed"].values()
                               if isinstance(v, dict) and model in v]
                    if not entries:
                        continue
                    key = f"{fam}|{split}|{layer}|{model}"
                    def agg(path, sub=None):
                        vals = []
                        for e in entries:
                            cur = e
                            for p in path:
                                cur = cur.get(p, {})
                            if isinstance(cur, (int, float)):
                                vals.append(cur)
                        if not vals:
                            return None
                        return {"mean": round(float(np.mean(vals)), 4),
                                "std": round(float(np.std(vals, ddof=1))
                                             if len(vals) > 1 else 0.0, 4),
                                "values": [round(float(v), 4) for v in vals]}
                    g1_mae = agg(["gate1", "row_mae"])
                    g1_ref = agg(["gate1", "content_ridge_row_mae"])
                    g2_ni = agg(["gate2", "content_noninteractive", "player_centered_mae"])
                    g2_xi = agg(["gate2", "content_x_player", "player_centered_mae"])
                    rk_ni = agg(["gate2", "content_noninteractive", "player_rank_spearman"])
                    rk_xi = agg(["gate2", "content_x_player", "player_rank_spearman"])
                    nd_ni = agg(["gate2", "content_noninteractive", "ndcg@10"])
                    nd_xi = agg(["gate2", "content_x_player", "ndcg@10"])
                    perm = agg(["gate2", "content_x_player_permuted", "mae"])

                    def stable_gain(a, b, mode="lower"):
                        """`a` is the challenger, `b` the reference. Stable = every seed
                        beats the reference mean AND the gain exceeds the challenger's
                        own seed spread."""
                        if not a or not b:
                            return None
                        gain = (b["mean"] - a["mean"]) if mode == "lower" \
                            else (a["mean"] - b["mean"])
                        wins = [((bv - v) if mode == "lower" else (v - bv))
                                for v, bv in zip(a["values"], b["values"])]
                        return {"challenger": a["mean"], "reference": b["mean"],
                                "gain": round(gain, 4), "challenger_std": a["std"],
                                "all_seeds_win": bool(wins and all(w > 0 for w in wins)),
                                "gain_gt_std": bool(gain > a["std"]),
                                "stable_win": bool(wins and all(w > 0 for w in wins)
                                                   and gain > a["std"])}

                    out[key] = {
                        "n_seeds": len(entries),
                        "gate1_row_mae": g1_mae, "gate1_content_ridge": g1_ref,
                        "gate1_verdict": stable_gain(g1_mae, g1_ref, "lower"),
                        "gate2_noninteractive_cmae": g2_ni, "gate2_interactive_cmae": g2_xi,
                        "gate2_cmae_verdict": stable_gain(g2_xi, g2_ni, "lower"),
                        "gate2_noninteractive_ranksp": rk_ni, "gate2_interactive_ranksp": rk_xi,
                        "gate2_ranksp_verdict": stable_gain(rk_xi, rk_ni, "higher"),
                        "gate2_noninteractive_ndcg10": nd_ni, "gate2_interactive_ndcg10": nd_xi,
                        "gate2_permuted_mae": perm,
                    }
                    v = out[key]
                    cmae_win = bool(v["gate2_cmae_verdict"] and v["gate2_cmae_verdict"]["stable_win"])
                    rank_win = bool(v["gate2_ranksp_verdict"] and v["gate2_ranksp_verdict"]["stable_win"])
                    v["gate2_passed"] = bool(cmae_win and rank_win)
                    v["difficulty_only"] = bool(cmae_win and not rank_win)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4B feature-matrix experiment")
    ap.add_argument("--clients", default="beatoraja,lr2")
    ap.add_argument("--families", default="obj,msd,perm,obj+msd,obj+msd+perm,obj+msd+perm+v2")
    ap.add_argument("--common-families", default="obj,msd,perm",
                    help="families that DEFINE the common-rows layer. Keep this to the "
                         "high-coverage families: including a sparse one shrinks the "
                         "shared row set and changes the difficulty of the comparison")
    ap.add_argument("--splits", default="cold_chart")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--models", default="ridge,gbdt")
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--msd-cap", type=float, default=100.0, choices=[40.0, 100.0])
    ap.add_argument("--out", default="phase4b_features.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    common_families = [f.strip() for f in args.common_families.split(",") if f.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    clients = [c.strip() for c in args.clients.split(",") if c.strip()]

    df, msd, info = load_features(msd_cap=args.msd_cap)
    payload = {"data": info, "families": families, "splits": splits, "seeds": seeds,
               "models": models, "frac": args.frac, "msd_cap": args.msd_cap,
               "common_families": common_families,
               "ridge_alphas": list(RIDGE_ALPHAS), "gbdt_params": GBDT_KW,
               "no_imputation": True,
               "main_conclusion_layer": "common (all families on identical rows)",
               "clients": {}}

    for client in clients:
        cdf = df[df["client"] == client].reset_index(drop=True)
        if not len(cdf):
            continue
        print(f"[{client}] running {len(families)} families x {len(splits)} splits "
              f"x {len(seeds)} seeds x {len(models)} models ...", flush=True)
        block = run_client(cdf, families, info, splits, seeds, models, args.frac,
                           args.verbose, common_families=common_families)
        block["summary"] = summarise(block, families, splits, models)
        if block["players"] < 5:
            block["identifiability"] = (
                f"only {block['players']} players: numbers are recorded but NO statistical "
                "verdict is drawn (PHASE4B_PLAN.md sec 3.3)")
        payload["clients"][client] = block
        print(f"  done: {len(block['summary'])} result cells")

    out = write_result(args.out,
                       header(script="run_phase4b_features.py",
                              data_config={"cross_section": str(CROSS_SECTION.relative_to(ROOT)),
                                           "target": TARGET,
                                           "min_judgements": MIN_JUDGEMENTS,
                                           "msd_cap": args.msd_cap,
                                           "msd_file": info["families"]["msd"]["source"],
                                           "families": families},
                              client=",".join(clients), split=",".join(splits),
                              model=",".join(models), seed=seeds),
                       payload)

    print(f"\nwrote {out.relative_to(ROOT)}")
    for client, cb in payload["clients"].items():
        print(f"== {client}: players={cb['players']} rows={cb['rows']}")
        print(f"   coverage: {json_cov(cb)}")
        for key, v in sorted(cb["summary"].items()):
            g1 = v.get("gate1_verdict")
            g2c = v.get("gate2_cmae_verdict")
            print(f"   {key:<52} G1gain={g1['gain'] if g1 else None}"
                  f" stable={g1['stable_win'] if g1 else None}"
                  f" | G2pass={v['gate2_passed']} diff_only={v['difficulty_only']}"
                  f" (cmae_gain={g2c['gain'] if g2c else None})")


def json_cov(cb: dict) -> str:
    c = cb.get("coverage", {})
    return " ".join(f"{k}={v.get('rows', v.get('rows_with_all_features'))}"
                    for k, v in c.items())


if __name__ == "__main__":
    main()
