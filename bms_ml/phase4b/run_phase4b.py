"""Phase 4B / cold_chart — the cold-chart CONTENT track (an explicitly separate branch).

This is NOT "M4 because M3 passed". M3 (biased matrix factorisation) FAILED and the
Phase 4 collaborative-filtering route is recorded as a negative result
(see PHASE4_M3_REPORT.md). Phase 4B exists because on the `cold_chart` split the
chart-content baseline was the one stable positive signal, and the user approved
researching exactly that, nothing more.

SCOPE (PHASE4_PROTOCOL §11.2)
  * only charts NOBODY in the fit has played (cold_chart);
  * no matrix completion, no latent factors;
  * LR2 and beatoraja reported separately, never pooled;
  * no hand-defined speed/stamina/LN skill axes;
  * objective statistics (`cs_*`) and MinerCalc MSD may be used as a baseline or as
    input, but they are NOT presented as "the" chart representation.

THE TWO GATES (§11.4)
  Gate 1 - chart-level difficulty: predict a chart's mean performance and BEAT the
           content_ridge baseline (6.243 on beatoraja), not merely the degenerate
           player-mean baseline (6.749). >=3 seeds, gain > seed spread.
  Gate 2 - player x chart interaction: the model must give DIFFERENT predictions for
           different players from the same chart content. Measured with player-internal
           MAE / Spearman / top-k NDCG. If only chart-level MAE improves while
           player-internal ordering does not, the verdict is "difficulty prediction
           only, not interaction" and Gate 2 FAILS.

WHY GATE 1 IS EVALUATED ON A RESIDUAL TARGET
  The naive comparison (fit content on raw acc, score rows) is confounded: a player's
  own level explains most of a row's accuracy, so a "content" model can win by
  reproducing player main effects. Both candidates are therefore reduced to the same
  comparable quantity: predict a chart's mean residual AFTER removing the skill of the
  players who rated it, then add the evaluated player's own effect back. The player
  term is identical across candidates by construction, so the comparison isolates
  content.

USAGE
  python bms_ml/phase4b/run_phase4b.py
  python bms_ml/phase4b/run_phase4b.py --clients beatoraja --seeds 0,1,2 --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PHASE4 = Path(__file__).resolve().parents[1] / "phase4"
sys.path.insert(0, str(PHASE4))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_matrix import MIN_JUDGEMENTS, load_frame  # noqa: E402
from metrics import (bias, mae, player_train_means, ranking_by_player,  # noqa: E402
                     rmse, spearman)
from provenance import ROOT, header, write_result  # noqa: E402
from splits import parse_split, split_mask  # noqa: E402

DEFAULT_PARQUET = ROOT / "bms_ml" / "output" / "phase4" / "dataset" / "cross_section.parquet"
MSD_PARQUET = ROOT / "bms_ml" / "output" / "phase3" / "dataset" / "msd.parquet"
TARGET = "acc"
# the frozen Phase 4 baselines (§11.3) - Gate 1 must beat CONTENT_RIDGE_REF
DEGENERATE_REF = 6.749
CONTENT_RIDGE_REF = 6.243
MIN_TEST_ROWS = 10
CONTENT_FEATURE_SETS = ("content", "msd", "content+msd")


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_phase4b(parquet: Path, min_judgements: int) -> tuple[pd.DataFrame, dict]:
    """Phase 4 rows + optional MSD columns (left join on sha256)."""
    df, filt = load_frame(parquet, TARGET, min_judgements)
    info = {"phase4_filter": filt}
    if MSD_PARQUET.exists():
        msd = pd.read_parquet(MSD_PARQUET)
        msd_cols = [c for c in msd.columns if c.startswith("msd_")]
        df = df.merge(msd[["sha256"] + msd_cols], on="sha256", how="left")
        info["msd"] = {
            "path": str(MSD_PARQUET.relative_to(ROOT)),
            "axes": msd_cols,
            "rows_in_msd": int(len(msd)),
            "coverage_of_rows": round(float(df[msd_cols[0]].notna().mean()), 4),
            "coverage_of_charts": round(
                float(df.loc[df[msd_cols[0]].notna(), "sha256"].nunique()
                      / max(df["sha256"].nunique(), 1)), 4),
            "note": "MSD covers only a subset of the corpus; models using it are fitted "
                    "and scored on the subset where it is present",
        }
    else:
        info["msd"] = {"path": None, "note": "msd.parquet absent; MSD feature sets skipped"}
    info["content_features"] = [c for c in df.columns if c.startswith("cs_")]
    return df, info


def feature_cols(df: pd.DataFrame, kind: str, info: dict) -> list[str]:
    cs = info["content_features"]
    msd = info["msd"].get("axes", [])
    if kind == "content":
        return cs
    if kind == "msd":
        return msd
    if kind == "content+msd":
        return cs + msd
    if kind == "none":
        return []
    raise ValueError(f"unknown feature set {kind}")


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd[sd == 0] = 1.0
    return (train - mu) / sd, (test - mu) / sd


# --------------------------------------------------------------------------- #
# Gate 1: chart-level difficulty
# --------------------------------------------------------------------------- #
def chart_difficulty_frame(tr: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """One row per train chart: (mean residual, mean skill of its raters, features).

    residual      = acc - player_train_mean(player)   (player skill removed)
    mean_skill    = mean over raters of their train mean  (the composition control)
    """
    means = player_train_means(tr, TARGET)
    d = tr.copy()
    d["_pmean"] = d["player"].map(means).astype(float)
    d["_resid"] = d[TARGET] - d["_pmean"]
    g = d.groupby("sha256", sort=True)
    out = g.agg(n_raters=("_resid", "size"), mean_resid=("_resid", "mean"),
                mean_skill=("_pmean", "mean"))
    if feats:
        out = out.join(g[feats].first())
    return out.reset_index()


def fit_predict_difficulty(cd_tr: pd.DataFrame, cd_te: pd.DataFrame, feats: list[str],
                           model: str, alpha: float = 1.0, seed: int = 0) -> np.ndarray:
    """Predict chart mean residuals for `cd_te`, fitted on `cd_tr`.

    `cd_te` may legitimately lack the label column (held-out charts have no
    observed residual by construction), so only the features are required.
    """
    if model == "mean":
        return np.full(len(cd_te), float(cd_tr["mean_resid"].mean()))
    missing = [f for f in feats if f not in cd_te.columns]
    if missing:
        raise KeyError(f"test chart frame is missing features {missing[:4]}... "
                       f"it must be built from the same source rows")
    Xtr = cd_tr[feats].to_numpy(float)
    Xte = cd_te[feats].to_numpy(float)
    if model == "ridge":
        from sklearn.linear_model import Ridge
        # composition control: without it the model can "predict" difficulty by
        # learning which players rated the chart
        Xtr = np.column_stack([Xtr, cd_tr["mean_skill"].to_numpy(float)])
        Xte = np.column_stack([Xte, cd_te["mean_skill"].to_numpy(float)])
        Xtr, Xte = standardize(Xtr, Xte)
        return Ridge(alpha=alpha).fit(Xtr, cd_tr["mean_resid"].to_numpy(float)).predict(Xte)
    if model == "gbdt":
        from sklearn.ensemble import HistGradientBoostingRegressor
        Xtr = np.column_stack([Xtr, cd_tr["mean_skill"].to_numpy(float)])
        Xte = np.column_stack([Xte, cd_te["mean_skill"].to_numpy(float)])
        m = HistGradientBoostingRegressor(random_state=seed, max_iter=200,
                                          learning_rate=0.06, max_depth=3)
        return m.fit(Xtr, cd_tr["mean_resid"].to_numpy(float)).predict(Xte)
    raise ValueError(model)


def gate1(cdf: pd.DataFrame, split: str, seeds: list[int], info: dict,
          frac: float, verbose: bool) -> dict:
    """Gate 1: can chart CONTENT predict a cold chart's difficulty?

    Each feature set is its own TRACK with its own evaluation set. MSD covers only a
    slice of the corpus, so intersecting all candidates onto shared rows would shrink
    the test set to nothing; instead every track records exactly which rows, charts
    and players it was scored on, and only tracks scored on comparable sets should be
    read against each other.

    The target is the chart's mean residual AFTER removing the skill of the players
    who rated it, so a model cannot "predict difficulty" by learning who rated what.
    """
    block = {"split": split, "tracks": {}}
    tracks = ["none"] + [fs for fs in CONTENT_FEATURE_SETS
                         if not (fs == "msd" and not info["msd"].get("axes"))]

    for seed in seeds:
        test = split_mask(cdf, split, seed, frac)
        if test.sum() < MIN_TEST_ROWS or (~test).sum() < MIN_TEST_ROWS:
            block.setdefault("per_seed", {})[str(seed)] = {"skipped": "split too small"}
            continue
        tr_all = cdf[~test].reset_index(drop=True)
        te_all = cdf[test].reset_index(drop=True)

        for fs in tracks:
            feats = feature_cols(tr_all, fs, info)
            tr = tr_all[tr_all[feats].notna().all(axis=1)].reset_index(drop=True) if feats \
                else tr_all
            te = te_all[te_all[feats].notna().all(axis=1)].reset_index(drop=True) if feats \
                else te_all
            if len(te) < MIN_TEST_ROWS or len(tr) < MIN_TEST_ROWS:
                block["tracks"].setdefault(fs, {}).setdefault("per_seed", {})[str(seed)] = {
                    "skipped": "track evaluation set too small",
                    "n_test": int(len(te)), "n_train": int(len(tr))}
                continue
            means = player_train_means(tr, TARGET)
            mu = float(tr[TARGET].mean())
            base = te["player"].map(means).astype(float).fillna(mu).to_numpy()
            y = te[TARGET].to_numpy(float)
            g_te = te.groupby("sha256", sort=True)
            comp = g_te["player"].apply(
                lambda s: float(np.mean([means.get(pl, mu) for pl in s])))
            cd_tr = chart_difficulty_frame(tr, feats)
            cd_te = comp.rename("mean_skill").to_frame()
            if feats:
                cd_te = cd_te.join(g_te[feats].first())
            cd_te = cd_te.reset_index().dropna(subset=feats if feats else ["mean_skill"])

            seed_res = {"_baseline_player_mean_mae": round(mae(y, base), 4)}
            # In-protocol reference: the frozen content_ridge baseline re-measured on
            # EXACTLY these rows, with the same player means. The published 6.243 was
            # measured on the full cold_chart test set of beatoraja under a slightly
            # different row filter, so it is not a fair yardstick for another client
            # or another row subset; this one is.
            from sklearn.linear_model import Ridge as _Ridge
            if feats:
                Xtr = cd_tr[feats].to_numpy(float)
                Xte_ref = cd_te[feats].to_numpy(float)
                Xtr_s, Xte_s = standardize(Xtr, Xte_ref)
                ref_model = _Ridge(alpha=1.0).fit(Xtr_s, cd_tr["mean_resid"].to_numpy(float))
                ref_resid = dict(zip(cd_te["sha256"], ref_model.predict(Xte_s)))
                keep_ref = te["sha256"].isin(ref_resid).to_numpy()
                ref_pred = np.clip(base[keep_ref] + te.loc[keep_ref, "sha256"]
                                   .map(ref_resid).astype(float).to_numpy(), 0, 100)
                seed_res["_content_ridge_in_protocol"] = round(mae(y[keep_ref], ref_pred), 4)
                seed_res["_content_ridge_rows"] = int(keep_ref.sum())
            for model in (["mean"] if fs == "none" else ["mean", "ridge", "gbdt"]):
                pred_resid = fit_predict_difficulty(cd_tr, cd_te, feats, model, 1.0, seed)
                resid_map = dict(zip(cd_te["sha256"], pred_resid))
                keep = te["sha256"].isin(resid_map).to_numpy()
                yk = y[keep]
                pred = np.clip(base[keep] + te.loc[keep, "sha256"]
                               .map(resid_map).astype(float).to_numpy(), 0, 100)
                # a chart-content model predicts a per-chart constant, so its
                # within-player MAE is necessarily unchanged; that column exists to
                # make that fact visible rather than to look like a win
                seed_res[model] = {
                    "n_test_rows": int(keep.sum()),
                    "n_test_charts": int(len(cd_te)),
                    "n_test_players": int(te.loc[keep, "player"].nunique()),
                    "row_mae": round(mae(yk, pred), 4),
                    "row_rmse": round(rmse(yk, pred), 4),
                    "row_mae_within_player": round(
                        mae(yk - base[keep], pred - base[keep]), 4),
                }
                if verbose:
                    print(f"        {fs:<12} {model:<6} rows={int(keep.sum()):>5} "
                          f"row_mae={seed_res[model]['row_mae']:.4f}")
            t = block["tracks"].setdefault(fs, {"per_seed": {}})
            t["per_seed"][str(seed)] = seed_res
            t["evaluation_set"] = {
                "rows": int(len(te)), "charts": int(te["sha256"].nunique()),
                "players": int(te["player"].nunique())}
            t["n_features"] = len(feats)

    for fs, t in block["tracks"].items():
        usable = [s for s in t.get("per_seed", {}).values() if s and "skipped" not in s]
        for model in ("mean", "ridge", "gbdt"):
            vals = [s[model] for s in usable if model in s]
            if not vals:
                continue
            t.setdefault("summary", {})[model] = {
                "row_mae_mean": round(float(np.mean([v["row_mae"] for v in vals])), 4),
                "row_mae_std": round(float(np.std([v["row_mae"] for v in vals], ddof=1))
                                     if len(vals) > 1 else 0.0, 4),
                "n_seeds": len(vals),
            }
        refs = [s["_content_ridge_in_protocol"] for s in usable
                if s.get("_content_ridge_in_protocol") is not None]
        if refs:
            t["summary"]["_content_ridge_in_protocol"] = {
                "row_mae_mean": round(float(np.mean(refs)), 4),
                "row_mae_std": round(float(np.std(refs, ddof=1)) if len(refs) > 1 else 0.0, 4),
                "n_seeds": len(refs),
                "note": "the frozen content_ridge baseline re-measured on THIS track's rows",
            }
    return block


# --------------------------------------------------------------------------- #
# Gate 2: player x chart interaction
# --------------------------------------------------------------------------- #
def interaction_matrix(X: np.ndarray, players: np.ndarray) -> np.ndarray:
    """[X | one-hot(player) * X]: lets each player weight the content differently."""
    if X.shape[1] == 0:
        return X
    out = [X]
    for p in np.unique(players):
        m = (players == p).astype(float)[:, None]
        out.append(X * m)
    return np.hstack(out)


def gate2(cdf: pd.DataFrame, split: str, seeds: list[int], info: dict,
          frac: float, alpha: float, verbose: bool) -> dict:
    """Non-interactive vs interactive content models on player-internal metrics."""
    from sklearn.linear_model import Ridge

    block = {"split": split, "per_seed": {}, "across_seeds": {}}
    for seed in seeds:
        test = split_mask(cdf, split, seed, frac)
        if test.sum() < MIN_TEST_ROWS or (~test).sum() < MIN_TEST_ROWS:
            block["per_seed"][str(seed)] = {"skipped": "split too small"}
            continue
        tr = cdf[~test].reset_index(drop=True)
        te = cdf[test].reset_index(drop=True)
        means = player_train_means(tr, TARGET)
        mu = float(tr[TARGET].mean())
        feats = feature_cols(tr, "content", info)
        Xtr_raw = tr[feats].to_numpy(float)
        Xte_raw = te[feats].to_numpy(float)
        Xtr, Xte = standardize(Xtr_raw, Xte_raw)
        base_tr = tr["player"].map(means).astype(float).fillna(mu).to_numpy()
        base_te = te["player"].map(means).astype(float).fillna(mu).to_numpy()
        ytr = tr[TARGET].to_numpy(float)
        y = te[TARGET].to_numpy(float)

        res = {}
        # --- M0: player mean only (the degenerate baseline)
        res["player_mean"] = evaluate_row(te, base_te, y, means)

        # --- M1: content, non-interactive
        m1 = Ridge(alpha=alpha).fit(Xtr, ytr - base_tr)
        p1 = np.clip(base_te + m1.predict(Xte), 0, 100)
        res["content_noninteractive"] = evaluate_row(te, p1, y, means)

        # --- M2: content x player (interaction)
        Itr = interaction_matrix(Xtr, tr["player"].to_numpy())
        Ite = interaction_matrix(Xte, te["player"].to_numpy())
        m2 = Ridge(alpha=alpha).fit(Itr, ytr - base_tr)
        p2 = np.clip(base_te + m2.predict(Ite), 0, 100)
        res["content_x_player"] = evaluate_row(te, p2, y, means)

        # --- M3: GBDT with content + player identity
        from sklearn.ensemble import HistGradientBoostingRegressor
        pids = {p: i for i, p in enumerate(sorted(tr["player"].unique()))}
        Ztr = np.column_stack([Xtr, tr["player"].map(pids).fillna(-1).to_numpy()])
        Zte = np.column_stack([Xte, te["player"].map(pids).fillna(-1).to_numpy()])
        g = HistGradientBoostingRegressor(random_state=seed, max_iter=300,
                                          learning_rate=0.06, max_depth=3,
                                          categorical_features=[Ztr.shape[1] - 1])
        g.fit(Ztr, ytr - base_tr)
        p3 = np.clip(base_te + g.predict(Zte), 0, 100)
        res["gbdt_content_plus_player"] = evaluate_row(te, p3, y, means)

        # --- M4: can the INTERACTION rank within players at all? Compare against the
        # non-interactive model on player-internal metrics (the actual Gate 2 test).
        for name in ("content_noninteractive", "content_x_player", "gbdt_content_plus_player"):
            res[name]["beats_noninteractive_centered"] = bool(
                res[name]["player_centered_mae"] < res["content_noninteractive"]["player_centered_mae"])
        res["gate2_interaction_delta"] = {
            "player_centered_mae": round(
                res["content_noninteractive"]["player_centered_mae"]
                - res["content_x_player"]["player_centered_mae"], 4),
            "player_rank_spearman": round(
                (res["content_x_player"]["player_rank_spearman"] or 0)
                - (res["content_noninteractive"]["player_rank_spearman"] or 0), 4),
        }

        # --- permutation guard: break the content<->player pairing on the test rows
        # and re-predict. If the "interaction" was real, shuffling content between
        # players must destroy the advantage; if the number barely moves, the gain
        # came from something other than player-specific content response.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(te))
        p2_perm = np.clip(base_te + m2.predict(Ite[perm]), 0, 100)
        res["content_x_player_permuted"] = evaluate_row(te, p2_perm, y, means)
        res["permutation_check"] = {
            "interactive_mae": res["content_x_player"]["mae"],
            "permuted_mae": res["content_x_player_permuted"]["mae"],
            "noninteractive_mae": res["content_noninteractive"]["mae"],
            "note": ("shuffling chart content across test players must hurt the "
                     "interactive model if its gain is genuinely player-specific"),
        }
        block["per_seed"][str(seed)] = res

    usable = [s for s in block["per_seed"].values() if s and "skipped" not in s]
    for name in ("player_mean", "content_noninteractive", "content_x_player",
                 "gbdt_content_plus_player"):
        for metric in ("mae", "player_centered_mae", "player_rank_spearman",
                       "ndcg@10", "ndcg@20"):
            vals = [r[name][metric] for r in usable
                    if name in r and r[name].get(metric) is not None]
            if not vals:
                continue
            block["across_seeds"].setdefault(name, {})[metric] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
                "values": [round(float(v), 4) for v in vals]}
    return block


def evaluate_row(te: pd.DataFrame, pred: np.ndarray, y: np.ndarray, means: dict) -> dict:
    base = te["player"].map(means).astype(float).fillna(float(np.mean(list(means.values())))).to_numpy()
    dev_y, dev_p = y - base, pred - base
    r = {
        "mae": round(mae(y, pred), 4),
        "rmse": round(rmse(y, pred), 4),
        "player_centered_mae": round(mae(dev_y, dev_p), 4),
        "player_centered_spearman": (lambda s: round(s, 4) if s is not None else None)(
            spearman(dev_y, dev_p)),
    }
    rk = ranking_by_player(te, pred, TARGET)
    r["player_rank_spearman"] = rk["aggregate"].get("spearman", {}).get("mean")
    for k in (10, 20):
        r[f"ndcg@{k}"] = rk["aggregate"].get(f"ndcg@{k}", {}).get("mean")
    return r


# --------------------------------------------------------------------------- #
# verdicts
# --------------------------------------------------------------------------- #
def gate1_verdict(block: dict) -> dict:
    """Gate 1 verdict per feature-set track, against the frozen content_ridge ref.

    Only the `content` track is a like-for-like comparison with the reference. MSD
    tracks cover a small (and probably easier) slice of the corpus, so a better number
    there is NOT automatically a win.
    """
    out = {"content_ridge_reference": CONTENT_RIDGE_REF,
           "degenerate_reference": DEGENERATE_REF, "tracks": {}, "passed": False,
           "caveat": ("the content_ridge reference (6.243) was measured on the FULL "
                      "cold_chart test set; a track scored on a smaller MSD-covered "
                      "subset is not directly comparable to it")}
    for fs, t in block.get("tracks", {}).items():
        summ = t.get("summary", {})
        if not summ:
            continue
        ranked = sorted(summ.items(), key=lambda kv: kv[1]["row_mae_mean"])
        model, best = ranked[0]
        gain = CONTENT_RIDGE_REF - best["row_mae_mean"]
        rec = {"best_model": model, "row_mae_mean": best["row_mae_mean"],
               "row_mae_std": best["row_mae_std"], "n_seeds": best["n_seeds"],
               "n_features": t.get("n_features"),
               "evaluation_set": t.get("evaluation_set"),
               "all_models": {k: v["row_mae_mean"] for k, v in ranked},
               "gain_over_content_ridge": round(gain, 4),
               "gain_exceeds_seed_std": bool(gain > best["row_mae_std"]),
               "beats_content_ridge": bool(best["row_mae_mean"] < CONTENT_RIDGE_REF)}
        out["tracks"][fs] = rec
        if fs == "content":
            # the like-for-like track decides the gate
            out["passed"] = bool(rec["beats_content_ridge"] and rec["gain_exceeds_seed_std"])
            out["decided_by_track"] = "content"
            out["gain_over_content_ridge"] = rec["gain_over_content_ridge"]
    return out


def gate2_verdict(block: dict) -> dict:
    a = block.get("across_seeds", {})
    if "content_noninteractive" not in a or "content_x_player" not in a:
        return {"passed": False, "reason": "no usable seeds"}
    ni, xi = a["content_noninteractive"], a["content_x_player"]
    out = {"passed": False, "detail": {}}
    for metric, mode in (("mae", "lower"), ("player_centered_mae", "lower"),
                         ("player_rank_spearman", "higher"), ("ndcg@10", "higher")):
        m, n = xi.get(metric), ni.get(metric)
        if not m or not n:
            continue
        gain = (n["mean"] - m["mean"]) if mode == "lower" else (m["mean"] - n["mean"])
        better = all(((n["mean"] - v) if mode == "lower" else (v - n["mean"])) > 0
                     for v in m["values"])
        out["detail"][metric] = {
            "interactive": m["mean"], "noninteractive": n["mean"],
            "gain": round(gain, 4), "interactive_seed_std": m["std"],
            "all_seeds_better": bool(better),
            "gain_exceeds_seed_std": bool(gain > m["std"]),
            "win": bool(better and gain > m["std"]),
        }
    # A "difficulty only" verdict is: chart-level MAE improves while NO ranking metric
    # improves. Requiring BOTH rank metrics to fail was too strict and could label a
    # genuine interaction as difficulty-only (or vice versa) on one noisy metric.
    cmae = out["detail"].get("player_centered_mae", {})
    rank_wins = [out["detail"].get(m, {}).get("win", False)
                 for m in ("player_rank_spearman", "ndcg@10", "ndcg@20")
                 if m in out["detail"]]
    any_rank_win = any(rank_wins)
    out["difficulty_only"] = bool(cmae.get("win") and not any_rank_win)
    out["any_rank_metric_improves"] = bool(any_rank_win)
    # Gate 2 passes only if the model both reduces within-player error AND improves
    # within-player ordering - otherwise it is just chart difficulty prediction.
    out["passed"] = bool(cmae.get("win") and any_rank_win)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4B / cold_chart content track")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--clients", default="beatoraja,lr2")
    ap.add_argument("--splits", default="cold_chart")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default="phase4b_cold_chart.json")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    clients = [c.strip() for c in args.clients.split(",") if c.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    df, info = load_phase4b(args.parquet, MIN_JUDGEMENTS)
    payload = {"data": info, "target": TARGET, "seeds": seeds, "frac": args.frac,
               "alpha": args.alpha, "splits": splits,
               "frozen_baselines": {"degenerate_player_mean": DEGENERATE_REF,
                                    "content_ridge": CONTENT_RIDGE_REF},
               "track": "Phase 4B / cold_chart (separate branch; M3 remains FAILED)",
               "clients": {}}

    for client in clients:
        cdf = df[df["client"] == client].reset_index(drop=True)
        if not len(cdf):
            continue
        cb = {"rows": int(len(cdf)), "players": int(cdf["player"].nunique()),
              "charts": int(cdf["sha256"].nunique()), "gates": {}}
        print(f"[{client}] Gate 1 (chart difficulty) ...", flush=True)
        g1 = {}
        g2 = {}
        for split in splits:
            g1[split] = gate1(cdf, split, seeds, info, args.frac, args.verbose)
            print(f"[{client}] Gate 2 (player x chart) {split} ...", flush=True)
            g2[split] = gate2(cdf, split, seeds, info, args.frac, args.alpha, args.verbose)
        cb["gates"]["gate1_chart_difficulty"] = {
            "per_split": g1,
            "verdict": {s: gate1_verdict(b) for s, b in g1.items()},
        }
        cb["gates"]["gate2_interaction"] = {
            "per_split": g2,
            "verdict": {s: gate2_verdict(b) for s, b in g2.items()},
        }
        payload["clients"][client] = cb

    data_config = {"parquet": str(args.parquet.relative_to(ROOT)),
                   "min_judgements": MIN_JUDGEMENTS, "target": TARGET,
                   "msd": info.get("msd", {}), "alpha": args.alpha,
                   "note": "Phase 4B cold-chart content track; no matrix completion, "
                           "no time, no difficulty-table level, LR2/beatoraja separate"}
    out = write_result(args.out, header(script="run_phase4b.py", data_config=data_config,
                                        client=",".join(clients), split=",".join(splits),
                                        model="ridge/gbdt content vs content_ridge baseline",
                                        seed=seeds), payload)

    txt = [f"\nwrote {out.relative_to(ROOT)}"]
    for client, cb in payload["clients"].items():
        txt.append(f"== {client}: players={cb['players']} rows={cb['rows']}")
        v1 = cb["gates"]["gate1_chart_difficulty"]["verdict"]
        for split, v in v1.items():
            txt.append(f"   GATE1 {split}: {'PASS' if v.get('passed') else 'FAIL'} "
                       f"gain_over_content_ridge={v.get('gain_over_content_ridge')} "
                       f"(ref={CONTENT_RIDGE_REF})")
            for fs, rec in v.get("tracks", {}).items():
                txt.append(f"      track {fs:<12} best={rec['best_model']:<5} "
                           f"mae={rec['row_mae_mean']:.4f}±{rec['row_mae_std']:.4f} "
                           f"n_feat={rec['n_features']} "
                           f"beats_ref={rec['beats_content_ridge']}")
        v2 = cb["gates"]["gate2_interaction"]["verdict"]
        for split, v in v2.items():
            txt.append(f"   GATE2 {split}: {'PASS' if v.get('passed') else 'FAIL'} "
                       f"difficulty_only={v.get('difficulty_only')}")
            for metric, d in v.get("detail", {}).items():
                txt.append(f"      {metric:<26} inter={d['interactive']} "
                           f"noninter={d['noninteractive']} gain={d['gain']} "
                           f"win={d['win']}")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
