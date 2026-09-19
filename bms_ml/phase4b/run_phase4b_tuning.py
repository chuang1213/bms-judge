"""Phase 4B closing experiment B — GBDT hyper-parameter tuning under nested CV.

WHY THIS EXISTS
  The Phase 4B headline (`obj+msd` GBDT, cold-chart MAE 5.929) used UNTUNED
  hyper-parameters (max_iter=300, lr=0.06, depth=3). A fair reading of "is 5.93 the
  ceiling, or are we just badly tuned?" needs the tuning done properly:

    * the search runs ONLY on train + validation, never on the outer test split;
    * the outer test split is scored exactly ONCE per seed, after the config is frozen;
    * the tuned number is compared against the untuned number and against the seed
      spread. If the gain is smaller than the seed spread, "untuned" was NOT the
      bottleneck and ~5.93 should be accepted.

  This is deliberately a small grid (18 configs x 3 inner folds) - enough to detect a
  real improvement, cheap enough to stay honest.

SCOPE: only the frozen main feature set `obj + msd` (33 columns), per user decision 1.
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
from metrics import mae, player_train_means  # noqa: E402
from provenance import header, write_result  # noqa: E402
from run_phase4b_features import (DATASET, TARGET, coverage,  # noqa: E402
                                  family_cols, load_features)
from splits import split_mask  # noqa: E402

UNTUNED = dict(max_iter=300, learning_rate=0.06, max_depth=3)
GRID = {
    "max_iter": [200, 400],
    "learning_rate": [0.04, 0.08],
    "max_depth": [2, 3, 6],
    "min_samples_leaf": [20, 50],
}
N_INNER_FOLDS = 3


def chart_frame(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], means: dict,
                mu: float) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def score_rows(te: pd.DataFrame, cd_te: pd.DataFrame, resid: np.ndarray,
               means: dict, mu: float) -> tuple[np.ndarray, np.ndarray]:
    base = te["player"].map(means).astype(float).fillna(mu).to_numpy()
    rmap = dict(zip(cd_te["sha256"], resid))
    pred = np.clip(base + te["sha256"].map(rmap).astype(float).fillna(0.0).to_numpy(), 0, 100)
    return te[TARGET].to_numpy(float), pred


def inner_cv_mae(cd_tr: pd.DataFrame, cols: list[str], cfg: dict, seed: int) -> float:
    """Chart-disjoint inner CV inside the TRAIN charts only."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    charts = cd_tr["sha256"].to_numpy()
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, N_INNER_FOLDS, size=len(cd_tr))
    errs = []
    y = cd_tr["mean_resid"].to_numpy(float)
    X = cd_tr[cols].to_numpy(float)
    for f in range(N_INNER_FOLDS):
        m = fold == f
        if m.sum() < 5 or (~m).sum() < 20:
            continue
        model = HistGradientBoostingRegressor(random_state=seed, **cfg)
        model.fit(X[~m], y[~m])
        errs.append(float(np.mean(np.abs(model.predict(X[m]) - y[m]))))
    return float(np.mean(errs)) if errs else np.inf


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4B closing experiment B: nested-CV tuning")
    ap.add_argument("--clients", default="beatoraja,lr2")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--split", default="cold_chart")
    ap.add_argument("--features", default="obj+msd")
    ap.add_argument("--out", default="phase4b_tuning.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    clients = [c.strip() for c in args.clients.split(",") if c.strip()]
    df, _msd, info = load_features(msd_cap=100.0)
    cols = family_cols(args.features, info)
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"feature set {args.features}: {len(cols)} cols; grid = {len(combos)} configs "
          f"x {N_INNER_FOLDS} inner folds")

    payload = {"features": args.features, "n_features": len(cols), "split_name": args.split,
               "seeds": seeds, "frac": args.frac, "grid_size": len(combos),
               "grid": GRID, "untuned_config": UNTUNED, "inner_folds": N_INNER_FOLDS,
               "selection": "inner CV on TRAIN charts only; outer test scored once per seed",
               "clients": {}}

    for client in clients:
        cdf = df[df["client"] == client].reset_index(drop=True)
        if not len(cdf):
            continue
        cov = coverage(cdf, cols)
        sub = cdf[cov].reset_index(drop=True)
        cb = {"rows_full": int(len(cdf)), "rows_with_features": int(len(sub)),
              "row_coverage": round(float(cov.mean()), 4),
              "players": int(sub["player"].nunique()), "per_seed": {}}
        print(f"\n[{client}] {len(sub)} rows, {sub['player'].nunique()} players", flush=True)
        for seed in seeds:
            test = split_mask(sub, args.split, seed, args.frac)
            if test.sum() < 10 or (~test).sum() < 10:
                cb["per_seed"][str(seed)] = {"skipped": "split too small"}
                continue
            tr = sub[~test].reset_index(drop=True)
            te = sub[test].reset_index(drop=True)
            means = player_train_means(tr, TARGET)
            mu = float(tr[TARGET].mean())
            cd_tr, cd_te = chart_frame(tr, te, cols, means, mu)

            # ---- pick config on inner CV (train charts only)
            scored = []
            for cfg in combos:
                v = inner_cv_mae(cd_tr, cols, cfg, seed)
                scored.append((v, cfg))
            scored.sort(key=lambda t: t[0])
            best_val, best_cfg = scored[0]

            from sklearn.ensemble import HistGradientBoostingRegressor
            y = te[TARGET].to_numpy(float)
            base = te["player"].map(means).astype(float).fillna(mu).to_numpy()

            # ---- TEST IS TOUCHED ONCE, HERE
            res = {}
            for name, cfg in (("untuned", UNTUNED), ("tuned", best_cfg)):
                m = HistGradientBoostingRegressor(random_state=seed, **cfg)
                m.fit(cd_tr[cols].to_numpy(float), cd_tr["mean_resid"].to_numpy(float))
                resid = m.predict(cd_te[cols].to_numpy(float))
                yv, pred = score_rows(te, cd_te, resid, means, mu)
                res[name] = {"mae": round(mae(yv, pred), 4),
                             "mae_within_player": round(mae(yv - base, pred - base), 4),
                             "config": cfg}
            # reference baseline on the same rows
            from sklearn.linear_model import Ridge
            Xr = cd_tr[cols].to_numpy(float)
            Xr_te = cd_te[cols].to_numpy(float)
            m_ = ~np.isnan(Xr).any(axis=1)
            mu_r, sd_r = Xr[m_].mean(axis=0), Xr[m_].std(axis=0)
            sd_r[sd_r == 0] = 1.0
            ref = Ridge(alpha=1.0).fit((Xr[m_] - mu_r) / sd_r,
                                       cd_tr.loc[m_, "mean_resid"].to_numpy(float))
            rresid = np.zeros(len(cd_te))
            keep = ~np.isnan(Xr_te).any(axis=1)
            if keep.any():
                rresid[keep] = ref.predict((Xr_te[keep] - mu_r) / sd_r)
            _, ref_pred = score_rows(te, cd_te, rresid, means, mu)
            res["content_ridge"] = {"mae": round(mae(y, ref_pred), 4)}
            res["gain_tuned_vs_untuned"] = round(res["untuned"]["mae"] - res["tuned"]["mae"], 4)
            res["best_inner_cv_mae"] = round(best_val, 4)
            res["top3_configs"] = [{"inner_cv_mae": round(v, 4), **c} for v, c in scored[:3]]
            cb["per_seed"][str(seed)] = res
            print(f"  seed={seed}: untuned={res['untuned']['mae']:.4f} "
                  f"tuned={res['tuned']['mae']:.4f} "
                  f"(gain {res['gain_tuned_vs_untuned']:+.4f}) "
                  f"ref={res['content_ridge']['mae']:.4f} cfg={best_cfg}", flush=True)

        usable = [v for v in cb["per_seed"].values() if "untuned" in v]
        if usable:
            un = [v["untuned"]["mae"] for v in usable]
            tu = [v["tuned"]["mae"] for v in usable]
            gains = [v["untuned"]["mae"] - v["tuned"]["mae"] for v in usable]
            cb["summary"] = {
                "untuned_mean": round(float(np.mean(un)), 4),
                "untuned_std": round(float(np.std(un, ddof=1)) if len(un) > 1 else 0.0, 4),
                "tuned_mean": round(float(np.mean(tu)), 4),
                "tuned_std": round(float(np.std(tu, ddof=1)) if len(tu) > 1 else 0.0, 4),
                "gain_mean": round(float(np.mean(gains)), 4),
                "gain_per_seed": [round(g, 4) for g in gains],
                "all_seeds_improved": bool(all(g > 0 for g in gains)),
                "gain_exceeds_untuned_seed_std": bool(
                    float(np.mean(gains)) > (float(np.std(un, ddof=1)) if len(un) > 1 else 0.0)),
                "reference_mean": round(float(np.mean(
                    [v["content_ridge"]["mae"] for v in usable])), 4),
                "chosen_configs": [v["tuned"]["config"] for v in usable],
            }
            s = cb["summary"]
            s["verdict"] = ("tuning is NOT the bottleneck: gain is inside the seed spread, "
                            "accept the untuned result"
                            if not (s["all_seeds_improved"]
                                    and s["gain_exceeds_untuned_seed_std"])
                            else "tuning helps beyond seed spread")
            print(f"  -> untuned {s['untuned_mean']}±{s['untuned_std']} "
                  f"tuned {s['tuned_mean']}±{s['tuned_std']} gain={s['gain_mean']}")
            print(f"  -> {s['verdict']}")
        payload["clients"][client] = cb

    out = write_result(args.out,
                       header(script="run_phase4b_tuning.py",
                              data_config={"features": args.features, "msd_cap": 100.0,
                                           "split": args.split, "frac": args.frac},
                              client=",".join(clients), split=args.split,
                              model="gbdt nested-CV", seed=seeds),
                       payload)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
