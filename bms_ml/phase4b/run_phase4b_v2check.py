"""Phase 4B closing experiment C — does v2 add anything ON ITS OWN ROWS?

WHY THIS EXISTS
  In the feature matrix, `v2` only exists for ~26% of rows, and those rows are an
  EASIER subset (reference MAE 6.60 vs 6.80). Comparing `obj+msd` (99.3% of rows) with
  `obj+msd+v2` (26% of rows) therefore mixes two changes at once: the extra feature
  AND a change of row set.

  This script removes the row-set confound: both candidates are scored on the SAME
  rows - exactly the rows where v2 is available - with the same split, same seeds and
  the same model. The user's decision 1 says v2 stays out of the main line unless it
  beats the seed spread here; otherwise it closes and becomes an appendix.

  Note this deliberately does NOT put v2 into the `common` definition (that would
  shrink the main comparison row set, which is exactly the trap found earlier).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bms_ml" / "phase4"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_matrix import MIN_JUDGEMENTS  # noqa: E402
from metrics import mae, player_train_means, ranking_by_player  # noqa: E402
from provenance import header, write_result  # noqa: E402
from run_phase4b_features import (TARGET, coverage, family_cols,  # noqa: E402
                                  fit_predict, interaction_matrix, load_features,
                                  row_metrics)
from splits import split_mask  # noqa: E402


def gate1(tr, te, cols, seed, model):
    """Cold-chart difficulty: chart mean residual, player skill removed."""
    means = player_train_means(tr, TARGET)
    mu = float(tr[TARGET].mean())
    d = tr.copy()
    d["_pmean"] = d["player"].map(means).astype(float)
    d["_resid"] = d[TARGET] - d["_pmean"]
    g = d.groupby("sha256", sort=True)
    cd_tr = g.agg(mean_resid=("_resid", "mean")).join(g[cols].first()).reset_index()
    comp = te.groupby("sha256", sort=True)["player"].apply(
        lambda s: float(np.mean([means.get(p, mu) for p in s])))
    cd_te = comp.rename("mean_skill").to_frame().join(
        te.groupby("sha256", sort=True)[cols].first()).reset_index()
    resid, cfg = fit_predict(model, cd_tr[cols].to_numpy(float),
                             cd_tr["mean_resid"].to_numpy(float),
                             cd_te[cols].to_numpy(float), seed)
    base = te["player"].map(means).astype(float).fillna(mu).to_numpy()
    rmap = dict(zip(cd_te["sha256"], resid))
    pred = np.clip(base + te["sha256"].map(rmap).astype(float).fillna(0.0).to_numpy(), 0, 100)
    return {"row_mae": round(mae(te[TARGET].to_numpy(float), pred), 4),
            "n_test_rows": int(len(te)), "config": cfg}


def gate2(tr, te, cols, seed, model):
    """Player x chart interaction, judged on player-internal metrics."""
    means = player_train_means(tr, TARGET)
    mu = float(tr[TARGET].mean())
    base_tr = tr["player"].map(means).astype(float).fillna(mu).to_numpy()
    base_te = te["player"].map(means).astype(float).fillna(mu).to_numpy()
    ytr = tr[TARGET].to_numpy(float)
    y = te[TARGET].to_numpy(float)
    mu_x = np.nanmean(tr[cols].to_numpy(float), axis=0)
    sd_x = np.nanstd(tr[cols].to_numpy(float), axis=0)
    sd_x[sd_x == 0] = 1.0
    Xtr = np.nan_to_num((tr[cols].to_numpy(float) - mu_x) / sd_x)
    Xte = np.nan_to_num((te[cols].to_numpy(float) - mu_x) / sd_x)
    p_ni, _ = fit_predict(model, Xtr, ytr - base_tr, Xte, seed)
    ni = row_metrics(te, np.clip(base_te + p_ni, 0, 100), means, y)
    Itr = interaction_matrix(Xtr, tr["player"].to_numpy())
    Ite = interaction_matrix(Xte, te["player"].to_numpy())
    p_xi, _ = fit_predict(model, Itr, ytr - base_tr, Ite, seed)
    xi = row_metrics(te, np.clip(base_te + p_xi, 0, 100), means, y)
    rng = np.random.default_rng(seed)
    p_perm, _ = fit_predict(model, Itr, ytr - base_tr, Ite[rng.permutation(len(te))], seed)
    perm = row_metrics(te, np.clip(base_te + p_perm, 0, 100), means, y)
    return {"noninteractive": ni, "interactive": xi, "permuted": perm}


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4B closing C: v2 same-row marginal")
    ap.add_argument("--clients", default="beatoraja,lr2")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--split", default="cold_chart")
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--models", default="ridge,gbdt")
    ap.add_argument("--base", default="obj+msd")
    ap.add_argument("--candidate", default="v2")
    ap.add_argument("--out", default="phase4b_v2_marginal.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    clients = [c.strip() for c in args.clients.split(",") if c.strip()]
    df, _msd, info = load_features(msd_cap=100.0)
    base_cols = family_cols(args.base, info)
    cand_cols = base_cols + family_cols(args.candidate, info)

    payload = {"base": args.base, "candidate": args.candidate,
               "base_n_cols": len(base_cols), "candidate_n_cols": len(cand_cols),
               "split_name": args.split, "seeds": seeds, "frac": args.frac,
               "models": models,
               "row_set": "rows where the CANDIDATE features exist (both arms identical)",
               "clients": {}}

    for client in clients:
        cdf = df[df["client"] == client].reset_index(drop=True)
        if not len(cdf):
            continue
        cov = coverage(cdf, cand_cols)     # the v2-available rows
        sub = cdf[cov].reset_index(drop=True)
        cb = {"rows_full": int(len(cdf)), "rows_common": int(len(sub)),
              "row_coverage": round(float(cov.mean()), 4),
              "players": int(sub["player"].nunique()),
              "charts": int(sub["sha256"].nunique()), "per_seed": {}}
        print(f"\n[{client}] common rows for base and candidate: {len(sub)} "
              f"({cb['row_coverage']:.1%})", flush=True)
        if len(sub) < 200:
            cb["skipped"] = "too few common rows"
            payload["clients"][client] = cb
            continue
        for seed in seeds:
            test = split_mask(sub, args.split, seed, args.frac)
            if test.sum() < 10 or (~test).sum() < 10:
                cb["per_seed"][str(seed)] = {"skipped": "split too small"}
                continue
            tr = sub[~test].reset_index(drop=True)
            te = sub[test].reset_index(drop=True)
            entry = {}
            for model in models:
                entry[model] = {
                    "gate1_base": gate1(tr, te, base_cols, seed, model),
                    "gate1_candidate": gate1(tr, te, cand_cols, seed, model),
                    "gate2_base": gate2(tr, te, base_cols, seed, model),
                    "gate2_candidate": gate2(tr, te, cand_cols, seed, model),
                }
            cb["per_seed"][str(seed)] = entry
            for model in models:
                b = entry[model]["gate1_base"]["row_mae"]
                c = entry[model]["gate1_candidate"]["row_mae"]
                cc = entry[model]["gate2_candidate"]["interactive"]["player_centered_mae"]
                bc = entry[model]["gate2_base"]["interactive"]["player_centered_mae"]
                print(f"  seed={seed} {model:<6} G1 base={b:.4f} +v2={c:.4f} ({b - c:+.4f}) "
                      f"| G2 cmae base={bc:.4f} +v2={cc:.4f} ({bc - cc:+.4f})", flush=True)

        usable = [v for v in cb["per_seed"].values() if isinstance(v, dict) and "gbdt" in v
                  or (isinstance(v, dict) and "ridge" in v)]
        cb["summary"] = {}
        for model in models:
            rows = [v[model] for v in cb["per_seed"].values()
                    if isinstance(v, dict) and model in v]
            if not rows:
                continue
            for arm, metric in (("gate1", "row_mae"),):
                b = [r[f"{arm}_base"][metric] for r in rows]
                c = [r[f"{arm}_candidate"][metric] for r in rows]
                gains = [x - y for x, y in zip(b, c)]
                cb["summary"][f"{model}_{arm}"] = {
                    "base_mean": round(float(np.mean(b)), 4),
                    "candidate_mean": round(float(np.mean(c)), 4),
                    "gain_mean": round(float(np.mean(gains)), 4),
                    "gain_per_seed": [round(g, 4) for g in gains],
                    "base_std": round(float(np.std(b, ddof=1)) if len(b) > 1 else 0.0, 4),
                    "all_seeds_improve": bool(all(g > 0 for g in gains)),
                    "gain_exceeds_std": bool(float(np.mean(gains))
                                             > (float(np.std(b, ddof=1)) if len(b) > 1 else 0.0)),
                }
                s = cb["summary"][f"{model}_{arm}"]
                s["v2_decision"] = ("keep (stable gain)" if s["all_seeds_improve"]
                                    and s["gain_exceeds_std"] else "CLOSE v2 (no stable gain)")
            for arm in ("gate2_base", "gate2_candidate"):
                vals = [r[arm]["interactive"]["player_centered_mae"] for r in rows]
                cb["summary"][f"{model}_{arm}_cmae"] = round(float(np.mean(vals)), 4)
                rk = [r[arm]["interactive"]["player_rank_spearman"] for r in rows]
                cb["summary"][f"{model}_{arm}_ranksp"] = round(
                    float(np.mean([v for v in rk if v is not None])), 4)
            d = [r["gate2_candidate"]["interactive"]["player_centered_mae"]
                 - r["gate2_base"]["interactive"]["player_centered_mae"] for r in rows]
            cb["summary"][f"{model}_gate2_cmae_delta"] = {
                "mean": round(float(np.mean(d)), 4),
                "per_seed": [round(x, 4) for x in d],
                "v2_decision": ("keep" if all(x < 0 for x in d)
                                else "CLOSE v2 (interaction not improved)"),
            }
        for k, v in cb["summary"].items():
            print(f"  SUMMARY {k}: {v}")
        payload["clients"][client] = cb

    out = write_result(args.out,
                       header(script="run_phase4b_v2check.py",
                              data_config={"base": args.base, "candidate": args.candidate,
                                           "msd_cap": 100.0, "split": args.split},
                              client=",".join(clients), split=args.split,
                              model=",".join(models), seed=seeds),
                       payload)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
