"""Phase 4 M3 driver — biased matrix factorisation (ALS) against the gate baseline.

The model is

    pred(player, chart) = mu + b_player + b_chart + <p_player, q_chart>

and nothing else: no time, no first plays, no recency, no difficulty-table level,
no hand-defined skill axis, and LR2 / beatoraja are fitted and scored separately.

WHAT THE PROTOCOL REQUIRES (PHASE4_PROTOCOL §5.2, §6)
-----------------------------------------------------
  1. random_interaction: beat the gate `player_chart_bias` stably;
  2. loo_k_charts_per_player (k=5,10, >=3 seeds): beat the gate, not swamped by
     seed spread;
  3. player-internal metrics (player_centered_mae, player_rank_spearman,
     player_topk_ndcg) must also beat the gate - a plain overall-MAE win is not a
     pass, because overall MAE is dominated by how strong each player is;
  4. cold_player: must not exceed the chart-mean baseline;
  5. if any of the above fails -> stop, report the negative result, do not start M4/M5.

HOW SELECTION AVOIDS THE TEST SET
---------------------------------
Hyper-parameters (dim, reg) are chosen on an INNER validation slice carved out of
the TRAIN rows only, once per (client, split), and then frozen for every seed of
that split. The held-out test rows are never used for any choice, so the reported
test numbers are not hyper-parameter-selected. `selection` records this.

Usage
-----
  python bms_ml/phase4/run_m3.py
  python bms_ml/phase4/run_m3.py --clients beatoraja --splits random_interaction --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_matrix import load_frame  # noqa: E402
from metrics import evaluate_predictions, player_train_means  # noqa: E402
from models import BaselineModel, ContentRidgeModel, MFModel  # noqa: E402
from provenance import CLIENTS, ROOT, header, write_result  # noqa: E402
from splits import (all_split_names, inner_validation_indices,  # noqa: E402
                    parse_split, split_mask, split_report)

DEFAULT_PARQUET = ROOT / "bms_ml" / "output" / "phase4" / "dataset" / "cross_section.parquet"
GATE = "player_chart_bias"
DIM_GRID = (2, 4, 8, 16, 32, 64)
REG_GRID = (0.3, 1.0, 3.0, 10.0, 30.0)
LOO_KS = (1, 5, 10)
INNER_VAL_FRAC = 0.1
MIN_TEST_ROWS = 10


def gate_verdict(mf: dict, gate: dict) -> dict:
    """Compare one seed's MF result against the gate on the protocol's own terms."""
    checks = {}
    for metric in ("mae", "player_centered_mae", "spearman", "player_centered_spearman",
                   "player_rank_spearman"):
        m, g = mf.get(metric), gate.get(metric)
        if m is None or g is None:
            continue
        # lower is better for the MAE family, higher for the rank family
        better = (m < g) if "mae" in metric else (m > g)
        checks[metric] = {"model": m, "gate": g, "delta": round(m - g, 4),
                          "better_than_gate": bool(better)}
    nd = mf.get("player_topk_ndcg", {})
    gd = gate.get("player_topk_ndcg", {})
    for k in nd:
        g = gd.get(k)
        if g is None or nd[k] is None:
            continue
        checks[f"player_topk_{k}"] = {"model": nd[k], "gate": g,
                                      "delta": round(nd[k] - g, 4),
                                      "better_than_gate": bool(nd[k] > g)}
    return checks


def select_hyperparams(tr: pd.DataFrame, target: str, val_idx: np.ndarray,
                       dims, regs, verbose: bool) -> tuple[dict, list[dict]]:
    """Pick (dim, reg) by inner-validation MAE.

    MAE, not RMSE: MAE is the headline metric, and RMSE on these sparse matrices
    is dominated by a handful of large residuals, so it is a needlessly noisy
    criterion. Returns (best_cfg, trace) where the trace records every grid point
    so the selection can be audited rather than trusted.
    """
    trace = []
    best = (np.inf, None)
    for dim in dims:
        for reg in regs:
            m = MFModel(dim=dim, reg=reg, seed=0, n_epochs=25)
            m.fit(tr, target, val_idx=val_idx)
            pred = predict_inner(m, tr, val_idx, target)
            v = float(np.mean(np.abs(pred[0] - pred[1])))
            trace.append({"dim": dim, "reg": reg, "inner_val_mae": round(v, 5),
                          "inner_val_rmse": round(float(np.sqrt(np.mean(
                              (pred[0] - pred[1]) ** 2))), 5),
                          "best_epoch": m.best_epoch, "diverged": m.diverged})
            if verbose:
                print(f"      dim={dim:<3} reg={reg:<5} val_mae={v:.4f} "
                      f"best_ep={m.best_epoch}" + ("  DIVERGED" if m.diverged else ""))
            if np.isfinite(v) and v < best[0]:
                best = (v, {"dim": dim, "reg": reg})
    return (best[1] or {"dim": DIM_GRID[0], "reg": REG_GRID[0]}), trace


def predict_inner(m: MFModel, tr: pd.DataFrame, val_idx: np.ndarray, target: str):
    """(actual, predicted) on the inner validation rows, using the fitted model."""
    val = tr.iloc[val_idx]
    p = m.players.get_indexer(val["player"])
    c = m.charts.get_indexer(val["sha256"])
    ok = (p >= 0) & (c >= 0)
    p, c = p[ok], c[ok]
    pred = m.mu + m.bu[p] + m.bc[c] + np.einsum("ij,ij->i", m.P[p], m.Q[c])
    return val[target].to_numpy(float)[ok], np.clip(pred, 0.0, 100.0)


def full_grid_on_test(tr: pd.DataFrame, te: pd.DataFrame, target: str, means: dict,
                      val_idx: np.ndarray, dims, regs) -> dict:
    """DIAGNOSTIC: score every grid point on the test split, at ONE seed.

    This exists only to answer "is a failure a modelling failure or a selection
    failure?" - it is explicitly an oracle over the test split and MUST NOT be
    read as a protocol result. The reported M3 number is always the inner-
    validation-selected configuration; this table just bounds what tuning could
    have achieved. It also records, per grid point, the inner-validation MAE so
    the selection gap (inner vs outer ranking) is visible.
    """
    rows = []
    for dim in dims:
        for reg in regs:
            m = MFModel(dim=dim, reg=reg, seed=0, n_epochs=25)
            m.fit(tr, target, val_idx=val_idx)
            ay, ap = predict_inner(m, tr, val_idx, target)
            r = evaluate_predictions(te, m.predict(te), target, means)
            rows.append({"dim": dim, "reg": reg,
                         "inner_val_mae": round(float(np.mean(np.abs(ay - ap))), 4),
                         "test_mae": r["mae"],
                         "test_player_centered_mae": r["player_centered_mae"]})
    best_sel = min(rows, key=lambda r: r["inner_val_mae"])
    best_test = min(rows, key=lambda r: r["test_mae"])
    return {
        "warning": "ORACLE OVER THE TEST SPLIT - diagnostic only, never a result",
        "n_grid_points": len(rows),
        "rows": rows,
        "best_by_inner_validation": best_sel,
        "best_by_test_mae": best_test,
        "selection_gap_mae": round(best_sel["test_mae"] - best_test["test_mae"], 4),
    }


def run_split(cdf: pd.DataFrame, split: str, seeds: list[int], target: str,
              frac: float, dims, regs, verbose: bool,
              grid_diagnostic: bool = False) -> dict:
    base_kind, loo_k = parse_split(split)
    block = {"split": split, "kind": base_kind, "loo_k": loo_k, "per_seed": {},
             "hyperparam_selection": {}}

    frozen: dict | None = None
    for seed in seeds:
        test = split_mask(cdf, split, seed, frac)
        if test.sum() < MIN_TEST_ROWS or (~test).sum() < MIN_TEST_ROWS:
            block["per_seed"][str(seed)] = {
                "skipped": "train or test too small",
                "n_test": int(test.sum()), "n_train": int((~test).sum())}
            continue
        tr = cdf[~test].reset_index(drop=True)
        te = cdf[test].reset_index(drop=True)
        means = player_train_means(tr, target)
        val_idx = inner_validation_indices(tr, split, seed, INNER_VAL_FRAC)

        # Hyper-parameters are chosen ONCE per split on the first usable seed, using
        # only that seed's TRAIN rows, then frozen. Re-selecting per seed would make
        # the seeds incomparable (different models, not different splits).
        if frozen is None:
            choice, trace = select_hyperparams(tr, target, val_idx, dims, regs, verbose)
            frozen = choice
            block["hyperparam_selection"] = {
                "chosen": choice, "grid_dims": list(dims), "grid_regs": list(regs),
                "inner_val_frac": INNER_VAL_FRAC, "trace": trace,
                "note": "chosen on train-only inner validation on the first usable seed, "
                        "then frozen for all seeds of this split"}
            if grid_diagnostic:
                block["grid_on_this_seed"] = full_grid_on_test(
                    tr, te, target, means, val_idx, dims, regs)

        res = {}
        gate = BaselineModel(GATE).fit(tr, target)
        res[GATE] = evaluate_predictions(te, gate.predict(te), target, means,
                                         want_ranking=True, with_calibration=True)
        shr = BaselineModel("player_chart_bias_shrunk", shrink_k=5.0).fit(tr, target)
        # Ranking metrics on the shrunk baseline too: it is the natural "is the gain
        # just regularised biases?" comparator, and without its Spearman we cannot
        # tell an MAE-only improvement from a ranking improvement.
        res["player_chart_bias_shrunk"] = evaluate_predictions(
            te, shr.predict(te), target, means, want_ranking=True)
        pl = BaselineModel("player_mean").fit(tr, target)
        res["player_mean"] = evaluate_predictions(te, pl.predict(te), target, means)
        ch = BaselineModel("chart_mean").fit(tr, target)
        res["chart_mean"] = evaluate_predictions(te, ch.predict(te), target, means)

        mf = MFModel(dim=frozen["dim"], reg=frozen["reg"], seed=seed, n_epochs=25)
        mf.fit(tr, target, val_idx=val_idx)
        res["mf_als"] = evaluate_predictions(te, mf.predict(te), target, means,
                                             want_ranking=True, with_calibration=True)
        res["mf_als"]["config"] = mf.config()
        res["mf_als"]["clipped_high"] = mf.clip_hi
        res["mf_als"]["clipped_low"] = mf.clip_lo

        # DIAGNOSTIC, not a competing model: the latent block is switched off
        # (dim=1 with enormous ridge), leaving only mu + b_player + b_chart fitted
        # with the same regularised solver. This splits the total gain over the gate
        # into "the gate's own group means were noisy, regularising them helped" and
        # "the latent interaction actually added something". Without it, a gain can
        # be over-claimed as evidence for matrix completion when it is only shrinkage.
        diag = MFModel(dim=1, reg=1e6, seed=0, n_epochs=5)
        diag.fit(tr, target, val_idx=val_idx)
        res["mf_bias_only_regularized"] = evaluate_predictions(
            te, diag.predict(te), target, means)

        # Content baseline runs on every split (cheap) so the cold-chart answer is
        # measured the same way as everything else.
        cr = ContentRidgeModel(alpha=1.0).fit(tr, target)
        res["content_ridge"] = evaluate_predictions(te, cr.predict(te), target, means)
        res["content_ridge"]["config"] = cr.config()

        res["verdict_vs_gate"] = gate_verdict(res["mf_als"], res[GATE])
        block["per_seed"][str(seed)] = {
            "split": split_report(cdf, test, split), "models": res}

    blocks = [s for s in block["per_seed"].values() if "models" in s]
    if blocks:
        block["across_seeds"] = {}
        for name in (GATE, "player_chart_bias_shrunk", "mf_als", "mf_bias_only_regularized",
                     "content_ridge", "player_mean", "chart_mean"):
            for metric in ("mae", "rmse", "player_centered_mae", "spearman",
                           "player_centered_spearman", "player_rank_spearman"):
                vals = [b["models"][name].get(metric) for b in blocks
                        if b["models"][name].get(metric) is not None]
                if not vals:
                    continue
                block["across_seeds"].setdefault(name, {})[metric] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
                    "values": [round(float(v), 4) for v in vals]}
    return block


def trial_verdict(results: dict) -> dict:
    """Apply the §6 gate to the collected numbers; a fail here stops the phase."""
    out = {"splits_checked": {}, "passed": True, "reasons": []}
    for split, block in results.items():
        if "across_seeds" not in block:
            continue
        a = block["across_seeds"]
        if "mf_als" not in a or GATE not in a:
            continue
        checks = {}
        for metric, mode in (("mae", "lower"), ("player_centered_mae", "lower"),
                             ("player_rank_spearman", "higher")):
            m = a["mf_als"].get(metric)
            g = a[GATE].get(metric)
            if not m or not g:
                continue
            gain = (g["mean"] - m["mean"]) if mode == "lower" else (m["mean"] - g["mean"])
            # "stable" = every seed beats the gate baseline mean, and the gain is
            # larger than the model's own seed spread
            wins = [(g["mean"] - v) if mode == "lower" else (v - g["mean"])
                    for v in m["values"]]
            checks[metric] = {
                "model_mean": m["mean"], "gate_mean": g["mean"],
                "gain": round(gain, 4), "model_seed_std": m["std"],
                "all_seeds_beat_gate_mean": bool(all(w > 0 for w in wins)),
                "gain_exceeds_seed_std": bool(gain > m["std"]),
                "stable_win": bool(all(w > 0 for w in wins) and gain > m["std"])}
        out["splits_checked"][split] = checks

    # Gate decision uses the warm splits only (cold_chart has no matrix signal by
    # construction; cold_player is gated separately by "not worse than chart_mean").
    warm = {s: c for s, c in out["splits_checked"].items()
            if s == "random_interaction" or s.startswith("loo_")}
    # loo_1 is excluded from the player-internal part of the gate (see below) but its
    # overall MAE is still checked.
    for split, checks in warm.items():
        mae = checks.get("mae", {})
        cen = checks.get("player_centered_mae", {})
        rk = checks.get("player_rank_spearman", {})
        if not mae.get("stable_win"):
            out["passed"] = False
            out["reasons"].append(f"{split}: overall MAE did not beat the gate stably")
        if not cen.get("stable_win"):
            out["passed"] = False
            out["reasons"].append(f"{split}: player_centered_mae did not beat the gate stably")
        if rk and not rk.get("stable_win"):
            out["passed"] = False
            out["reasons"].append(f"{split}: player_rank_spearman did not beat the gate stably")
        # loo_1 holds out ONE row per player. After per-player centring that leaves a
        # single point per player, so the centred metrics are identically 0 and carry
        # no information. Flagged so a "0.000" is never read as a perfect score.
        if split == "loo_1_charts_per_player":
            out["reasons"].append(
                f"{split}: DEGENERATE for player-internal metrics (one test row per "
                "player); only the overall MAE is meaningful on this split")

    # Secondary, harder reading: did the LATENT term add anything, or is the whole
    # gain just the gate's noisy group means being regularised? Both are improvements
    # over the gate, but only the former is evidence for matrix completion.
    out["latent_vs_regularized_bias"] = {}
    for split, block in results.items():
        a = block.get("across_seeds")
        if not a or "mf_bias_only_regularized" not in a:
            continue
        row = {}
        for metric in ("mae", "player_centered_mae", "player_rank_spearman"):
            m = a["mf_als"].get(metric)
            b = a["mf_bias_only_regularized"].get(metric)
            if not m or not b:
                continue
            gain = (b["mean"] - m["mean"]) if "mae" in metric else (m["mean"] - b["mean"])
            row[metric] = {"mf": m["mean"], "regularized_bias": b["mean"],
                           "latent_gain": round(gain, 4),
                           "exceeds_seed_std": bool(gain > m["std"])}
        out["latent_vs_regularized_bias"][split] = row

    cold = out["splits_checked"].get("cold_player", {})
    if cold:
        m = cold.get("mae", {})
        out["cold_player_check"] = {
            "mf_mae": m.get("model_mean"), "chart_mean_mae": m.get("gate_mean"),
            "note": ("cold_player is gated separately: for a player with no train rows the "
                     "MF player effect and latent term are absent, so the model must not be "
                     "worse than the chart-mean baseline. The gate baseline is degenerate "
                     "here (it equals chart_mean), so this check reads the same numbers."),
            "within_gate": bool(m.get("model_mean") is not None
                                and m.get("model_mean") <= m.get("gate_mean", np.inf) + 1e-9),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 M3: biased matrix factorisation")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--target", default="acc", choices=["acc", "acc_alt"])
    ap.add_argument("--clients", default=",".join(CLIENTS))
    ap.add_argument("--splits", default=",".join(all_split_names(LOO_KS)))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--dims", default=",".join(map(str, DIM_GRID)))
    ap.add_argument("--regs", default=",".join(map(str, REG_GRID)))
    ap.add_argument("--out", default="m3_matrix_completion.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--grid-diagnostic", action="store_true",
                    help="also score the whole grid on the test split (ORACLE, "
                         "diagnostic only - see full_grid_on_test)")
    args = ap.parse_args()

    clients = [c.strip() for c in args.clients.split(",") if c.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    dims = tuple(int(d) for d in args.dims.split(","))
    regs = tuple(float(r) for r in args.regs.split(","))

    df, filt = load_frame(args.parquet, args.target, 120)
    payload = {"data_filter": filt, "gate_baseline": GATE, "splits": splits, "seeds": seeds,
               "dim_grid": list(dims), "reg_grid": list(regs), "inner_val_frac": INNER_VAL_FRAC,
               "model_definition": "mu + b_player + b_chart + <p_player, q_chart> (ALS)",
               "hyperparam_rule": "dim/reg chosen on train-only inner validation, frozen per split",
               "clients": {}}

    for client in clients:
        cdf = df[df["client"] == client].reset_index(drop=True)
        if not len(cdf):
            continue
        cblock = {"rows": int(len(cdf)), "players": int(cdf["player"].nunique()),
                  "charts": int(cdf["sha256"].nunique()), "per_split": {}}
        if cblock["players"] < 2:
            cblock["identifiability"] = (
                "fewer than 2 players: there is no player factor to learn and matrix "
                "completion cannot be distinguished from a per-chart mean.")
        if cblock["players"] < 5:
            cblock["identifiability_note"] = (
                f"only {cblock['players']} players. An ALS latent space has at most "
                f"{cblock['players']} identifiable directions, so any 'win' here is a weak "
                "result and must not be reported as evidence for collaborative filtering.")
        for split in splits:
            print(f"[{client}] {split} ...", flush=True)
            cblock["per_split"][split] = run_split(
                cdf, split, seeds, args.target, args.frac, dims, regs, args.verbose,
                grid_diagnostic=args.grid_diagnostic)
        cblock["gate_verdict"] = trial_verdict(cblock["per_split"])
        payload["clients"][client] = cblock

    data_config = {"parquet": str(args.parquet.relative_to(ROOT)), "target": args.target,
                   "row_filter": filt, "inner_val_frac": INNER_VAL_FRAC,
                   "no_time_no_first_play_no_level_feature": True}
    out = write_result(args.out,
                       header(script="run_m3.py", data_config=data_config,
                              client=",".join(clients), split=",".join(splits),
                              model="mf_als vs " + GATE, seed=seeds),
                       payload)
    txt = [f"\nwrote {out.relative_to(ROOT)}"]
    for client, cb in payload["clients"].items():
        txt.append(f"== {client}: players={cb['players']} rows={cb['rows']}")
        for split, block in cb["per_split"].items():
            a = block.get("across_seeds")
            if not a:
                continue
            mf, gt = a.get("mf_als", {}).get("mae"), a.get(GATE, {}).get("mae")
            cm, cg = a.get("mf_als", {}).get("player_centered_mae"), a.get(GATE, {}).get("player_centered_mae")
            rs, rg = a.get("mf_als", {}).get("player_rank_spearman"), a.get(GATE, {}).get("player_rank_spearman")
            txt.append(
                f"   {split:<26} mae {mf['mean']:.3f}±{mf['std']:.3f} vs {gt['mean']:.3f}"
                f" | cmae {cm['mean']:.3f} vs {cg['mean']:.3f}"
                f" | rank_sp {(rs or {}).get('mean')} vs {(rg or {}).get('mean')}")
        v = cb["gate_verdict"]
        txt.append(f"   GATE §6: {'PASS' if v['passed'] else 'FAIL'} {v['reasons']}")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
