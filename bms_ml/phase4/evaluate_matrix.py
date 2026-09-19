"""Phase 4 step 5 — held-out evaluation protocol + fixed baselines.

This file is the BASELINE contract, not a model. It implements exactly the splits
and the fixed baselines the project fixed before seeing any Phase 4 result, and it
writes every number with its provenance (client, split, seed, git commit).

The split definitions and the metrics live in `splits.py` / `metrics.py` so that
`run_m3.py` cannot score a model under a quietly different protocol. This module
owns the row filter, the fixed baselines, and the result JSON.

SPLITS (per client; LR2 and beatoraja are never pooled) - see `splits.py`
  random_interaction      mask a fraction of observed cells at random
  loo_k_charts_per_player hold out k observed charts per player (k=1/5/10)
  cold_player             all cells of a held-out set of players are unseen
  cold_chart              all cells of a held-out set of charts are unseen

BASELINES (all fitted on TRAIN rows only)
  global_mean         : mu
  player_mean         : mu + player effect (unseen player -> mu)
  chart_mean          : mu + chart effect (unseen chart -> mu)
  player_chart_bias   : mu + player effect + chart effect   <- the M3 gate
  player_chart_bias_shrunk : the same decomposition with empirical-Bayes shrinkage
                        (a model, not a gate)
  content_ridge       : chart content (`cs_*`) + player offset, for cold_chart only

METRICS: MAE / RMSE (primary); player-internal centred MAE / rank Spearman / top-k
NDCG; calibration by prediction decile. See `metrics.py`.

Usage
-----
  python bms_ml/phase4/evaluate_matrix.py                       # all splits, both clients
  python bms_ml/phase4/evaluate_matrix.py --splits random_interaction --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import (TOP_K, evaluate_predictions,  # noqa: E402
                     player_train_means)
from models import BaselineModel, ContentRidgeModel  # noqa: E402
from provenance import CLIENTS, ROOT, header, write_result  # noqa: E402
from splits import all_split_names, split_mask, split_report  # noqa: E402

# Re-exported for backwards compatibility with tests/scripts written against the
# first Phase 4 session, which imported these from this module. The definitions
# live in splits.py / metrics.py.
__all__ = ["load_frame", "evaluate_one", "split_mask", "split_report",
           "evaluate_predictions", "player_train_means", "BaselineModel",
           "ContentRidgeModel", "model_by_name", "BASELINES", "GATE_BASELINE",
           "CONTENT_MODELS", "DEFAULT_SPLITS", "MIN_JUDGEMENTS", "SHRINK_K"]

DEFAULT_PARQUET = ROOT / "bms_ml" / "output" / "phase4" / "dataset" / "cross_section.parquet"
DEFAULT_SPLITS = all_split_names((1, 5, 10))
BASELINES = ("global_mean", "player_mean", "chart_mean", "player_chart_bias",
             "player_chart_bias_shrunk")
GATE_BASELINE = "player_chart_bias"
CONTENT_MODELS = ("content_ridge",)

# Row filter. Every choice is reported with its cost in `data_filter`.
MIN_JUDGEMENTS = 120
SHRINK_K = 5.0        # empirical-Bayes weight for the shrunk variant
MIN_TEST_ROWS = 10


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_frame(parquet: Path, target: str, min_judgements: int) -> tuple[pd.DataFrame, dict]:
    """Apply the Phase 4 row filter and record what each step costs."""
    df = pd.read_parquet(parquet)
    steps = {"input_rows": len(df)}

    def step(name: str, mask: pd.Series) -> None:
        nonlocal df
        steps[name] = int(len(df) - int(mask.sum()))
        df = df[mask]

    step("drop_mode_kind_not_sp", df["mode_kind"].eq("sp"))
    step("drop_below_min_judgements", df["n_judgements"].fillna(0) >= min_judgements)
    step("drop_target_nan", df[target].notna())
    step("drop_notes_nan_or_zero", df["notes"].fillna(0) > 0)
    step("drop_playcount_zero_or_nan", df["playcount"].fillna(0) > 0)
    steps["kept_rows"] = len(df)
    steps["min_judgements_threshold"] = min_judgements
    steps["target"] = target
    steps["why_playcount_filter"] = (
        "Phase 4 predicts a chart the player has never played; rows with "
        "playcount==0 are not real observations and would leak a no-play row "
        "back into the target set.")
    df = df.reset_index(drop=True)
    return df, steps


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def model_by_name(name: str):
    if name == "player_chart_bias_shrunk":
        return BaselineModel(name, shrink_k=SHRINK_K)
    if name == "content_ridge":
        return ContentRidgeModel(alpha=1.0)
    return BaselineModel(name)


def evaluate_one(tr: pd.DataFrame, te: pd.DataFrame, target: str,
                 models=BASELINES, content_models=CONTENT_MODELS) -> dict:
    """Fit every baseline on `tr` only, then score it on `te`."""
    means = player_train_means(tr, target)
    out = {}
    for name in models:
        m = model_by_name(name).fit(tr, target)
        out[name] = evaluate_predictions(
            te, m.predict(te), target, means,
            want_ranking=(name == GATE_BASELINE),
            with_calibration=(name == GATE_BASELINE))
    for name in content_models:
        m = model_by_name(name).fit(tr, target)
        r = evaluate_predictions(te, m.predict(te), target, means)
        r["config"] = m.config()
        out[name] = r
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 step 5: splits + fixed baselines")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    ap.add_argument("--target", default="acc", choices=["acc", "acc_alt"])
    ap.add_argument("--min-judgements", type=int, default=MIN_JUDGEMENTS)
    ap.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    ap.add_argument("--clients", default=",".join(CLIENTS))
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--frac", type=float, default=0.2)
    ap.add_argument("--out", default=None, help="result JSON name (default: evaluate_matrix.json)")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    clients = [c.strip() for c in args.clients.split(",") if c.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    all_models = list(BASELINES) + list(CONTENT_MODELS)

    df, filt = load_frame(args.parquet, args.target, args.min_judgements)
    payload = {
        "data_filter": filt,
        "splits": splits, "seeds": seeds, "frac": args.frac,
        "baselines": list(BASELINES), "content_models": list(CONTENT_MODELS),
        "gate_baseline": GATE_BASELINE, "top_k": list(TOP_K), "shrink_k": SHRINK_K,
        "split_definitions": {
            "random_interaction": "mask a random frac of observed cells "
                                  "(all players/charts still seen in train)",
            "loo_k_charts_per_player": "hold out k observed charts per player",
            "cold_player": "hold out whole players (absent from train)",
            "cold_chart": "hold out whole charts (absent from train)",
        },
        "clients": {},
    }

    for client in clients:
        cdf = df[df["client"] == client].reset_index(drop=True)
        if not len(cdf):
            continue
        cblock = {"rows": int(len(cdf)), "players": int(cdf["player"].nunique()),
                  "charts": int(cdf["sha256"].nunique()), "per_split": {}}
        for kind in splits:
            sblock = {"per_seed": {}}
            for seed in seeds:
                test = split_mask(cdf, kind, seed, args.frac)
                if test.sum() < MIN_TEST_ROWS or (~test).sum() < MIN_TEST_ROWS:
                    sblock["per_seed"][str(seed)] = {
                        "skipped": "train or test too small",
                        "n_test": int(test.sum()), "n_train": int((~test).sum())}
                    continue
                tr, te = cdf[~test], cdf[test]
                sblock["per_seed"][str(seed)] = {
                    "split": split_report(cdf, test, kind),
                    "models": evaluate_one(tr, te, args.target),
                }
            if len(seeds) > 1:
                sblock["across_seeds"] = {}
                for m in all_models:
                    for metric in ("mae", "player_centered_mae", "player_rank_spearman"):
                        vals = [s["models"][m].get(metric) for s in sblock["per_seed"].values()
                                if "models" in s and s["models"][m].get(metric) is not None]
                        if len(vals) > 1:
                            sblock["across_seeds"].setdefault(m, {})[metric] = {
                                "mean": round(float(np.mean(vals)), 4),
                                "std": round(float(np.std(vals, ddof=1)), 4),
                                "values": [round(float(v), 4) for v in vals]}
            cblock["per_split"][kind] = sblock

        if cblock["players"] == 1:
            cblock["identifiability"] = (
                "single-player matrix: `player_mean` and `player_chart_bias` are "
                "algebraically identical to `global_mean`, `cold_player` is "
                "undefined, and the loo splits hold out only that player's rows. No "
                "matrix-completion claim can be made on this client.")
        elif cblock["players"] < 5:
            cblock["identifiability_note"] = (
                f"only {cblock['players']} players: a latent factor model has at most "
                f"{cblock['players']} identifiable directions, so any matrix-completion "
                "advantage here is weak evidence and must be reported as such.")
        cblock["degenerate_baselines"] = {}
        for kind in cblock["per_split"]:
            if kind == "cold_player":
                cblock["degenerate_baselines"][kind] = ["player_mean", "player_chart_bias"]
            elif kind == "cold_chart":
                cblock["degenerate_baselines"][kind] = ["chart_mean", "player_chart_bias"]
        payload["clients"][client] = cblock

    data_config = {
        "parquet": str(args.parquet.relative_to(ROOT)),
        "target": args.target,
        "min_judgements": args.min_judgements,
        "row_filter": filt,
        "note": "no play time, no first-play construction, no difficulty-table level "
                "as a feature; clients never pooled",
    }
    name = args.out or "evaluate_matrix.json"
    if args.target != "acc" and args.out is None:
        name = f"evaluate_matrix_{args.target}.json"
    out = write_result(name, header(script="evaluate_matrix.py", data_config=data_config,
                                    client=",".join(clients), split=",".join(splits),
                                    model=",".join(all_models), seed=seeds),
                       payload)

    for client, cb in payload["clients"].items():
        print(f"== {client}: rows={cb['rows']} players={cb['players']} charts={cb['charts']}")
        for kind, sb in cb["per_split"].items():
            for seed, s in sb["per_seed"].items():
                if "models" not in s:
                    print(f"   {kind:<24} seed={seed} SKIPPED {s}")
                    continue
                ms = s["models"]
                line = "  ".join(
                    f"{m.split('_')[0][:6]}={ms[m]['mae']:6.3f}"
                    for m in BASELINES if m in ms)
                print(f"   {kind:<24} seed={seed} n_te={s['split']['n_test']:>5}  {line}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
