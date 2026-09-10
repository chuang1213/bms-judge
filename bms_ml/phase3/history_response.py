"""Phase 3.6: per-axis PERSONAL RESPONSE PROFILE (history side) - protocol variant.

The estimator and its rationale live in `response_features.py` (shared with
transfer_eval.py, which uses a windowed + truncation-augmented variant for few-shot).
This script is the FULL-HISTORY protocol variant: every row is described by an OLS
fitted on its owner's entire strictly-prior archive.

Why this exists: the 12 hand-crafted history features contain exactly ONE
chart-conditioned player feature, `h_knn_acc` (a kNN mean in the 27-dim stat space).
It cannot express direction - "this player collapses on density but is fine on
scratch" - and it needs a lot of history to stabilise. A one-parameter-per-axis
response curve is a far cheaper estimator of the same idea, and PHASE3_5_REVIEW §4.3
named it as the highest-value upgrade to the player side.

Result (2026-09-11, EXPERIMENT_LOG): B 7.009 -> B_full 6.583 acc, cR2 +0.386 -> +0.430,
17/18 players improve, and a same-width shuffled control (B+shuffled24 = 7.001) shows
the gain is information rather than extra HGB capacity.

Output: bms_ml/output/phase3/dataset/history_response.parquet, keyed (player, sha256).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from chart_repr import (HISTORY_RESPONSE_BP_COLS, HISTORY_RESPONSE_COLS,
                        HISTORY_RESPONSE_LAMP_COLS, RESPONSE_AXES)
from response_features import axis_sd, build_table

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "bms_ml" / "output" / "phase3" / "dataset"
MIN_HISTORY = 20      # below this an OLS slope is noise; feature stays NaN (imputed)


def main() -> None:
    fp = pd.read_parquet(DS / "firstplays.parquet")
    # four axes are v2 columns stored in a side table keyed 1:1 on sha256
    v2 = pd.read_parquet(DS / "chart_stats_v2.parquet")
    need = [c for c in set(RESPONSE_AXES.values()) if c not in fp.columns]
    if need:
        fp = fp.merge(v2[["sha256"] + [c for c in need if c in v2.columns]],
                      on="sha256", how="left")
    # chronological within player is required for the causal prefix sums
    fp = fp.sort_values(["player", "time"], kind="stable").reset_index(drop=True)

    sd = axis_sd(fp)
    fp["log1p_bp"] = np.log1p(fp["bp"].values.astype(np.float64))   # project-wide BP form
    acc_blk = build_table(fp, sd, min_n=MIN_HISTORY, window=None, rng=None,
                          target="acc", prefix="h_", clip=(0.0, 100.0))
    lamp_blk = build_table(fp, sd, min_n=MIN_HISTORY, window=None, rng=None,
                           target="lamp", prefix="l_", clip=(1.0, 9.0))
    bp_blk = build_table(fp, sd, min_n=MIN_HISTORY, window=None, rng=None,
                         target="log1p_bp", prefix="b_", clip=(0.0, 12.0))
    feats = pd.concat([acc_blk, lamp_blk, bp_blk], axis=1)
    out = pd.concat([fp[["player", "sha256", "phase", "time"]], feats], axis=1)
    out.to_parquet(DS / "history_response.parquet")
    for blk, cols in (("acc", HISTORY_RESPONSE_COLS),
                      ("lamp", HISTORY_RESPONSE_LAMP_COLS),
                      ("bp", HISTORY_RESPONSE_BP_COLS)):
        # cols[-2] is <prefix>resp_mean
        print(f"{blk}: coverage {float(feats[cols[-2]].notna().mean()):.3f}")
    print(f"rows {len(out)} -> {DS / 'history_response.parquet'}")
    print(feats[HISTORY_RESPONSE_COLS].describe().loc[["mean", "std", "min", "max"]]
          .round(3).T.to_string())
    print(feats[HISTORY_RESPONSE_LAMP_COLS].describe().loc[["mean", "std", "min", "max"]]
          .round(3).T.to_string())


if __name__ == "__main__":
    main()
