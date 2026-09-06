"""Phase 3.5 (task 6): evaluate the v2 threshold-free chart statistics.

Compares, on the SAME rows / split / protocol (PROTOCOL.md):
  A          chart-only, v1 27-dim                          (current lower reference)
  A_v2       chart-only, v1 + v2 (27 + 12)
  B          chart + history, v1                            (current best)
  B_v2       chart + history, v1 + v2                       (the candidate)
  B_no_cj    B with chord_count/chord2/chord3plus/jack_count removed
             — the redundancy test: do the community-conventional pattern
             categories carry signal beyond density? (user question 2026-09-07)
  B_v2only   chart(v2 12-dim only) + history                 — is v2 self-sufficient?

All models are deterministic HGB (fixed random_state); lamp via ordinal
regression, BP via log1p — same conventions as compare_nolevel.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from chart_repr import (HISTORY_FEATURES, OBJECTIVE_STAT_COLS,
                        OBJECTIVE_V2_COLS, feature_manifest)
from common import centered_r2, hgb_fit_predict, load_samples, mae, r2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"

# community-conventional pattern categories under test
PATTERN_COLS = ["c_chord_count", "c_chord2_count", "c_chord3plus_count", "c_jack_count"]


def main() -> None:
    df = load_samples()
    v2 = pd.read_parquet(DS / "chart_stats_v2.parquet")
    df = df.merge(v2, on="sha256", how="left")
    assert df["v2_ioi_lane_p05"].notna().mean() > 0.99, "v2 join failed"
    tr, te = df[df["phase"] == "train"], df[df["phase"] == "test"]

    sets = {
        "A": OBJECTIVE_STAT_COLS,
        "A_v2": OBJECTIVE_STAT_COLS + OBJECTIVE_V2_COLS,
        "B": OBJECTIVE_STAT_COLS + HISTORY_FEATURES,
        "B_v2": OBJECTIVE_STAT_COLS + OBJECTIVE_V2_COLS + HISTORY_FEATURES,
        "B_no_cj": [c for c in OBJECTIVE_STAT_COLS if c not in PATTERN_COLS]
                   + HISTORY_FEATURES,
        "B_no_peak": [c for c in OBJECTIVE_STAT_COLS if c != "c_peak_nps_1s"]
                     + HISTORY_FEATURES,
        "B_v2only": OBJECTIVE_V2_COLS + HISTORY_FEATURES,
    }

    results: dict = {"n_train": len(tr), "n_test": len(te),
                     "v2_missing_frac": round(float(df["v2_ioi_lane_p05"].isna().mean()), 4)}

    for tag, feats in sets.items():
        acc_p = hgb_fit_predict(tr, te, feats, tr["acc"].values)
        lamp_p = hgb_fit_predict(tr, te, feats, tr["lamp"].values.astype(float))
        bp_p = np.expm1(hgb_fit_predict(tr, te, feats, np.log1p(tr["bp"].values)))
        results[tag] = {
            "acc": {"mae": round(mae(te["acc"], acc_p), 3),
                    "r2": round(r2(te["acc"], acc_p), 3),
                    "centered_r2": round(centered_r2(te, acc_p, "acc"), 3),
                    "per_player": {p: round(mae(g["acc"], acc_p[te["player"].values == p]), 3)
                                   for p, g in te.groupby("player")}},
            "lamp": {"ord_mae": round(mae(te["lamp"], lamp_p), 3),
                     "qwk": round(float(cohen_kappa_score(
                         te["lamp"], np.clip(np.round(lamp_p), 1, 9).astype(int),
                         weights="quadratic", labels=list(range(1, 10)))), 3)},
            "bp": {"raw_mae": round(mae(te["bp"], bp_p), 2)},
        }
        print(f"{tag:9} acc {results[tag]['acc']['mae']:7.3f}  "
              f"cR2 {results[tag]['acc']['centered_r2']:+.3f}  "
              f"lamp {results[tag]['lamp']['ord_mae']:.3f}  "
              f"QWK {results[tag]['lamp']['qwk']:.3f}  "
              f"BP {results[tag]['bp']['raw_mae']:.1f}")

    results["features_used"] = feature_manifest(*sets.values())
    json.dump(results, open(OUT / "chart_v2_eval.json", "w"), indent=2)
    print("saved ->", OUT / "chart_v2_eval.json")


if __name__ == "__main__":
    main()
