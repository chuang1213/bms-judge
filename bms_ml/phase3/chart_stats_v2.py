"""Phase 3.5 (task 6): threshold-free objective chart statistics, v2.

Why v2 exists
-------------
The v1 chart features include community-conventional pattern categories:
`jack_count` (same-lane repetition below SOME threshold) and `chord2/3plus_count`
(instantaneous note clusters). These are (a) threshold/convention dependent and
(b) largely projections of density (a chord is density at an instant; a jack is
density concentrated on one lane). User observation 2026-09-07.

v2 therefore uses NO pattern vocabulary at all. Everything is derived from the
raw onset data (time x lane) as DISTRIBUTION STATISTICS, which are
threshold-free by construction:

  per-lane inter-onset intervals -> percentiles   (jack-ness without naming it)
  per-second onset counts        -> std / p90     (density variability, no bin rule beyond 1s)
  notes per exact timestamp      -> max / std     (simultaneity shape; same grid
                                                   position is file-objective)
  lane distribution              -> entropy, hand balance, scratch rate

All lanes follow the parser/manifest convention: lane 0 = scratch, 1-7 = keys
(verified against manifest lane0_scratch on real sequences; NOTE grid_data.py's
docstring claims scratch=7, which is wrong — see PHASE3_5_REVIEW.md).

Output: bms_ml/output/phase3/dataset/chart_stats_v2.parquet (sha256, v2_*).
Sequences missing for ~0.2% of charts -> NaN -> train-median imputed downstream.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "bms_ml" / "output" / "phase3" / "dataset"
SEQ_ROOT = ROOT / "bms_ml" / "output" / "corpus" / "sequences"
SCRATCH_LANE = 0          # parser convention (keys = 1..7)
KEY_LANES = list(range(1, 8))

# the column list lives in the registry (single source of truth)
from chart_repr import OBJECTIVE_V2_COLS as V2_COLS  # noqa: E402


def chart_v2_stats(seq: np.ndarray) -> dict:
    """seq: N x 4 [time_sec, lane, type, duration]. Returns the V2_COLS dict."""
    out = {c: np.nan for c in V2_COLS}
    if len(seq) < 2:
        return out
    t = np.sort(seq[:, 0])
    lanes = seq[:, 1].astype(int)
    dur = float(t[-1] - t[0])
    if dur <= 0:
        return out

    # ---- per-lane inter-onset intervals (pooled across lanes) ----
    iois = []
    for ln in np.unique(lanes):
        tl = np.sort(t[lanes == ln])
        if len(tl) > 1:
            iois.append(np.diff(tl))
    if iois:
        pooled = np.concatenate(iois)
        pooled = pooled[pooled > 1e-9]          # identical-position duplicates
        if len(pooled):
            out["v2_ioi_lane_p05"] = float(np.percentile(pooled, 5))
            out["v2_ioi_lane_p25"] = float(np.percentile(pooled, 25))
            out["v2_ioi_lane_mean"] = float(pooled.mean())

    # ---- ALL-onset IOI (window-free density extremes; replaces the arbitrary
    # 1-second window of v1 peak_nps_1s) ----
    tg = np.diff(t)
    tg = tg[tg > 1e-9]
    if len(tg):
        out["v2_ioi_global_p05"] = float(np.percentile(tg, 5))
        out["v2_ioi_global_p25"] = float(np.percentile(tg, 25))

    # ---- scratch-lane IOI ----
    ts = np.sort(t[lanes == SCRATCH_LANE])
    if len(ts) > 1:
        d = np.diff(ts)
        d = d[d > 1e-9]
        if len(d):
            out["v2_ioi_scratch_p05"] = float(np.percentile(d, 5))

    # ---- per-second onset counts ----
    n_bins = int(np.floor(dur)) + 1
    if n_bins >= 3:
        counts, _ = np.histogram(t, bins=np.arange(n_bins + 1))
        out["v2_nps_std"] = float(counts.std())
        out["v2_nps_p90"] = float(np.percentile(counts, 90))

    # ---- simultaneity: notes per exact timestamp (grid position is file-objective) ----
    _, per_t = np.unique(t, return_counts=True)
    out["v2_simul_max"] = float(per_t.max())
    out["v2_simul_std"] = float(per_t.std())

    # ---- lane distribution ----
    cnt = np.bincount(lanes, minlength=8).astype(float)
    p = cnt[cnt > 0] / cnt.sum()
    out["v2_lane_entropy"] = float(-(p * np.log(p)).sum())
    left, right = cnt[1:4].sum(), cnt[5:8].sum()   # keys 1-3 vs 5-7, centre (4) excluded
    if left + right > 0:
        out["v2_hand_balance"] = float(abs(left - right) / (left + right))
    out["v2_scratch_nps"] = float(cnt[SCRATCH_LANE] / dur)

    # ---- LN structure ----
    ln_mask = seq[:, 2] == 1
    if ln_mask.any():
        durs = seq[ln_mask, 3]
        durs = durs[durs > 0]
        if len(durs):
            out["v2_ln_mean_dur"] = float(durs.mean())
    return out


def main() -> None:
    fp = pd.read_parquet(DS / "firstplays.parquet")
    shas = sorted(fp["sha256"].unique())
    rows, missing = [], 0
    for sha in shas:
        p = SEQ_ROOT / f"{sha}.npy"
        if not p.exists():
            missing += 1
            rows.append({"sha256": sha, **{c: np.nan for c in V2_COLS}})
            continue
        rows.append({"sha256": sha, **chart_v2_stats(np.load(p))})
    out = pd.DataFrame(rows)
    out.to_parquet(DS / "chart_stats_v2.parquet")
    print(f"charts {len(out)}, missing sequences {missing} -> {DS / 'chart_stats_v2.parquet'}")
    have = out.dropna(subset=["v2_ioi_lane_p05"])
    print(have[V2_COLS].describe().loc[["mean", "50%", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
