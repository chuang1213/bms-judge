"""Phase 3.6: permutation-space chart geometry ("perm_space"), a new encoder.

Motivation
----------
`OBJECTIVE_STAT_COLS` (27) and `OBJECTIVE_V2_COLS` (14) both describe the
chart AS WRITTEN. But a BMS player almost never plays the written arrangement:
RANDOM/MIRROR options permute the 7 key lanes, so the *experience* is one draw
from a 5040-element family. Permikon (Permikon-main/, panchira_cli/src/evaluator.rs)
takes this seriously and evaluates all 5040 permutations with four
hand-travel metrics. Two things follow:

1. There is no single "objective" description — so describe the *space of
   descriptions* instead: summarise each metric over all 5040 permutations
   (mean / std / min / max) plus where the WRITTEN arrangement sits in that
   distribution (its percentile). This is a pure function of the note data, it
   needs no community vocabulary, and it is invariant to the player's random
   option by construction.
2. v2 already covers density/IOI *distribution shape* and treats lanes almost
   symmetrically (lane entropy, hand balance). Nothing so far describes lane
   TRAJECTORY — how far the hands must travel between consecutive positions.
   Permikon's smooth/base/tight do exactly that, and are orthogonal to v2.

Deliberate departures from Permikon (declared, not silent)
----------------------------------------------------------
* NO per-chart min-max normalisation. Permikon normalises each metric to [0,1]
  *within one chart* for its ranking UI; that destroys cross-chart comparability,
  which is fatal for a predictive feature. We keep absolute values and let the
  summary over permutations carry the signal.
* `spike` (weighted p90 of variability) is dropped: it needs a per-permutation
  sort (O(5040 * P log P)) and is largely a quantile view of `base`. Replaced by
  `spread` (weighted std of the average lane position), which is free and is a
  genuinely different quantity (total position variance vs regression residual).
* Permikon drops scratch and LN. We follow its 7-key permutation convention
  (scratch anchored, out of the permutation group) but keep two
  permutation-invariant scratch-interleaving features, since scratch is
  mechanically distinct and is NOT remapped the same way by players' muscle
  memory. LN starts count as onsets (same as Permikon).
* Lane index 1..7 is used as a 1-D spatial coordinate (4 = centre). This is a
  human-factor / physical-layout assumption, exactly like v2's hand_balance —
  it is NOT file structure, and is declared as such in the registry.

Cost
----
Per chart ~6 passes over a (5040, n_positions) float32 block, chunked over
permutations to cap memory. Charts are independent -> multiprocessing.
Average chart in scope has ~2.4k notes; 4.2k charts take ~2 min on 12 workers.

Output: bms_ml/output/phase3/dataset/chart_perm_space.parquet (sha256, ps_*).
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "bms_ml" / "output" / "phase3" / "dataset"
SEQ_ROOT = ROOT / "bms_ml" / "output" / "corpus" / "sequences"
SCRATCH_LANE = 0          # parser convention (keys = 1..7); see chart_stats_v2.py

from chart_repr import OBJECTIVE_PERM_COLS as PS_COLS  # noqa: E402  (registry = single source)

# ---------------------------------------------------------------- permutations
_N_PERM = 5040
_PERMS = np.array(list(itertools.permutations(range(1, 8))), dtype=np.uint8)  # (5040,7)
# row 0 is the identity (1,2,3,4,5,6,7) == the written arrangement; relied upon below.
assert tuple(_PERMS[0]) == (1, 2, 3, 4, 5, 6, 7)

_LANE_TO_KEY = np.argsort(_PERMS, axis=1) + 1          # (5040,7) lane -> key, by perm
_LANE_BITS = (1 << (_LANE_TO_KEY - 1)).astype(np.uint8)  # (5040,7) key bit, by lane
_MASKS = np.arange(128, dtype=np.uint8)

# MAP[pi, mask] = the key-mask that permutation pi turns `mask` into.
_MAP = np.zeros((_N_PERM, 128), dtype=np.uint8)
for _lane in range(1, 8):
    _has = (_MASKS & np.uint8(1 << (_lane - 1))) != 0
    _MAP[:, _has] |= _LANE_BITS[:, _lane - 1:_lane].astype(np.uint8)

# per-mask geometry (over key lanes only; scratch is anchored, see module docstring)
_CNT = np.array([bin(m).count("1") for m in range(128)], dtype=np.float32)
_LANE_SUM = np.zeros(128, dtype=np.float32)
_MIN_L, _MAX_L = np.zeros(128, dtype=np.float32), np.zeros(128, dtype=np.float32)
for _m in range(128):
    _bits = [l for l in range(1, 8) if _m & (1 << (l - 1))]
    if _bits:
        _LANE_SUM[_m] = float(sum(_bits))
        _MIN_L[_m], _MAX_L[_m] = float(min(_bits)), float(max(_bits))
_AVG_MAP = (_LANE_SUM[_MAP] / np.maximum(_CNT[_MAP], 1.0)).astype(np.float32)
_SPAN_MAP = np.where(_CNT[_MAP] > 1, (_MAX_L - _MIN_L)[_MAP], 0.0).astype(np.float32)

_CHUNK = 1260  # permutation block size (4 blocks); caps the working set at ~10 MB


def _positions(seq: np.ndarray):
    """(time, key_mask, scratch_flag, weight) per distinct grid position.

    Weight follows Permikon: gap_i = t_{i+1} - t_i, w_i = min(min_gap / gap_i, 1)
    with the final position weighted 0. Fast notes therefore dominate the
    hand-travel metrics, which is the point (travel cost is per unit time).
    """
    t = seq[:, 0].astype(np.float64)
    lane = seq[:, 1].astype(np.int64)
    order = np.argsort(t, kind="stable")
    t, lane = t[order], lane[order]
    uniq, inv = np.unique(t, return_inverse=True)

    mask = np.zeros(len(uniq), dtype=np.int64)
    scr = np.zeros(len(uniq), dtype=bool)
    key = (lane >= 1) & (lane <= 7)
    if key.any():
        # bitwise_or.at, NOT `mask[idx] |= bit`: fancy-indexed in-place `|=` reads a
        # copy and writes back, so a position receiving several key lanes keeps only
        # the last bit written — every chord silently collapses to one note (found
        # 2026-09-11: 2071 notes / 905 positions reported max chord size 1).
        np.bitwise_or.at(mask, inv[key], 1 << (lane[key] - 1))
    if (lane == SCRATCH_LANE).any():
        scr[inv[lane == SCRATCH_LANE]] = True

    gaps = np.diff(uniq)
    gaps = np.append(gaps, 0.0)
    pos = gaps[gaps > 0]
    if len(pos) == 0:
        w = np.zeros(len(uniq), dtype=np.float64)
    else:
        g = pos.min()
        w = np.where(gaps == 0.0, 0.0, np.minimum(g / np.maximum(gaps, 1e-12), 1.0))
    return mask.astype(np.uint8), scr, w


def _summarise(name: str, per_perm: np.ndarray, out: dict) -> None:
    """mean/std/min/max over the 5040 permutations + the written chart's percentile."""
    out[f"ps_{name}_mean"] = float(per_perm.mean())
    out[f"ps_{name}_std"] = float(per_perm.std())
    out[f"ps_{name}_min"] = float(per_perm.min())
    out[f"ps_{name}_max"] = float(per_perm.max())
    out[f"ps_{name}_base_pct"] = float((per_perm < per_perm[0]).mean())


def chart_perm_stats(seq: np.ndarray) -> dict:
    out = {c: np.nan for c in PS_COLS}
    if seq is None or len(seq) < 2:
        return out
    mask, scr, w = _positions(seq)
    P = len(mask)
    if P < 3:
        return out
    W = float(w.sum())
    if W <= 0.0:
        return out
    x = np.arange(P, dtype=np.float32)
    wx, wxx = (w * x).astype(np.float32), (w * x * x).astype(np.float32)
    WX, WXX = float(wx.sum()), float(wxx.sum())
    denom = WXX - WX * WX / W

    sm = np.empty(_N_PERM, dtype=np.float32)
    ti = np.empty(_N_PERM, dtype=np.float32)
    ba = np.empty(_N_PERM, dtype=np.float32)
    sp = np.empty(_N_PERM, dtype=np.float32)

    for s in range(0, _N_PERM, _CHUNK):
        e = min(s + _CHUNK, _N_PERM)
        avg = _AVG_MAP[s:e, mask]                     # (B, P)
        span = _SPAN_MAP[s:e, mask]                   # (B, P)
        # weighted moments of the lane trajectory
        WY = avg @ w
        WXY = avg @ wx
        WYY = (avg * avg) @ w
        if denom > 1e-9:
            slope = (WXY - WX * WY / W) / denom
            inter = (WY - slope * WX) / W
            # weighted residual variance around the fitted lane drift
            rv = (WYY - 2.0 * slope * WXY - 2.0 * inter * WY
                  + slope * slope * WXX + 2.0 * slope * inter * WX
                  + inter * inter * W) / W
            sm[s:e] = np.sqrt(np.maximum(rv, 0.0))
        else:
            sm[s:e] = 0.0
        ti[s:e] = (span @ w) / W
        # spread: weighted variance of the lane position itself (vs the residual above)
        sp[s:e] = np.sqrt(np.maximum(WYY / W - (WY / W) ** 2, 0.0))
        # base: weighted mean variability = |delta avg lane| + chord span
        var = span.copy()
        d = np.abs(np.diff(avg, axis=1))
        var[:, 1:] += d
        ba[s:e] = (var @ w) / W

    for nm, arr in (("smooth", sm), ("tight", ti), ("base", ba), ("spread", sp)):
        _summarise(nm, arr, out)

    # permutation-invariant scratch interleaving (scratch is anchored, not permuted)
    out["ps_scratch_pos_frac"] = float(scr.mean())
    both = scr & (mask != 0)
    out["ps_scratch_key_frac"] = float(both.mean())
    return out


def _work(sha: str):
    p = SEQ_ROOT / f"{sha}.npy"
    if not p.exists():
        return {"sha256": sha, **{c: np.nan for c in PS_COLS}}
    try:
        return {"sha256": sha, **chart_perm_stats(np.load(p))}
    except Exception as exc:  # noqa: BLE001 - keep the batch alive, record the sha
        print(f"  ! {sha}: {exc}", flush=True)
        return {"sha256": sha, **{c: np.nan for c in PS_COLS}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(12, max(1, mp.cpu_count() - 2)))
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N charts")
    ap.add_argument("--flush-every", type=int, default=400,
                    help="persist partial progress every N charts (resumable)")
    args = ap.parse_args()

    final = DS / "chart_perm_space.parquet"
    partial = DS / "_chart_perm_space.partial.parquet"
    shas = sorted(pd.read_parquet(DS / "firstplays.parquet")["sha256"].unique())
    if args.limit:
        shas = shas[: args.limit]

    # Resume support: a cancelled run must not lose the work already done (an earlier
    # full run was killed at ~4k charts and wrote nothing). Completed shas are kept in
    # a partial file and skipped on the next invocation.
    done = {}
    if partial.exists():
        prev = pd.read_parquet(partial)
        done = {r["sha256"]: r for r in prev.to_dict("records")}
    if final.exists():
        prev = pd.read_parquet(final)
        done.update({r["sha256"]: r for r in prev.to_dict("records")})
    todo = [s for s in shas if s not in done]
    print(f"total {len(shas)} | cached {len(shas) - len(todo)} | todo {len(todo)} | "
          f"workers {args.workers}", flush=True)
    if not todo:
        print("nothing to do.")
        return

    def flush():
        df = pd.DataFrame(list(done.values())).sort_values("sha256").reset_index(drop=True)
        df.to_parquet(partial)
        return df

    buf = []
    with mp.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_work, todo, chunksize=16), 1):
            done[r["sha256"]] = r
            buf.append(r)
            if i % args.flush_every == 0:
                flush()
                print(f"  {i}/{len(todo)} (written)", flush=True)
            elif i % 200 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    out = flush()
    if args.limit and len(out) < len(sorted(
            pd.read_parquet(DS / "firstplays.parquet")["sha256"].unique())):
        print(f"limited run: {len(out)} charts cached, final not written")
        return
    out.to_parquet(final)
    if partial.exists():
        partial.unlink()
    nan = int(out[PS_COLS].isna().all(axis=1).sum())
    print(f"charts {len(out)}, all-NaN {nan} -> {final}")
    print(out[PS_COLS].describe().loc[["mean", "std", "min", "max"]].round(4).to_string())


if __name__ == "__main__":
    main()
