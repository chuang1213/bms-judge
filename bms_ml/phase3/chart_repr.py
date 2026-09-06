"""Phase 3.3: canonical chart-representation & feature registry.

Three tiers, with an explicit contract:

1. difficulty-table information  -> EXTERNAL REFERENCE + SCOPE FENCE only.
   The sl/st/発狂2018 union bounds which charts are eligible targets (quality fence,
   user decision 2026-09-03; see PROTOCOL.md §1) and serves coverage-audit
   coordinates. Table levels MUST NOT appear in any training feature list —
   there is no API here that returns them.
2. objective chart statistics    -> the strong baseline encoder (27 dims = 26
   parsing statistics + c_jrank; no community input).
3. Phase2A representation        -> learned-encoder candidate (64-dim T1 pooled
   windows, built by embed_charts.py).

A new chart encoder joins by adding a loader here and being evaluated through the
same downstream task (see PROTOCOL.md) — never by redefining the task.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "bms_ml" / "output" / "phase3" / "dataset"

OBJECTIVE_STAT_COLS = [
    "c_total_notes", "c_ln_ratio", "c_duration_sec", "c_measures", "c_initial_bpm",
    "c_min_bpm", "c_max_bpm", "c_bpm_change_count", "c_stop_count", "c_stop_total_sec",
    "c_lane0_scratch", "c_lane1", "c_lane2", "c_lane3", "c_lane4", "c_lane5",
    "c_lane6", "c_lane7", "c_scratch_ratio", "c_avg_nps", "c_peak_nps_1s",
    "c_peak_measure_nps", "c_chord_count", "c_chord2_count", "c_chord3plus_count",
    "c_jack_count",
    # #RANK judge-window tier (added 2026-09-05; parser default 2 when #RANK absent).
    # Without it, tight-rank charts have their acc over-predicted by 5-11pp
    # (see EXPERIMENT_LOG 2026-09-05).
    "c_jrank",
]

HISTORY_FEATURES = [
    "h_knn_acc", "h_n_firstplays", "h_acc_mean", "h_acc_std", "h_acc_last10",
    "h_bp_mean", "h_bp_ratio_mean", "h_fail_rate", "h_fc_rate",
    "h_days_since_active", "h_plays_last30d", "h_days_span",
]

# Recency / calendar terms. In protocol training these come from the DENSE scorelog
# row stream; in few-shot evaluation they can only come from the SPARSE first-play
# prefix, so their distributions are incomparable (evaluation values land beyond
# training p99 — this once faked a collapse of the M2 few-shot curve). Few-shot
# schemas drop them (train and eval, same schema). See PHASE3_4_TRANSFER.md appendix.
HISTORY_TIME_FEATURES = ["h_days_since_active", "h_plays_last30d", "h_days_span"]
HISTORY_FEW_FEATURES = [c for c in HISTORY_FEATURES if c not in HISTORY_TIME_FEATURES]

# v2 chart statistics (2026-09-07): threshold-free, pattern-vocabulary-free
# distribution stats of the raw onset data. The v1 jack/chord counts are
# community-conventional categories (threshold-dependent, and largely projections
# of density — a chord is density at an instant, a jack is density on one lane);
# v2 replaces the vocabulary with distribution percentiles/shapes. Built by
# chart_stats_v2.py from the corpus note sequences; sequence coverage 99.3%,
# the rest impute to the train median downstream.
OBJECTIVE_V2_COLS = [
    "v2_ioi_lane_p05", "v2_ioi_lane_p25", "v2_ioi_lane_mean",
    "v2_ioi_scratch_p05", "v2_nps_std", "v2_nps_p90",
    "v2_simul_max", "v2_simul_std", "v2_lane_entropy",
    "v2_hand_balance", "v2_scratch_nps", "v2_ln_mean_dur",
]


def load_phase2a(shas) -> pd.DataFrame:
    """Phase2A T1 chart representations (64-dim) for the requested sha256 set."""
    reps = pd.read_parquet(DS / "chart_repr_t1.parquet")
    return reps[reps["sha256"].isin(set(shas))]


def load_v2(shas) -> pd.DataFrame:
    """v2 threshold-free distribution stats (see OBJECTIVE_V2_COLS) for the sha256 set."""
    reps = pd.read_parquet(DS / "chart_stats_v2.parquet")
    return reps[reps["sha256"].isin(set(shas))]


def chart_encoder_registry() -> dict:
    """Encoder name -> (feature builder description, dim). New encoders register here
    and are compared on the same downstream task (PROTOCOL.md section 3)."""
    return {
        "objective_stats": ("26-dim parsing statistics + c_jrank (#RANK window tier)",
                            27),
        "objective_stats_v2": ("v1 (27) + 12 threshold-free distribution stats: "
                               "per-lane IOI percentiles, density variability, "
                               "simultaneity shape, lane entropy/hand balance",
                            27 + len(OBJECTIVE_V2_COLS)),
        "phase2a_t1_pooled": ("64-dim mean-pooled T1 GridEncoder windows", 64),
    }


def feature_manifest(*feature_lists: list[str]) -> dict:
    """Record the exact input features of an experiment into its results JSON."""
    return {"feature_lists": [list(f) for f in feature_lists],
            "uses_difficulty_table_features": False}
