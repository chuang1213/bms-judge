"""Phase 3.3: canonical chart-representation & feature registry.

Three tiers, with an explicit contract:

1. difficulty-table information  -> EXTERNAL REFERENCE + SCOPE FENCE only.
   The sl/st/発狂2018 union bounds which charts are eligible targets (quality fence,
   user decision 2026-09-03; see PROTOCOL.md §1) and serves coverage-audit
   coordinates. Table levels MUST NOT appear in any training feature list —
   there is no API here that returns them.
2. objective chart statistics    -> the strong baseline encoder (26 dims from
   bms_ml parsing; no community input).
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
]

HISTORY_FEATURES = [
    "h_knn_acc", "h_n_firstplays", "h_acc_mean", "h_acc_std", "h_acc_last10",
    "h_bp_mean", "h_bp_ratio_mean", "h_fail_rate", "h_fc_rate",
    "h_days_since_active", "h_plays_last30d", "h_days_span",
]


def load_phase2a(shas) -> pd.DataFrame:
    """Phase2A T1 chart representations (64-dim) for the requested sha256 set."""
    reps = pd.read_parquet(DS / "chart_repr_t1.parquet")
    return reps[reps["sha256"].isin(set(shas))]


def chart_encoder_registry() -> dict:
    """Encoder name -> (feature builder description, dim). New encoders register here
    and are compared on the same downstream task (PROTOCOL.md section 3)."""
    return {
        "objective_stats": ("26-dim parsing statistics + c_jrank (#RANK window tier)",
                            27),
        "phase2a_t1_pooled": ("64-dim mean-pooled T1 GridEncoder windows", 64),
    }


def feature_manifest(*feature_lists: list[str]) -> dict:
    """Record the exact input features of an experiment into its results JSON."""
    return {"feature_lists": [list(f) for f in feature_lists],
            "uses_difficulty_table_features": False}
