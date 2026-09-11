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

# Per-axis PERSONAL RESPONSE PROFILE (2026-09-11, PHASE3_5_REVIEW §4.3 follow-up).
# The only chart-conditioned player feature before this was `h_knn_acc`; these give
# each player an explicit univariate response curve per chart axis, fitted on their
# strictly-prior plays (see history_response.py). `h_resp_*` is chart-conditioned
# (uses the target's own axis value); `h_slope_*` is a pure player trait.
# Axis -> v1 column. Chosen to span the pressure types the framework paper names
# (density / LN / scratch / length / chord / same-lane repetition) WITHOUT importing
# its 7-axis vocabulary — these are plain objective columns, not skill labels.
RESPONSE_AXES = {
    "nps": "c_avg_nps",
    "ln": "c_ln_ratio",
    "scratch": "c_scratch_ratio",
    "dur": "c_duration_sec",
    "chord": "c_chord3plus_count",
    "jack": "c_jack_count",
    # v2 threshold-free axes (added 2026-09-11): same-lane speed (the window-free
    # analogue of jack), density variability, simultaneity and lane spread.
    "ioi": "v2_ioi_lane_p05",
    "npsstd": "v2_nps_std",
    "simul": "v2_simul_max",
    "entropy": "v2_lane_entropy",
    # judge-window tier: NOT a difficulty property but a scoring setting, and the
    # history study showed rank1/rank2 charts are mis-scored by 5-11pp without it.
    # A player's slope here is "how much do tight windows cost me" - a real trait.
    "jrank": "c_jrank",
}
def _response_cols(prefix: str) -> list[str]:
    """Column names of one response block. `h_` = the acc block (shipped 2026-09-11);
    `l_` = the lamp block (2026-09-11 evening): lamp is the one target the acc-response
    profile slightly hurt (B_full lamp 1.331 vs B 1.325), and a player's per-axis
    survival profile is not the same object as their per-axis accuracy profile."""
    return ([f"{prefix}resp_{k}" for k in RESPONSE_AXES]
            + [f"{prefix}slope_{k}" for k in RESPONSE_AXES]
            + [f"{prefix}resp_mean", f"{prefix}resp_std"])


HISTORY_RESPONSE_COLS = _response_cols("h_")            # acc response (protocol best)
HISTORY_RESPONSE_LAMP_COLS = _response_cols("l_")       # lamp response
# BP response, on log1p(bp) like every BP model in the project (skew 2.56 -> 0.15).
# BP improved by 5.2 MAE when the lamp block was added, so the target is responsive to
# this family of features and deserves its own block.
HISTORY_RESPONSE_BP_COLS = _response_cols("b_")


def _dev_cols(prefix: str) -> list[str]:
    """Player-relative axis deviation: (x_target - mean_player_history) / sd_player_history.

    `resp_` is the player's LINEAR prediction at this chart's axis value, so it can only
    express a straight line. `dev_` says how far outside the player's OWN usual range the
    chart sits; a tree splitting on it places thresholds in player-relative units, which
    is exactly the mechanism for a saturating / cliff response that the linear profile
    cannot represent. Same window statistics as the response block, so it is nearly free.
    """
    return [f"{prefix}dev_{k}" for k in RESPONSE_AXES]


HISTORY_RESPONSE_DEV_COLS = _dev_cols("h_")             # acc block deviation
HISTORY_LAMP_DEV_COLS = _dev_cols("l_")
HISTORY_BP_DEV_COLS = _dev_cols("b_")

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
    # window-free density extremes: percentiles of ALL-onset inter-onset
    # intervals. These replace the arbitrary 1-second window of v1's
    # peak_nps_1s (any window choice is a convention; percentiles are not).
    "v2_ioi_global_p05", "v2_ioi_global_p25",
]

# Known convention-dependence audit (2026-09-07, user prompt: "there may be other
# human-made metrics beyond jack/chord"):
#   v1 peak_nps_1s      -> 1s window is arbitrary; v2_ioi_global_* is the window-free analogue
#   v1 jack/chord counts-> threshold/categorisation conventions (kept: they still carry
#                          unique signal, see chart_v2_eval B_no_cj)
#   v2 hand_balance     -> assumes the physical 7-key layout (lanes 1-3 left, 5-7 right);
#                          a human-factor assumption, NOT file structure
#   v2 nps_std/p90      -> 1s bins (same convention as v1 avg/peak, but distribution-level)
#   everything else     -> file-objective (LN via LNTYPE/LNOBJ, STOP via spec, lane
#                          0 = channel 16 scratch, grid positions)

# Permutation-space hand-travel geometry (2026-09-11, user prompt: "the target
# chart's statistics are probably not the best choice ... there is no truly
# objective BMS description, hand-craft a new representation"). Built by
# chart_perm_space.py; inspired by Permikon (Permikon-main/), which evaluates all
# 5040 key-lane permutations instead of pretending the written arrangement is THE
# chart. We keep absolute cross-chart-comparable values (no Permikon min-max
# normalisation) and summarise each metric over the whole permutation space.
#
# Convention dependence, declared (same spirit as the v2 audit above):
#   lane 1..7 as a 1-D spatial coordinate -> physical 7-key layout assumption
#     (human factor, NOT file structure) -- identical in kind to v2 hand_balance
#   scratch anchored (not permuted)      -> follows Permikon; BMS RANDOM does
#     remap scratch, so ps_scratch_* are the only scratch-aware terms here
#   1/60s-agnostic: positions are exact grid timestamps (file-objective)
#   weight = min(min_gap/gap, 1)         -> Permikon's "faster notes matter more"
OBJECTIVE_PERM_COLS = [
    "ps_smooth_mean", "ps_smooth_std", "ps_smooth_min", "ps_smooth_max",
    "ps_smooth_base_pct",
    "ps_tight_mean", "ps_tight_std", "ps_tight_min", "ps_tight_max",
    "ps_tight_base_pct",
    "ps_base_mean", "ps_base_std", "ps_base_min", "ps_base_max",
    "ps_base_base_pct",
    "ps_spread_mean", "ps_spread_std", "ps_spread_min", "ps_spread_max",
    "ps_spread_base_pct",
    # permutation-invariant scratch interleaving (scratch is anchored)
    "ps_scratch_pos_frac", "ps_scratch_key_frac",
]


def load_phase2a(shas) -> pd.DataFrame:
    """Phase2A T1 chart representations (64-dim) for the requested sha256 set."""
    reps = pd.read_parquet(DS / "chart_repr_t1.parquet")
    return reps[reps["sha256"].isin(set(shas))]


def load_v2(shas) -> pd.DataFrame:
    """v2 threshold-free distribution stats (see OBJECTIVE_V2_COLS) for the sha256 set."""
    reps = pd.read_parquet(DS / "chart_stats_v2.parquet")
    return reps[reps["sha256"].isin(set(shas))]


def load_perm_space(shas) -> pd.DataFrame:
    """Permutation-space geometry stats (see OBJECTIVE_PERM_COLS) for the sha256 set."""
    reps = pd.read_parquet(DS / "chart_perm_space.parquet")
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
        "perm_space": ("hand-travel geometry summarised over all 5040 key-lane "
                       "permutations (smooth/tight/base/spread mean,std,min,max + "
                       "written-arrangement percentile + scratch interleaving); "
                       "lane 1..7 treated as a 1-D spatial axis (human-factor "
                       "assumption, declared)",
                       len(OBJECTIVE_PERM_COLS)),
        "phase2a_t1_pooled": ("64-dim mean-pooled T1 GridEncoder windows", 64),
    }


def feature_manifest(*feature_lists: list[str]) -> dict:
    """Record the exact input features of an experiment into its results JSON."""
    return {"feature_lists": [list(f) for f in feature_lists],
            "uses_difficulty_table_features": False}
