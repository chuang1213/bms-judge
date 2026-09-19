"""Phase 4 step 4 — audit the cross-sectional matrix before any model is written.

Reports, per client (never pooled), the things that decide whether the matrix is
usable at all:

  1. player count and observations per player (and the distribution, not just the mean);
  2. chart coverage: how many distinct charts exist, how many are observed, matrix density;
  3. overlap: charts observed by >=2 players (the only region matrix completion can
     learn a chart effect from; if this is thin, cold-chart work is hopeless and
     that is worth knowing NOW);
  4. duplicates and missingness: true duplicate (player, sha) keys, missing notes,
     missing target, sha not in corpus;
  5. anomalies: accuracy outside [0,100], accuracy disagreeing with its denominator
     variant, notes disagreeing with the manifest, near-empty judgement counts.

Anomalies are reported as counts and fractions; nothing is silently dropped here,
because the evaluation script owns the row filter and must be able to state it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from provenance import ROOT, header, write_result  # noqa: E402

DEFAULT_PARQUET = ROOT / "bms_ml" / "output" / "phase4" / "dataset" / "cross_section.parquet"

# An observed cell with almost no judgements is not a real score observation.
MIN_JUDGEMENTS = 120


def _q(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if not len(s):
        return {}
    return {k: round(float(v), 4) for k, v in
            {"min": s.min(), "p10": s.quantile(.10), "p25": s.quantile(.25),
             "median": s.median(), "p75": s.quantile(.75), "p90": s.quantile(.90),
             "max": s.max(), "mean": s.mean(), "std": s.std()}.items()}


def audit_client(df: pd.DataFrame, client: str, n_corpus_charts: int) -> dict:
    d = df[df["client"] == client]
    players = sorted(d["player"].unique().tolist())
    n_players, n_obs = len(players), int(len(d))
    n_charts = int(d["sha256"].nunique())

    per_player = (d.groupby("player")
                  .agg(n_obs=("sha256", "size"), n_charts=("sha256", "nunique"),
                       acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                       notes_median=("notes", "median"))
                  .sort_values("n_obs", ascending=False).round(4))
    charts_per_player = d.groupby("sha256")["player"].nunique()

    out = {
        "players": n_players,
        "player_names": players,
        "observations": n_obs,
        "distinct_charts_observed": n_charts,
        "corpus_charts_available": n_corpus_charts,
        "chart_coverage_of_corpus": round(n_charts / max(n_corpus_charts, 1), 4),
        "matrix_density": round(n_obs / max(n_players * n_corpus_charts, 1), 6),
        "obs_per_player": _q(per_player["n_obs"]),
        "acc_distribution": _q(d["acc"]),
        "charts_shared_by_n_players": {
            "gte_2": int((charts_per_player >= 2).sum()),
            "gte_3": int((charts_per_player >= 3).sum()),
            "eq_1": int((charts_per_player == 1).sum()),
            "max": int(charts_per_player.max()) if len(charts_per_player) else 0,
            "median": float(charts_per_player.median()) if len(charts_per_player) else 0.0,
        },
        "frac_obs_in_shared_charts": round(
            float(d["sha256"].map(charts_per_player).ge(2).mean()), 4),
        "per_player": per_player.reset_index().to_dict("records"),
        "duplicates": {
            "duplicate_player_sha_rows": int(d.duplicated(["player", "sha256"]).sum()),
            "duplicate_client_sha_diff_player": int(
                d.duplicated(["sha256"]).sum() - max(d.duplicated(["player", "sha256"]).sum(), 0)),
        },
        "missing": {
            "acc_nan": int(d["acc"].isna().sum()),
            "notes_nan_or_zero": int((d["notes"].fillna(0) <= 0).sum()),
            "manifest_notes_nan": int(d["manifest_notes"].isna().sum()),
            "md5_nan": int(d["md5"].isna().sum()),
            "lamp_unknown": int((d["lamp_name"] == "UNKNOWN").sum()),
        },
        "anomalies": {
            "acc_gt_100": int((d["acc"] > 100).sum()),
            "acc_lt_0": int((d["acc"] < 0).sum()),
            "n_judgements_zero": int((d["n_judgements"].fillna(0) <= 0).sum()),
            "n_judgements_lt_min": int((d["n_judgements"].fillna(0) < MIN_JUDGEMENTS).sum()),
            "notes_neq_manifest": int((~d["notes_match_manifest"].fillna(False)).sum()),
            "acc_alt_vs_acc_absdiff_gt_1pp": int(
                (d["acc_alt"] - d["acc"]).abs().gt(1.0).sum()),
            "charts_with_multiple_modes_collapsed": int((d["n_modes_collapsed"] > 1).sum()),
            "quarantined_chart_rows": int(d["quarantine"].fillna("").ne("").sum()),
            "flagged_chart_rows": int(d["flags"].fillna("").ne("").sum()),
            "mode_kind_other_rows": int(d["mode_kind"].ne("sp").sum()),
        },
        "lamp_counts": {str(k): int(v) for k, v in
                        d["lamp_name"].value_counts().sort_index().items()},
        "notes_source_counts": {str(k): int(v) for k, v in
                                d["notes_source"].value_counts().items()},
    }
    if client == "beatoraja":
        out["playcount"] = _q(d["playcount"])
        out["bp"] = _q(d["bp"])
    return out


def audit(parquet: Path) -> dict:
    df = pd.read_parquet(parquet)
    n_corpus = int(pd.read_parquet(parquet)["sha256"].nunique()) if False else None
    # corpus size from the build report, so coverage means "share of the 7-key universe"
    import json
    brep = json.load(open(ROOT / "bms_ml/output/phase4/build_cross_section.json", encoding="utf-8"))
    # canonical 7-key chart count = unique sha256 in the chart table; recompute cheaply
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_cross_section import load_manifest  # noqa: E402
    charts = load_manifest()
    n_corpus = int(len(charts))

    report = {"cross_section_rows": int(len(df)),
              "parquet": str(parquet.relative_to(ROOT)),
              "min_judgements_threshold": MIN_JUDGEMENTS,
              "build_report_rows": brep.get("observations_built"),
              "by_client": {}}
    for client in sorted(df["client"].unique()):
        report["by_client"][client] = audit_client(df, client, n_corpus)

    # cross-client overlap: same sha observed in both clients (never pooled, but
    # the overlap is what makes a client-effect comparison possible at all)
    both = df.groupby("sha256")["client"].nunique()
    report["cross_client_overlap"] = {
        "charts_in_both_clients": int((both >= 2).sum()),
        "charts_lr2_only": int(((both == 1) & (df.groupby("sha256")["client"].first() == "lr2")).sum()),
        "charts_beatoraja_only": int(((both == 1) & (df.groupby("sha256")["client"].first() == "beatoraja")).sum()),
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 step 4: audit cross_section.parquet")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    args = ap.parse_args()

    rep = audit(args.parquet)
    cfg = {"parquet": str(args.parquet.relative_to(ROOT)),
           "min_judgements_threshold": MIN_JUDGEMENTS,
           "clients_reported_separately": True}
    out = write_result("audit_cross_section.json",
                       header(script="audit_cross_section.py", data_config=cfg), rep)

    for cl, r in rep["by_client"].items():
        print(f"== {cl}: players={r['players']} obs={r['observations']} "
              f"charts={r['distinct_charts_observed']} "
              f"coverage={r['chart_coverage_of_corpus']:.3f} density={r['matrix_density']}")
        print(f"   obs/player: {r['obs_per_player']}")
        print(f"   shared charts >=2 players: {r['charts_shared_by_n_players']['gte_2']} "
              f"(max {r['charts_shared_by_n_players']['max']}); "
              f"{r['frac_obs_in_shared_charts']:.1%} of observations live in them")
        print(f"   acc: {r['acc_distribution']}")
        print(f"   anomalies: {r['anomalies']}")
        print(f"   missing: {r['missing']}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
