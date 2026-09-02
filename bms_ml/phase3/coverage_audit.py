"""Phase 3.3 Player Coverage Audit.

SL/ST levels are used ONLY as a difficulty coordinate for describing coverage —
they are never model features or targets (see PROTOCOL.md).

Coordinate: SL = satellite level (0-12); ST = stella level mapped to 12+level
(ST0 ~ SL12-13 boundary); 発狂-only charts and charts with no table membership
get their own buckets. Based on first-play events (the Phase 3 sample unit).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"
PLAYERS = ["chuang", "muiclac", "tzh", "nanji"]


def coordinate(row) -> str:
    t, lv = row["table"], row["level"]
    if t == "satellite":
        if lv <= 2: return "SL0-2"
        if lv <= 5: return "SL3-5"
        if lv <= 8: return "SL6-8"
        return "SL9-12"
    if t == "stella":
        if lv <= 3: return "ST0-3 (~SL13-15)"
        if lv <= 7: return "ST4-7 (~SL16-19)"
        if lv <= 12: return "ST8-12 (~SL20-24)"
        return "ST13+"
    if t == "insane":
        return "発狂(无SLST坐标)"
    return "无坐标(表外)"


BINS = ["SL0-2", "SL3-5", "SL6-8", "SL9-12",
        "ST0-3 (~SL13-15)", "ST4-7 (~SL16-19)", "ST8-12 (~SL20-24)", "ST13+",
        "発狂(无SLST坐标)", "无坐标(表外)"]


def main() -> None:
    fp = pd.read_parquet(DS / "firstplays.parquet")
    fp["coord"] = fp.apply(coordinate, axis=1)

    # --- per player x bin first-play counts ---
    counts = fp.groupby(["player", "coord"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=PLAYERS, columns=BINS, fill_value=0)
    print("first-play counts per (player x difficulty coordinate):")
    print(counts.to_string())

    # --- charts first-played by >=2 players, per bin ---
    n_players_per_chart = fp.groupby("sha256")["player"].nunique()
    fp["n_players"] = fp["sha256"].map(n_players_per_chart)
    multi = fp[fp["n_players"] >= 2]
    multi_bin = multi.groupby("coord").agg(
        charts=("sha256", "nunique"), rows=("sha256", "size")).reindex(BINS, fill_value=0)
    four = fp[fp["n_players"] == 4].groupby("coord")["sha256"].nunique().reindex(BINS, fill_value=0)
    multi_bin["charts_4player"] = four
    print("\nco-first-played charts (>=2 players) per bin:")
    print(multi_bin.to_string())

    # --- pairwise overlap per bin (co-played chart counts) ---
    pair = {}
    for i, a in enumerate(PLAYERS):
        for b in PLAYERS[i + 1:]:
            sa = set(fp[fp["player"] == a]["sha256"])
            sb = set(fp[fp["player"] == b]["sha256"])
            pair[f"{a}&{b}"] = len(sa & sb)
    # pairwise overlap restricted to charts that exist in a bin
    pair_bin = {}
    for b in BINS:
        sub = fp[fp["coord"] == b]
        charts = sub.groupby("sha256")["player"].apply(set)
        row = {}
        for i, a in enumerate(PLAYERS):
            for c in PLAYERS[i + 1:]:
                row[f"{a[:2]}&{c[:2]}"] = sum(1 for s in charts if {a, c} <= s)
        pair_bin[b] = row
    pair_bin = pd.DataFrame(pair_bin).T

    # --- interaction identification condition: charts with >=2 players AND bin mass ---
    report = {
        "per_player_bin_counts": counts.to_dict(),
        "per_player_total_firstplays": fp.groupby("player").size().to_dict(),
        "co_firstplay_bins": multi_bin.to_dict(),
        "pairwise_overlap_charts": pair,
        "pairwise_overlap_by_bin": pair_bin.to_dict(),
        "charts_firstplayed_by_4players": int((n_players_per_chart == 4).sum()),
        "charts_firstplayed_by_3plus": int((n_players_per_chart >= 3).sum()),
        "charts_firstplayed_by_2plus": int((n_players_per_chart >= 2).sum()),
        "total_unique_charts": int(fp["sha256"].nunique()),
    }

    # sparsity verdict per bin (interaction needs co-played charts; training needs mass)
    sparsity = {}
    for b in BINS:
        tot = int(counts[b].sum())
        sparsity[b] = {
            "total_firstplays": tot,
            "co_played_charts": int(multi_bin.loc[b, "charts"]) if b in multi_bin.index else 0,
            "verdict": ("ok" if tot >= 300 and multi_bin.loc[b, "charts"] >= 100 else
                        "sparse" if tot >= 80 else "very sparse"),
        }
    report["bin_verdicts"] = sparsity

    json.dump(report, open(OUT / "coverage_audit.json", "w"), indent=2, default=str)

    # plot: stacked coverage per bin
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    bottom = np.zeros(len(BINS))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]
    for p, c in zip(PLAYERS, colors):
        vals = counts.loc[p].values.astype(float)
        ax.bar(BINS, vals, bottom=bottom, label=p, color=c, alpha=0.85)
        bottom += vals
    ax2 = ax.twinx()
    ax2.plot(BINS, multi_bin["charts"].values, "k-o", ms=4, lw=1, label="co-played charts (>=2p)")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.set_ylabel("first plays"); ax2.set_ylabel("co-played charts")
    ax.set_title("player coverage by difficulty coordinate (SL/ST as coordinate only)")
    ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "coverage_audit.png", dpi=140)
    print("saved ->", OUT / "coverage_audit.json", "|", OUT / "coverage_audit.png")


if __name__ == "__main__":
    main()
