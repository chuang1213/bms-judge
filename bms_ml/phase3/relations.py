"""Phase 3.1: relationship statistics between the three performance targets
(first-play acc / lamp / BP) and their distributions.

Answers Phase 3.1 question 5/7: are the three targets redundant, or does acc fail
to describe player performance? Uses both Pearson and Spearman, plus conditional
spread of lamp/BP within acc bins.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bms_ml" / "output" / "phase3"
DS = OUT / "dataset"

LAMP_NAMES = {1: "FAILED", 4: "EASY", 5: "NORMAL", 6: "HARD", 7: "EXHARD",
              8: "FC", 9: "PERFECT"}


def main() -> None:
    fp = pd.read_parquet(DS / "firstplays.parquet")   # all in-table first plays (4 players)
    tr = pd.read_parquet(DS / "samples.parquet")
    samples = tr[tr["phase"] == "train"]              # descriptive stats on train phase only
    res: dict = {}

    def corr_block(d: pd.DataFrame) -> dict:
        d = d.assign(bp_ratio=d["bp"] / d["notes"])
        return {
            "pearson_acc_bp": round(float(d["acc"].corr(d["bp"], method="pearson")), 3),
            "spearman_acc_bp": round(float(d["acc"].corr(d["bp"], method="spearman")), 3),
            "spearman_acc_ratio": round(float(d["acc"].corr(d["bp_ratio"], method="spearman")), 3),
            "pearson_acc_lamp": round(float(d["acc"].corr(d["lamp"], method="pearson")), 3),
            "spearman_acc_lamp": round(float(d["acc"].corr(d["lamp"], method="spearman")), 3),
            "pearson_bp_lamp": round(float(d["bp"].corr(d["lamp"], method="pearson")), 3),
            "spearman_bp_lamp": round(float(d["bp"].corr(d["lamp"], method="spearman")), 3),
            "spearman_ratio_lamp": round(float(d["bp_ratio"].corr(d["lamp"], method="spearman")), 3),
        }

    res["corr_all"] = corr_block(fp)
    res["corr_per_player"] = {p: corr_block(g) for p, g in fp.groupby("player")}

    # ----- BP tracks: raw vs notes-normalized (see Phase 3.1 brief) -----
    fp = fp.assign(bp_ratio=fp["bp"] / fp["notes"],
                   bp_log=np.log1p(fp["bp"]),
                   bp_ratio_log=np.log1p(fp["bp"] / fp["notes"]))
    res["bp_transform"] = {
        "skew_raw": round(float(fp["bp"].skew()), 3),
        "skew_log1p_raw": round(float(fp["bp_log"].skew()), 3),
        "skew_ratio": round(float(fp["bp_ratio"].skew()), 3),
        "skew_log1p_ratio": round(float(fp["bp_ratio_log"].skew()), 3),
    }
    # cross-player comparability: same (table,level) should give similar values across
    # players if the scale is comparable; report within-level between-player spread
    def between_player_spread(d, col):
        m = d.groupby(["table", "level"])[col].median().groupby("level").agg(["std", "size"])
        m = m[m["size"] >= 3]
        return {"median_between_player_std": round(float(m["std"].median()), 4),
                "n_levels": int(len(m))}
    res["comparability_raw"] = between_player_spread(fp, "bp")
    res["comparability_ratio"] = between_player_spread(fp, "bp_ratio")

    # low-BP discreteness: BP is a small integer count in the good-player regime
    res["low_bp_discreteness"] = {
        "count_bp_le_10": int((fp["bp"] <= 10).sum()),
        "count_bp_le_30": int((fp["bp"] <= 30).sum()),
        "top_small_bp_values": {int(k): int(v) for k, v in
                                fp[fp["bp"] <= 10]["bp"].value_counts().sort_index().items()},
        "unique_ratio_values_below_1pct": int(fp.loc[fp["bp_ratio"] < 0.01, "bp_ratio"].round(6).nunique()),
        "n_rows_ratio_below_1pct": int((fp["bp_ratio"] < 0.01).sum()),
    }
    res["bp_dist_per_player"] = {
        p: {"n": len(g),
            "raw_p50_p75_p90": [float(g["bp"].quantile(q)) for q in (.5, .75, .9)],
            "ratio_p50_p75_p90": [round(float(g["bp_ratio"].quantile(q)), 4) for q in (.5, .75, .9)],
            "zero_rate": round(float((g["bp"] == 0).mean()), 4)}
        for p, g in fp.groupby("player")}
    res["bp_by_level"] = {
        str(int(l)): {"n": int(len(g)),
                      "raw_p50": float(g["bp"].median()),
                      "ratio_p50": round(float(g["bp_ratio"].median()), 4)}
        for l, g in fp.groupby("level") if len(g) >= 30}

    # lamp distribution per player
    res["lamp_dist_per_player"] = {
        p: {LAMP_NAMES.get(int(k), str(k)): round(float(v), 4)
            for k, v in g["lamp"].value_counts(normalize=True).items()}
        for p, g in fp.groupby("player")}

    # KEY: conditional spread — within narrow acc bins, how much do lamp/BP vary?
    fp2 = fp.dropna(subset=["acc"]).copy()
    fp2["acc_bin"] = pd.cut(fp2["acc"], bins=np.arange(40, 101, 5))
    cond = fp2.groupby("acc_bin", observed=True).agg(
        n=("bp", "size"),
        acc_lo=("acc", lambda s: s.quantile(.1)), acc_hi=("acc", lambda s: s.quantile(.9)),
        lamp_p10=("lamp", lambda s: s.quantile(.1)), lamp_p50=("lamp", "median"),
        lamp_p90=("lamp", lambda s: s.quantile(.9)),
        bp_p10=("bp", lambda s: s.quantile(.1)), bp_p50=("bp", "median"),
        bp_p90=("bp", lambda s: s.quantile(.9)),
        fail_rate=("lamp", lambda s: float((s == 1).mean())),
    )
    res["conditional_on_acc_bin"] = {str(k): v for k, v in cond.round(2).to_dict("index").items()}

    # per-player: within acc bin 80-85 (comparable skill slice), lamp/BP spread
    sel = fp2[(fp2["acc"] >= 80) & (fp2["acc"] < 85)]
    res["acc80_85_per_player"] = {
        p: {"n": int(len(g)),
            "lamp_dist": {LAMP_NAMES.get(int(k), str(k)): int(v)
                          for k, v in g["lamp"].value_counts().items()},
            "bp_p10_p50_p90": [float(g["bp"].quantile(q)) for q in (.1, .5, .9)],
            "ratio_p10_p50_p90": [round(float(g["bp_ratio"].quantile(q)), 4) for q in (.1, .5, .9)]}
        for p, g in sel.groupby("player")}

    json.dump(res, open(OUT / "target_relations.json", "w"), indent=2, default=str)

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (x, y, xl, yl) in zip(axes, [("acc", "bp", "acc", "BP"),
                                         ("acc", "lamp", "acc", "lamp (ordinal)"),
                                         ("bp", "lamp", "BP", "lamp (ordinal)")]):
        for p, g in fp2.groupby("player"):
            ax.scatter(g[x], g[y], s=5, alpha=0.25, label=p)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
    axes[0].legend(fontsize=8)
    fig.suptitle("first-play target relationships (in-table charts, 4 players)")
    fig.tight_layout()
    fig.savefig(OUT / "target_relations.png", dpi=140)
    plt.close(fig)

    print(json.dumps({k: res[k] for k in ["corr_all", "corr_per_player", "bp_dist_per_player",
                                          "bp_transform"]}, indent=2, default=str))
    print("->", OUT / "target_relations.json")


if __name__ == "__main__":
    main()
