"""Step 3/3 of the MinaCalc MSD build: JSONL -> parquet + external validity check.

The whole point of running MinaCalc is that it is crowd-calibrated, so the first thing
to verify is exactly that: MSD Overall should order charts by difficulty-table level far
better than our raw c_avg_nps does. If it does not, something in the conversion (row
grouping, keycount, seconds) is wrong and the features must not be used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[2]
MSD = ROOT / "bms_ml" / "output" / "phase3" / "msd"
DS = ROOT / "bms_ml" / "output" / "phase3" / "dataset"
OUT = DS / "msd.parquet"
SKILLSETS = ["Overall", "Stream", "Jumpstream", "Handstream", "Stamina",
             "JackSpeed", "Chordjack", "Technical"]


def main() -> None:
    rows = []
    with open(MSD / "msd.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if "values" in r:
                rows.append({"sha256": r["sha256"],
                             **{f"msd_{k.lower()}": r["values"][k] for k in SKILLSETS}})
    df = pd.DataFrame(rows)
    # Technical is a 4K-only skillset in MinaCalc: on the generic n-key path (every one
    # of our charts is 8 columns) 0.74.0 returns a constant 0.18, i.e. zero information.
    # Drop it rather than let a dead column masquerade as a feature downstream.
    tc = df["msd_technical"]
    if tc.nunique() <= 2:
        print(f"dropping msd_technical: constant on the n-key path "
              f"(values {sorted(tc.unique())[:3]})")
        df = df.drop(columns=["msd_technical"])
    assert not df.duplicated("sha256").any()
    df.to_parquet(OUT)
    print(f"saved {len(df)} charts -> {OUT}")
    msd_cols = [c for c in df.columns if c.startswith("msd_")]
    print(df[msd_cols].describe().loc[["mean", "std", "min", "max"]].round(2).T.to_string())

    # ---- external validity: does MSD order charts by table level? ---------------
    sys.path.insert(0, str(ROOT / "bms_ml" / "phase3"))
    from data import load_tables
    s = (load_samples_safe()
         .drop(columns=["table", "level"], errors="ignore")   # samples carries its own
         .merge(load_tables(), on="sha256", how="left")
         .merge(df, on="sha256", how="inner"))
    s = s[s["table"].notna() & s["notes"].notna() & (s["notes"] > 0)].copy()
    s = s[s["bp"] <= s["notes"] + 5]
    print(f"\nplayed charts with MSD and a table: {len(s)}")
    print(f"{'table':10}{'n':>6}{'msd_overall':>26}{'c_avg_nps':>22}")
    for t, g in s.groupby("table"):
        r1 = sps.spearmanr(g["msd_overall"], g["level"]).statistic
        r2 = sps.spearmanr(g["c_avg_nps"], g["level"]).statistic
        q = g["msd_overall"].quantile([0, 0.5, 1]).values
        print(f"{t:10}{len(g):6d}{'rho=%.3f' % r1:>14}"
              f"  msd[{q[0]:.1f},{q[1]:.1f},{q[2]:.1f}]{'rho=%.3f' % r2:>14}")
    # same check per skillset on the hardest table (where difficulty ordering is sharp)
    hard = s[s["table"] == s["table"].value_counts().idxmax()]
    print("\nper-skillset Spearman vs level (most common table):")
    for c in [c for c in s.columns if c.startswith("msd_")]:
        r = sps.spearmanr(hard[c], hard["level"]).statistic
        print(f"  {c:16}{r:+.3f}")


def load_samples_safe() -> pd.DataFrame:
    from common import load_samples
    return load_samples()


if __name__ == "__main__":
    main()
