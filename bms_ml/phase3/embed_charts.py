"""Phase 3.2 (C3): per-chart Phase2A representation.

Embeds every chart that appears in firstplays.parquet with the Phase 2A T1-pretrained
GridEncoder (output/phase2a/t1_hold.pt): the chart's note sequence is cut into
non-overlapping 4s windows (capped at 48, evenly spaced), each window is embedded to
the 64-dim global vector, and window embeddings are mean-pooled into a chart vector.

Output: bms_ml/output/phase3/dataset/chart_repr_t1.parquet (sha256, r0..r63, n_windows).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bms_ml.phase2a.grid_data import SeqCache, pick_starts, seq_grid, window_starts  # noqa: E402
from bms_ml.phase2a.models import GridEncoder  # noqa: E402

DS = ROOT / "bms_ml" / "output" / "phase3" / "dataset"
SEQ_ROOT = ROOT / "bms_ml" / "output" / "corpus" / "sequences"
CKPT = ROOT / "bms_ml" / "output" / "phase2a" / "t1_hold.pt"
MAX_WINDOWS = 48
BATCH = 256


def main() -> None:
    fp = pd.read_parquet(DS / "firstplays.parquet")
    durations = {}
    with open(ROOT / "bms_ml" / "output" / "corpus" / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            durations[r["sha256"]] = r["meta"]["duration_sec"]

    shas = sorted(fp["sha256"].unique())
    have = [s for s in shas if (SEQ_ROOT / f"{s}.npy").exists()]
    print(f"charts needed {len(shas)}, with sequence npy {len(have)}")

    enc = GridEncoder()
    enc.load_state_dict(torch.load(CKPT, map_location="cpu"))
    enc.eval()

    cache = SeqCache(str(SEQ_ROOT))
    rows, done = [], 0
    with torch.no_grad():
        for sha in have:
            dur = float(durations.get(sha, 0.0))
            starts = window_starts(dur)
            if not starts:
                starts = [0.0]
            starts = pick_starts(starts, MAX_WINDOWS, np.random.RandomState(0))
            grids = np.stack([seq_grid(cache(sha), t0) for t0 in starts])
            tens = torch.from_numpy(grids).permute(0, 3, 1, 2).float()
            embs = [enc.pool(tens[i:i + BATCH]).numpy() for i in range(0, len(tens), BATCH)]
            rows.append([sha, len(starts)] + np.concatenate(embs).mean(axis=0).tolist())
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(have)}")

    out = pd.DataFrame(rows, columns=["sha256", "n_windows"] +
                       [f"r{i}" for i in range(64)])
    out.to_parquet(DS / "chart_repr_t1.parquet")
    print("saved", len(out), "->", DS / "chart_repr_t1.parquet")


if __name__ == "__main__":
    main()
